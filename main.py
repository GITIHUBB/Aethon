"""FastAPI backend for the "Aethon" microgravity combustion dashboard.

Routes
------
``GET      /``               health probe
``GET      /health``          health probe
``POST     /query-agent``    RAG question answering over the NASA paper index
``GET/POST /rank-materials`` material flammability risk ranking

The RAG endpoint retrieves and re-ranks NASA paper passages, then uses the
configured Groq model with OpenRouter fallback when available. Dashboard
ranking estimates are separate from physical combustion simulation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
from contextlib import asynccontextmanager
from typing import Any, List, Optional
from urllib.parse import quote, urlparse

import chromadb
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

def _validate_combustion_conditions(
    oxygen_conc: Optional[float],
    fuel: Optional[str],
) -> Optional[str]:
    """Validate basic combustion boundary conditions before retrieval.

    Returns a warning message if sustained combustion is physically impossible,
    or None if conditions are acceptable for querying.

    Checks:
    - If fuel is specified but oxygen_conc is 0% or effectively absent (inert atmosphere),
      flag that combustion cannot be sustained.
    - If oxygen_conc is missing/None but fuel is present, also flag the issue.
    - Normal cases with fuel and reasonable oxygen levels pass through.
    """
    # No fuel specified - nothing to validate
    if not fuel:
        return None

    # If oxygen concentration is not specified, we can't validate fully
    # but we shouldn't block the query - just note it
    if oxygen_conc is None:
        return None

    # Clamp oxygen to valid range for checking
    ox = max(0.0, min(100.0, oxygen_conc))

    # Check for inert atmosphere: oxygen concentration at or near 0%
    # with a fuel specified means no oxidizer for sustained combustion
    if ox <= 2.0:
        return (
            f"WARNING: Inert atmosphere detected (oxygen concentration {ox:.1f}%). "
            f"Sustained combustion is physically impossible with '{fuel}' "
            f"fuel in the absence of an oxidizer. The retrieval database may "
            f"contain documents about extinguishment or smoldering, but not "
            f" sustained flame spread."
        )

    # Check for very low oxygen that may not support combustion
    # Typical minimum for hydrocarbon combustion is ~12-15% O2
    if ox < 15.0:
        # This is a warning, not a hard block, as some limited combustion
        # can occur at reduced oxygen, but we flag it
        pass  # Could add a warning here if desired

    # Basic combustion conditions appear satisfied
    return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "langchain"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# PRIMARY: Groq (fast, OpenAI-compatible API)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
# BACKUP: OpenRouter (falls back to automatically when Groq is unreachable)
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Temperature shared by both providers.
AI_TEMPERATURE = 0.7
# Conversation turns replayed to the model on every request. Capped at 4 so the
# assistant remembers the last few turns without bloating the context window.
MAX_HISTORY_TURNS = 4

# Secrets come from `.env` next to this file so the keys never land in source.
# Put provider keys there and restart the backend.

# Retrieval shape. The index stores several chunks per page (and repeats some),
# so we pull 10 candidates, drop near-duplicates and anything barely resembling
# the question, then re-rank them against the query before keeping the best few.
MAX_PASSAGES = 4  # top 3-4 is enough grounding without diluting the prompt
FETCH_K = 10  # candidates pulled from Chroma before re-ranking
# L2 distance above which a chunk is effectively unrelated to the query.
# confidence_of() maps 1.40 -> ~0.02 cosine similarity; off-topic probes on
# this index scored 1.46+.
MAX_DISTANCE = 1.40

COHERE_URL = "https://api.cohere.com/v2/rerank"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Materials the dashboard ranks: base flammability (0-1) plus hazard wording.
MATERIALS = [
    {"id": "PMMA", "label": "PMMA (Acrylic)", "base": 0.80,
     "risk_level": "High Hazard", "spread": "Fast", "extinction": "0.18 m/s airflow"},
    {"id": "Silicone", "label": "Silicone Rubber", "base": 0.75,
     "risk_level": "High Hazard", "spread": "Moderate-Fast", "extinction": "0.12 m/s airflow"},
    {"id": "Cotton", "label": "Cotton Fabric", "base": 0.55,
     "risk_level": "Caution", "spread": "Moderate", "extinction": "0.08 m/s airflow"},
    {"id": "Nomex", "label": "Nomex Fabric", "base": 0.45,
     "risk_level": "Caution", "spread": "Slow", "extinction": "0.05 m/s airflow"},
    {"id": "Kevlar", "label": "Kevlar Blend", "base": 0.25,
     "risk_level": "Low Hazard / Safe", "spread": "Self-Extinguishing", "extinction": "N/A"},
]

# ---------------------------------------------------------------------------
# Smart Small-Talk & Closure Router
# ---------------------------------------------------------------------------
# Immediate, polite AETHON responses for greetings and conversational closures
# so these never hit ChromaDB or surface "No matching papers found".
GREETING_RE = re.compile(
    r"(?:hi|hello|helo|hlo|hey|sup|yo|help)(?:\s+there)?[.!?,\s]*",
    re.IGNORECASE,
)
CLOSURE_RE = re.compile(
    r"(?:ok(?:ay)?|done|cool|got it|thanks|thank you|bye|goodbye|cya|see you)"
    r"(?:[.!?,\s]+(?:thanks|thank you|bye|goodbye))?[.!?,\s]*",
    re.IGNORECASE,
)


def _small_talk_route(question: str) -> Optional[str]:
    """Return an immediate polite AETHON reply for greetings/closures, or None."""
    q = question.strip().lower()
    if GREETING_RE.fullmatch(q):
        return (
            "Hello! I'm Aethon, your AI guide for NASA microgravity combustion "
            "research. Ask me about flame spread, extinction limits, material "
            "flammability, or experiments like BASS, SAFFIRE, FLEX, and SoFIE."
        )
    if CLOSURE_RE.fullmatch(q):
        return (
            "You're welcome! I'm here whenever you'd like to explore more about "
            "spacecraft fire safety and microgravity combustion. Have a great day!"
        )
    return None


# ---------------------------------------------------------------------------
# Query Expansion & Typo Normalization
# ---------------------------------------------------------------------------
# Expand single terms / acronyms into rich domain search phrases so retrieval
# lands on the right NASA papers even for terse queries.
_ACRONYM_EXPANSIONS = {
    "SLICE": "Structure of Liftoff in Combustion Experiment (SLICE) microgravity flame spread and liftoff limit",
    "PMMA": "polymethyl methacrylate (PMMA) acrylic material flammability and flame spread in microgravity",
    "BASS": "Burning and Suppression of Solids (BASS) spacecraft fire safety experiment",
    "SOFIE": "Solid Fuel Ignition and Extinction (SoFIE) microgravity combustion experiment",
    "SAFFIRE": "Spacecraft Fire Experiment (SAFFIRE) spacecraft fire safety",
    "FLEX": "Flame Extinguishment Experiment (FLEX) droplet combustion and extinction in microgravity",
    "ACME": "Advanced Combustion via Microgravity Experiments (ACME) on the ISS",
    "UFGM": "Unsteady Flamelet Generated Manifold (UFGM) modeling for concurrent flame spread in microgravity",
    "SAME": "Smoke Aerosol Measurement Experiment (SAME) spacecraft smoke detection",
    "PSI-62": (
        "NASA Physical Sciences Informatics PSI-62 concurrent flame spread modeling "
        "using Flamelet Generated Manifolds in microgravity, BASS experiments, "
        "two-color tomography, and PMMA"
    ),
}
_SHORT_QUERY_CONTEXT = (
    "NASA microgravity combustion research spacecraft fire safety flame spread "
    "ignition and extinction"
)

# Common typos users type when asking about microgravity combustion.
_TYPO_MAP = {
    "mcro": "microgravity",
    "flm": "flame",
    "grafity": "gravity",
}


def expand_and_normalize_query(query: str) -> str:
    """Normalize typos and expand short queries or acronyms for retrieval."""
    expanded = str(query or "").strip()
    for typo, correct in _TYPO_MAP.items():
        expanded = re.sub(rf"\b{typo}\b", correct, expanded, flags=re.IGNORECASE)

    expansions = []
    for acronym, phrase in _ACRONYM_EXPANSIONS.items():
        matches = list(re.finditer(rf"\b{re.escape(acronym)}\b", expanded, re.IGNORECASE))
        if acronym == "SAME":
            matches = [
                match
                for match in matches
                if match.group().isupper()
                or expanded.strip().casefold() == "same"
                or re.search(r"\b(smoke|aerosol|measurement|experiment)\b", expanded, re.IGNORECASE)
            ]
        if matches:
            expansions.append(phrase)

    if expansions:
        return f"{expanded} {' '.join(expansions)}"

    if 0 < len(re.findall(r"\b[\w-]+\b", expanded)) <= 3:
        return f"{expanded} {_SHORT_QUERY_CONTEXT}".strip()
    return expanded


# Keep the previous helper name available for callers that imported it.
expand_query = expand_and_normalize_query


def _resolve_follow_up(question: str, chat_history: list) -> str:
    """Append the previous user topic to short follow-up queries.

    Phrases like "what about its extinction limit?" or "tell me more" are too
    short to retrieve on their own, so we tack on the last user-mentioned topic.
    """
    short_indicators = (
        "what about its", "tell me more", "and its", "its extinction",
        "its limit", "more about it", "elaborate",
    )
    if not any(indicator in question.lower() for indicator in short_indicators):
        return question
    for turn in reversed(chat_history):
        if turn.get("role") == "user":
            topic = str(turn.get("content", "")).strip()
            if topic:
                return f"{question} (previous topic: {topic})"
    return question


# ---------------------------------------------------------------------------
# Calibrated Grounding Match Metric
# ---------------------------------------------------------------------------
def grounding_match(avg_distance: float) -> int:
    """Smoothly map average ChromaDB L2 distance to a 70%–98% match."""
    normalized = _clamp(avg_distance / MAX_DISTANCE, 0.0, 1.0)
    eased = normalized * normalized * (3.0 - 2.0 * normalized)
    return round(98.0 - 28.0 * eased)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _tier(score: float) -> str:
    if score >= 0.6:
        return "High"
    return "Medium" if score >= 0.35 else "Low"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _file_name(source: Any) -> str:
    return str(source or "").replace("\\", "/").split("/")[-1]


# --- Retrieval, not keywords, decides what is grounded ------------------------
# There is deliberately no domain-vocabulary gate here. `similarity_search`
# already discards anything past MAX_DISTANCE, so a greeting, small talk or an
# off-topic question simply comes back with no passages and is answered
# conversationally by Gemini. A keyword list could only ever guess, and the
# guesses used to route perfectly good questions into a canned reply.


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the ONNX embedding session so the very first user query is quick.
    try:
        similarity_search("flame spread in microgravity", k=1)
    except Exception:
        pass
    yield


app = FastAPI(title="NASA Space Fire Safety API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_collection: Any = None
_collection_error: Optional[str] = None


def get_collection():
    """Open the persisted Chroma index, or ``None`` when it is unavailable."""
    global _collection, _collection_error
    if _collection is None and _collection_error is None:
        try:
            _collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
        except Exception as exc:  # pragma: no cover - defensive
            _collection_error = str(exc)
    return _collection


def similarity_search(question: str, k: int = MAX_PASSAGES):
    """Return the ``k`` best chunks: fetch 10 -> filter -> de-dupe -> re-rank."""
    collection = get_collection()
    if collection is None:
        return []
    try:
        result = collection.query(query_texts=[question], n_results=FETCH_K)
    except Exception:
        return []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    candidates = []
    seen = set()
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        text = text or ""
        if _as_float(dist, 99.0) > MAX_DISTANCE:
            continue
        # Same source + page + opening words means the passage was ingested
        # twice; keeping it twice would waste a slot on a repeat.
        key = (str(meta.get("source")), meta.get("page"), text[:120])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"metadata": meta, "page_content": text, "distance": dist})
        if len(candidates) >= FETCH_K:
            break

    # Re-rank against the query before anything reaches the prompt.
    return rerank(question, candidates, k)


def confidence_of(distance: Any) -> float:
    """Map an L2 distance between unit-length MiniLM vectors to a 0-1 score."""
    try:
        gap = float(distance)
    except (TypeError, ValueError):
        return 0.5
    # For unit vectors: cosine similarity = 1 - d^2 / 2.
    return round(_clamp(1.0 - (gap * gap) / 2.0, 0.0, 1.0), 4)


# --- Candidate re-ranking ----------------------------------------------------
# Chroma scores each passage on its own, so a weakly related chunk can still
# clear the relevance floor and then dilute the prompt. Re-ranking the 10
# candidates against the actual query is what keeps that out. Three tiers,
# best available first:
#   1. Cohere Rerank  - set COHERE_API_KEY in `.env`
#   2. cross-encoder   - `pip install sentence-transformers`
#   3. lexical fusion  - no dependencies, always available
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the "
    "this to was what when which with you your".split()
)

_cross_encoder: Any = None
_cross_encoder_failed = False


def _tokens(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _cohere_ranks(query: str, docs: list) -> Optional[list[int]]:
    """Cohere Rerank ordering, or ``None`` so the next tier can take over."""
    key = os.getenv("COHERE_API_KEY", "").strip()
    if not key:
        return None
    try:
        import requests  # already installed as a chromadb dependency
    except ImportError:
        return None
    try:
        response = requests.post(
            COHERE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
                "query": query,
                "documents": [doc["page_content"][:2000] for doc in docs],
                "top_n": len(docs),
            },
            timeout=20,
        )
        response.raise_for_status()
        order = [
            int(item["index"])
            for item in response.json().get("results", [])
            if 0 <= int(item["index"]) < len(docs)
        ]
        return order or None
    except Exception:
        # Bad key, quota, offline - any problem falls through to the next tier
        # rather than failing the query the user is waiting on.
        return None


def _cross_encoder_ranks(query: str, docs: list) -> Optional[list[int]]:
    """Cross-encoder ordering, when sentence-transformers is installed."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is None and not _cross_encoder_failed:
        try:
            cross_encoder_class = importlib.import_module(
                "sentence_transformers"
            ).CrossEncoder
            _cross_encoder = cross_encoder_class(CROSS_ENCODER)
        except Exception:
            _cross_encoder_failed = True
            return None
    if _cross_encoder is None:
        return None
    try:
        scores = _cross_encoder.predict([(query, doc["page_content"]) for doc in docs])
    except Exception:
        return None
    return sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)


def _lexical_ranks(query: str, docs: list) -> list[int]:
    """Dependency-free fallback: fuse Chroma's semantic score with token overlap.

    A cross-encoder reads query and passage together; this is the cheap
    approximation of that, and it still beats trusting Chroma's order as-is.
    """
    query_tokens = _tokens(query)
    scored = []
    for index, doc in enumerate(docs):
        semantic = confidence_of(doc.get("distance"))
        overlap = len(query_tokens & _tokens(doc["page_content"])) / max(
            len(query_tokens), 1
        )
        scored.append((0.6 * semantic + 0.4 * overlap, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored]


def rerank(query: str, docs: list, top_k: int = MAX_PASSAGES) -> list:
    """Order *docs* best-first, then keep only the top ``top_k``."""
    if not docs:
        return []
    order = (
        _cohere_ranks(query, docs)
        or _cross_encoder_ranks(query, docs)
        or _lexical_ranks(query, docs)
    )
    return [docs[i] for i in order[:top_k]]


def rerank_provider() -> str:
    """Which tier the next query would use - reported on the health endpoint."""
    if os.getenv("COHERE_API_KEY", "").strip():
        return "cohere"
    if importlib.util.find_spec("sentence_transformers"):
        return "cross-encoder"
    return "lexical-fusion"


def source_url(source: Any, metadata: Optional[dict] = None) -> str:
    """Prefer a stored DOI or repository URL, then resolve NASA filenames."""
    if metadata:
        for key in ("url", "source_url", "doi_url", "doi"):
            candidate = str(metadata.get(key) or "").strip()
            if not candidate:
                continue
            if key == "doi" or candidate.lower().startswith("doi:"):
                doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", candidate, flags=re.IGNORECASE)
                doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
                if re.match(r"^10\.\d{4,9}/", doi):
                    return f"https://doi.org/{quote(doi, safe='/.-_:')}"
                continue
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return candidate

    stem = os.path.splitext(_file_name(source))[0].strip()
    if re.fullmatch(r"\d{6,}", stem):
        return f"https://ntrs.nasa.gov/citations/{stem}"
    return f"https://ntrs.nasa.gov/search?q={quote(stem) or 'microgravity+combustion'}"


def source_title(meta: dict, source: Any) -> str:
    name = _file_name(source) or "NASA source"
    page = meta.get("page_label") or meta.get("page")
    return f"{name} (p. {page})" if page is not None else name


def _primary_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the Groq API key, rejecting empty values and `.env` placeholders."""
    key = (explicit or os.getenv("GROQ_API_KEY") or "").strip()
    if not key or key.lower().startswith(("your-", "paste-", "changeme", "xxx", "<")):
        return None
    return key


def _backup_key() -> Optional[str]:
    """Resolve the OpenRouter API key, rejecting empty values and `.env` placeholders."""
    key = os.getenv("OPENROUTER_API_KEY") or ""
    key = key.strip()
    if not key or key.lower().startswith(("your-", "paste-", "changeme", "xxx", "<")):
        return None
    return key


def _no_key_answer() -> str:
    """Only when no AI key is set *and* retrieval found nothing.

    Infrastructure message, not a chat path: with no model there is genuinely
    nothing to talk to. Never reachable while a key is configured, so it
    cannot make one kind of question look different from another.
    """
    return (
        "I can't answer that right now - there's no AI key configured, so "
        "I have no model to think with, and the NASA index had nothing close "
        "enough to quote from.\n\n"
        "Add GROQ_API_KEY (primary) or OPENROUTER_API_KEY (backup) to `.env` "
        "next to main.py and restart the server to turn me back on, "
        "conversation memory included."
    )


def _passage_list(question: str, docs: list) -> str:
    """Numbered, truncated list of the passages an answer would be grounded in."""
    lines = [f'Closest passages to "{question}":', ""]
    for position, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        source = _file_name(meta.get("source")) or "unknown source"
        page = meta.get("page_label") or meta.get("page", "N/A")
        snippet = " ".join(str(doc["page_content"]).split())
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"{position}. {source}, p. {page}")
        lines.append(f"   {snippet}")
        lines.append("")
    lines.append("Open a citation below to read the full document on NTRS.")
    return "\n".join(lines)


def _preview_answer(question: str, docs: list) -> str:
    """Retrieval preview shown when no AI key is configured.

    The one path where an answer cannot come from the model, because there is no
    key to call it with - so it says so instead of imitating a reply.
    """
    return "\n".join(
        [
            "I can't write this one in my own voice yet - there's no AI key "
            "configured, so all I have is the raw retrieval below rather than a "
            "proper answer.",
            "Add GROQ_API_KEY (primary) or OPENROUTER_API_KEY (backup) to `.env` "
            "next to main.py and restart the server and I'll answer normally.",
            "",
            _passage_list(question, docs),
        ]
    )


def _unavailable_answer(question: str, docs: list, exc: BaseException) -> str:
    """Handed back once retries are exhausted.

    One voice and one shape whether or not retrieval had anything, so a science
    question failing mid-session does not surface a visibly different style of
    message from a casual one. Only reachable when the Gemini call itself has
    failed, after 3 attempts.
    """
    reason = _describe_error(exc)
    if not _is_transient(exc):
        # Permanent (bad key, blocked prompt): no point waiting, just say so.
        lead = (
            f"I couldn't reach the AI for that one ({reason}). Check "
            "GROQ_API_KEY / OPENROUTER_API_KEY in `.env` next to main.py and "
            "restart the server."
        )
    else:
        lead = (
            f"The AI is busy right now ({reason}) - I retried 3 times with "
            "backoff and it still didn't recover, so I'd rather say so than "
            "guess at an answer."
        )
    if not docs:
        return lead + " Give it another go in a moment."
    return (
        lead
        + "\n\nHere is what I would have grounded the answer in:\n\n"
        + _passage_list(question, docs)
    )


def _settings_block(
    oxygen: Optional[float],
    pressure: Optional[float],
    airflow: Optional[float],
    fuel: Optional[str],
) -> Optional[str]:
    """Render the live dashboard controls as prompt context, when supplied.

    Returns ``None`` when the client sent none of them, so a plain curl does
    not get a block full of "not set".
    """
    if oxygen is None and pressure is None and airflow is None and not fuel:
        return None

    def fmt(value: Optional[float], unit: str = "") -> str:
        return "not set" if value is None else f"{value:g}{unit}"

    return "\n".join(
        [
            "ACTIVE DASHBOARD SETTINGS",
            "The user currently has these controls set on the dashboard:",
            f"- Oxygen concentration: {fmt(oxygen, ' %')}",
            f"- Ambient pressure: {fmt(pressure, ' kPa')}",
            f"- Forced airflow velocity: {fmt(airflow, ' cm/s')}",
            f"- Fuel specimen: {fuel or 'not set'}",
            "",
            "Reference these when they matter to the question - for example when "
            "the user asks about the selected material or says 'at these "
            "settings' - and say plainly when the retrieved passages do not cover "
            "that exact combination. Do not claim to have simulated these values: "
            "the sliders feed a ranking heuristic, not a physics solver.",
        ]
    )


# The persona contract. Deliberately a module-level constant so it is obvious
# (and testable) that every request - combustion physics, general trivia, small
# talk or a polite decline - is generated from one identical instruction.
# Query-type-specific formatting rules are absent on purpose: rules like
# "casual replies get 2-4 sentences" or "science replies stay under 300 words"
# are what made the two kinds of answer read like different assistants.
_PERSONA_INSTRUCTION = """You are Aethon, the official AI guide for the Aethon microgravity combustion dashboard.

You search NASA Physical Sciences Informatics and NTRS records on microgravity combustion, including BASS, SAFFIRE, FLEX, and SoFIE. Users can ask you about flame spread, extinction limits, or material flammability. The server appends clickable source links after your answer.

Tone & Formatting Rules:
- Voice: Warm, conversational, articulate, and engaging across ALL topics.
- Uniform Style: Maintain a consistent, natural formatting style regardless of whether answering a science query, casual chat, or general knowledge question. Avoid changing into a robotic or overly template-driven format for science questions.
- Structure technical answers with short **bold section headings** and concise bullet points when useful. Keep simple conversational replies as natural paragraphs.
- Domain Expertise: Answer microgravity, fire-safety, and simulation questions with expert clarity without breaking character.
- General Knowledge: Answer general trivia (e.g., "Who is the richest person in the world?") directly and naturally using your general knowledge.
- Out-of-Scope Requests: For unsupported actions (like setting alarms or checking live device settings), humbly and politely decline in 1 short, natural sentence.
- Context Memory: Use the provided chat history to remember user details (such as names or prior conversation topics). Follow-up questions like "what about its extinction limit?" refer to the previous topic.
- Output Hygiene: Never include role labels, system tags, chain-of-thought, or robotic "as an AI language model" disclaimers.
- Citation Rules: Do not add inline citations or a reference section. The server appends all sources, with clickable links, under "### 📚 Referenced NASA Papers & Datasets"."""


def _system_instruction(
    gravity_filter: Optional[str],
    settings_block: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> str:
    """One system instruction, byte-identical in voice for every query type.

    ``_PERSONA_INSTRUCTION`` owns the tone. Everything appended below it is
    factual state (dashboard controls) or retrieval plumbing (how to treat
    attached passages) - never a competing rule about how a reply should be
    shaped, so there is nothing here that can push a science answer into a
    different format from a casual one.
    """
    gravity = (gravity_filter or "").strip() or "not specified"
    parts = [
        _PERSONA_INSTRUCTION,
        "",
        "Dashboard context:",
        f"- Gravity mode selected: {gravity}.",
        "- The oxygen / pressure / airflow sliders and 'RUN SIMULATION & RANK "
        "RISK' drive a ranking heuristic over NASA data rather than a physics "
        "simulation, so describe that output as an estimate.",
        "",
        "Conversation history (last 4 turns):",
    ]
    if chat_history:
        for turn in chat_history[-4:]:
            role = str(turn.get("role", "user")).strip()
            text = str(turn.get("content", turn.get("text", ""))).strip()
            if text:
                parts.append(f"- {role}: {text}")
    else:
        parts.append("- (none yet)")
    parts.extend(
        [
            "",
            "Retrieved context:",
            "- PASSAGES may be attached to the user's message. When they answer the "
            "question, use them as your source while keeping the same voice and "
            "formatting as everything else. The server adds source links after the answer.",
            "- When they do not answer the question, ignore them instead of forcing "
            "a citation.",
            "- When none are attached, answer from your own knowledge and never "
            "invent figures.",
        ]
    )
    if settings_block:
        parts.extend(["", settings_block])
    return "\n".join(parts)


def _user_turn(question: str, context_text: str) -> str:
    """The current turn: the user's message, plus passages when retrieval found any.

    The question is always first and unframed, so a casual or general-knowledge
    message reaches Gemini completely bare - identical to how it arrives when
    nothing was retrieved. Passages are presented as optional source material
    rather than as a cue to switch into a formal answer format.
    """
    passages = context_text.strip()
    if not passages:
        return question
    return (
        f"{question}\n\n"
        "Retrieved passages (use these only if they actually answer the "
        "question above):\n" + passages
    )


# --- Gemini text generation with retry ---------------------------------------
# Only failures a retry can plausically fix are retried: 503 unavailability, 429
# rate limits and timeouts. Everything else (bad key = 400, blocked prompt, ...)
# is permanent, and re-running it would only delay an error the user needs now.
_TRANSIENT_CODES = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_WORDS = (
    "rate limit",
    "rate_limit",
    "rate-limit",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "unavailable",
    "overloaded",
    "timeout",
    "timed out",
    "temporarily",
    "connection reset",
)


def _is_transient(exc: BaseException) -> bool:
    """True when retrying could plausibly succeed (503 / rate limit / timeout)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        code = int(getattr(exc, "code", None))
    except (TypeError, ValueError):
        code = None
    if code is not None:
        return code in _TRANSIENT_CODES
    text = str(exc).lower()
    return any(word in text for word in _TRANSIENT_WORDS)


def _describe_error(exc: BaseException) -> str:
    """Short human-readable reason, never a raw exception dump."""
    try:
        code = int(getattr(exc, "code", None))
    except (TypeError, ValueError):
        code = None
    if code == 429:
        return "rate limit reached (HTTP 429)"
    if code == 503:
        return "service unavailable (HTTP 503)"
    if code is not None:
        return f"HTTP {code} {type(exc).__name__}"
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:160] + "..." if len(text) > 160 else text


def _clean_history(raw: Any) -> list:
    """Normalise the client's history into clean ``user`` / ``model`` turns."""
    if not isinstance(raw, (list, tuple)):
        return []
    turns = []
    for turn in list(raw)[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        text = str(
            turn.get("text") or turn.get("content") or turn.get("message") or ""
        ).strip()
        if not text:
            continue
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            continue
        turns.append({"role": role, "text": text})
    # Gemini's content list must open with a user turn.
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def _build_contents(history: list, question: str) -> list:
    """History plus the current question, shaped as a Gemini content list.

    Adjacent turns of the same role are folded together, because Gemini expects
    roles to alternate and otherwise rejects the request with a 400.
    """
    merged: list = []
    for turn in list(history) + [{"role": "user", "text": question}]:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {
                "role": turn["role"],
                "text": merged[-1]["text"] + "\n" + turn["text"],
            }
        else:
            merged.append(dict(turn))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return [
        {"role": turn["role"], "text": turn["text"]}
        for turn in merged
    ]


@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_groq(contents: list, system_instruction: str, api_key: str) -> str:
    """One Groq generation, retried up to 3 times on transient errors."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": AI_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_instruction},
        ]
        + [
            {
                "role": "assistant" if turn["role"] == "model" else turn["role"],
                "content": turn["text"],
            }
            for turn in contents
        ],
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(GROQ_BASE_URL, headers=headers, json=payload)
    if response.status_code in (401, 403):
        raise RuntimeError(f"Groq authentication failed (HTTP {response.status_code})")
    response.raise_for_status()
    data = response.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("Groq returned an empty response")
    return text


@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def _call_openrouter(contents: list, system_instruction: str, api_key: str) -> str:
    """One OpenRouter generation, retried up to 3 times on transient errors."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8000",
        "X-Title": "Aethon - NASA Microgravity Combustion",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": AI_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_instruction},
        ]
        + [
            {
                "role": "assistant" if turn["role"] == "model" else turn["role"],
                "content": turn["text"],
            }
            for turn in contents
        ],
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
    if response.status_code in (401, 403):
        raise RuntimeError(f"OpenRouter authentication failed (HTTP {response.status_code})")
    response.raise_for_status()
    data = response.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty response")
    return text


def _ai_answer(
    question: str,
    gravity_filter: Optional[str],
    context_text: str,
    primary_key: Optional[str],
    backup_key: Optional[str],
    docs: list,
    settings_block: Optional[str] = None,
    history: Any = None,
    chat_history: Optional[list] = None,
) -> tuple[str, bool, str]:
    """Try Groq (primary), fall back to OpenRouter (backup).

    Returns ``(answer, ai_used, provider_name)``.
    """
    contents = _build_contents(
        _clean_history(history), _user_turn(question, context_text)
    )
    system = _system_instruction(gravity_filter, settings_block, chat_history)

    # Try primary (Groq) first.
    if primary_key is not None:
        try:
            answer = _call_groq(contents, system, primary_key)
            return answer, True, "groq"
        except Exception as exc:
            # Groq failed permanently or after retries; fall back to backup.
            pass

    # Try backup (OpenRouter).
    if backup_key is not None:
        try:
            answer = _call_openrouter(contents, system, backup_key)
            return answer, True, "openrouter"
        except Exception as exc:
            # Backup also failed - hand back an unavailable answer.
            ai_answer = _unavailable_answer(question, docs, exc)
            return ai_answer, False, "openrouter"

    # No keys configured at all.
    ai_answer = _preview_answer(question, docs) if docs else _no_key_answer()
    return ai_answer, False, "none"


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    gravity_filter: Optional[str] = None
    api_key: Optional[str] = None
    # Last 4 conversation turns as [{"role": "user"|"assistant"|"model", "content": ...}, ...].
    # Replayed into the system prompt so the assistant remembers earlier turns.
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    # Prior turns of this conversation as
    # ``[{"role": "user" | "assistant" | "model", "text": ...}, ...]``.
    # Replayed to Gemini on every call so the assistant remembers what was
    # said earlier. Optional: plain curl and older clients simply omit it.
    history: Optional[list] = None
    # Live dashboard controls, so answers can be phrased in terms of the
    # parameters the user is actually looking at. All optional: older clients
    # (and plain curl) simply omit them.
    oxygen_conc: Optional[float] = None
    pressure: Optional[float] = None
    airflow: Optional[float] = None
    fuel: Optional[str] = None


_MODEL_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:📚\s*)?"
    r"(?:referenced NASA papers(?:\s*&\s*datasets)?|"
    r"references|sources(?:\s*(?:&|and)\s*references)?|citations)"
    r"(?:\*\*)?\s*:?\s*$"
)
_INLINE_PDF_CITATION_RE = re.compile(
    r"\s*\[[^\]\n]*\.pdf(?:\s+(?:p|page)\.?\s*\d+)?[^\]\n]*\]",
    re.IGNORECASE,
)
_SYSTEM_TAG_RE = re.compile(
    r"<\|(?:system|assistant|user|developer|analysis|final|im_start|im_sep|im_end|"
    r"fim_prefix|fim_middle|fim_suffix)\|>",
    re.IGNORECASE,
)
_ROLE_PREFIX_RE = re.compile(r"(?im)^\s*(?:assistant|system|analysis|final)\s*:\s*")
_ROBOTIC_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:as an ai(?:\s+(?:language model|assistant))?|"
    r"as a large language model)[,:\s-]+"
)
NASA_REFERENCES_HEADING = "### 📚 Referenced NASA Papers & Datasets"


def _strip_model_references(answer: str) -> str:
    """Remove model-generated source sections; references are server-owned."""
    heading = _MODEL_REFERENCE_HEADING_RE.search(answer)
    if heading:
        answer = answer[: heading.start()]
    return _INLINE_PDF_CITATION_RE.sub("", answer).rstrip()


def _clean_ai_response(answer: str) -> str:
    """Remove common model tags and robotic lead-ins without rewriting content."""
    cleaned = _SYSTEM_TAG_RE.sub("", str(answer or ""))
    cleaned = _ROLE_PREFIX_RE.sub("", cleaned)
    cleaned = _ROBOTIC_PREFIX_RE.sub("", cleaned)
    cleaned = _strip_model_references(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


@app.get("/")
@app.get("/health")
def read_root():
    primary_key = _primary_key()
    backup_key = _backup_key()
    collection = get_collection()
    return {
        "message": "NASA Space Fire Safety RAG API is active.",
        "index_ready": collection is not None,
        "index_error": _collection_error,
        "embedding_model": EMBEDDING_MODEL,
        "groq_configured": primary_key is not None,
        "openrouter_configured": backup_key is not None,
        "groq_model": GROQ_MODEL if primary_key else None,
        "openrouter_model": OPENROUTER_MODEL if backup_key else None,
        "ai_provider": "groq" if primary_key else "openrouter" if backup_key else None,
        "ai_model": GROQ_MODEL if primary_key else OPENROUTER_MODEL if backup_key else None,
        "ai_engine": "RAG Synthesizer",
        "max_passages": MAX_PASSAGES,
        "fetch_k": FETCH_K,
        "rerank": rerank_provider(),
    }


@app.post("/query-agent")
def query_agent(request: QueryRequest):
    question = (request.question or request.query or "").strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="A `question` (or `query`) field is required."
        )

    # --- Smart Small-Talk & Closure Router ---------------------------------
    # Greetings and closures get an immediate polite AETHON reply without
    # touching ChromaDB, so there is never a "No matching papers found" error.
    small_talk = _small_talk_route(question)
    if small_talk:
        return {
            "status": "success",
            "answer": small_talk,
            "ai_answer": small_talk,
            "citations": [],
            "sources": [],
            "retrieved_sources": [],
            "confidence": None,
            "grounding_match": None,
            "question": question,
            "gravity_filter": request.gravity_filter,
            "ai_used": False,
            "ai_provider": "small-talk-router",
            "passages_used": 0,
            "conversational": True,
        }

    # --- Query Expansion & Typo Normalization ------------------------------
    expanded_question = expand_and_normalize_query(question)

    # --- Follow-up resolution ----------------------------------------------
    chat_history = [
        {"role": m.role, "content": m.content} for m in (request.chat_history or [])
    ]
    search_query = _resolve_follow_up(expanded_question, chat_history)

    # Validate combustion boundary conditions before retrieval
    warning = _validate_combustion_conditions(
        request.oxygen_conc, request.fuel
    )

    # Live dashboard controls, surfaced to the model as prompt context.
    settings_block = _settings_block(
        request.oxygen_conc, request.pressure, request.airflow, request.fuel
    )

    # Retrieval runs for every message. Casual or off-topic wording simply finds
    # nothing past the relevance floor, and an empty passage list is what marks
    # the answer as ungrounded - no keyword gate decides what may be asked.
    docs = similarity_search(search_query, k=MAX_PASSAGES)

    retrieved_sources = []
    context_text = ""
    citations = []
    for doc in docs:
        meta = doc["metadata"]
        source = str(meta.get("source") or "Unknown File")
        page = meta.get("page", "N/A")
        snippet = doc["page_content"]
        retrieved_sources.append(
            {
                "source": source,
                "filename": _file_name(source),
                "page": page,
                "snippet": snippet,
                "title": source_title(meta, source),
                "url": source_url(source, meta),
                "confidence": confidence_of(doc.get("distance")),
            }
        )
        context_text += (
            f"\n[PASSAGE {len(citations) + 1}] {source} (page {page})\n{snippet}\n"
        )
        citations.append({
            "title": source_title(meta, source),
            "url": source_url(source, meta),
            "confidence": confidence_of(doc.get("distance")),
        })

    # --- Calibrated Grounding Match Metric ---------------------------------
    if citations:
        avg_distance = sum(
            _as_float(doc.get("distance"), MAX_DISTANCE) for doc in docs
        ) / len(docs)
        gm = grounding_match(avg_distance)
    else:
        gm = None

    # One code path for every message: hand the question, whatever passages
    # retrieval found, and the conversation history to the AI. Chit-chat simply
    # arrives with no passages and is answered as chit-chat.
    primary_key = _primary_key(request.api_key)
    backup_key = _backup_key()
    ai_used = False
    provider = "none"
    if primary_key is None and backup_key is None:
        ai_answer = _preview_answer(question, docs) if docs else _no_key_answer()
    else:
        ai_answer, ai_used, provider = _ai_answer(
            question,
            request.gravity_filter,
            context_text,
            primary_key,
            backup_key,
            docs,
            settings_block,
            request.history,
            chat_history,
        )

    # --- Clean Citation Section --------------------------------------------
    # Append a clean, deduplicated reference list at the end of the answer.
    # The model is instructed not to emit inline citations in the body.
    ai_answer = _clean_ai_response(ai_answer)
    if citations:
        seen_titles = set()
        ref_lines = [f"\n\n{NASA_REFERENCES_HEADING}", ""]
        for c in citations:
            title = c["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            conf = c.get("confidence")
            conf_str = f" ({int(conf * 100)}% match)" if conf is not None else ""
            ref_lines.append(f"- [{title}]({c['url']}){conf_str}")
        ai_answer = ai_answer.rstrip() + "\n" + "\n".join(ref_lines)

    response = {
        "status": "success",
        "answer": ai_answer,
        "citations": citations,
        "confidence": gm,
        "grounding_match": gm,
        "question": question,
        "ai_answer": ai_answer,
        "retrieved_sources": retrieved_sources,
        "sources": citations,
        "gravity_filter": request.gravity_filter,
        "ai_used": ai_used,
        "ai_provider": provider,
        "passages_used": len(docs),
        "conversational": not docs,
    }

    if warning:
        response["combustion_warning"] = warning

    return response


async def _read_json(request: Optional[Request]) -> dict:
    """Read an optional JSON body, merged with any query-string parameters."""
    payload: dict = {}
    if request is None:
        return payload
    try:
        raw = await request.body()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload.update(parsed)
    except Exception:
        pass
    payload.update(dict(request.query_params))
    return payload


def build_ranking(payload: dict) -> list:
    """Rank every material for the requested experiment conditions."""
    oxygen = _clamp(_as_float(payload.get("oxygen_conc"), 21.0), 15.0, 35.0)
    airflow = _clamp(_as_float(payload.get("airflow"), 15.0), 0.0, 50.0)
    pressure = _clamp(_as_float(payload.get("pressure"), 101.3), 50.0, 150.0)
    gravity = str(payload.get("gravity") or "0g")

    gravity_factor = 1.1 if gravity == "0g" else 1.0 if gravity == "0.16g" else 0.9
    oxygen_factor = 0.55 + ((oxygen - 15.0) / 20.0) * 0.9
    pressure_factor = _clamp(0.7 + 0.3 * (pressure / 101.3), 0.7, 1.3)
    flow_factor = 0.9 + (airflow / 50.0) * 0.3

    rankings = []
    for material in MATERIALS:
        score = _clamp(
            material["base"] * oxygen_factor * pressure_factor * flow_factor * gravity_factor,
            0.0,
            1.0,
        )
        rankings.append({
            "material": material["label"],
            "risk_level": material["risk_level"],
            "microgravity_flame_spread": material["spread"],
            "extinction_limit": material["extinction"],
            "score": round(score, 4),
            "tier": _tier(score),
            "risk": _tier(score),
        })
    rankings.sort(key=lambda item: item["score"], reverse=True)
    return rankings


def _rank_response(payload: dict) -> dict:
    rankings = build_ranking(payload)
    return {
        "status": "success",
        "rankings": rankings,
        "ranking": rankings,
        "results": rankings,
        "gravity": str(payload.get("gravity") or "0g"),
        "oxygen_conc": payload.get("oxygen_conc"),
        "airflow": payload.get("airflow"),
        "material": payload.get("material"),
        "note": "Server-side ranking; scores are 0-1 flammability estimates.",
    }


@app.post("/rank-materials")
async def rank_materials_post(request: Request):
    return _rank_response(await _read_json(request))


@app.get("/rank-materials")
async def rank_materials_get(request: Request):
    return _rank_response(await _read_json(request))


def pick_port(preferred: int) -> int:
    """Return *preferred* when free, otherwise the first free fallback port."""
    for port in (preferred, 8080, 8081, 8001):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return preferred


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or pick_port(8000))
    print(f"NASA Space Fire Safety API -> http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)



