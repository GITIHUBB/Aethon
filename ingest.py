"""Document ingestion engine for the MicroCombust-AI platform.

Ingests NASA combustion papers from ``papers/`` into a persistent ChromaDB
index using local HuggingFace embeddings.

Usage:
    python ingest.py              # ingest / update all papers
    python ingest.py --reset      # wipe the index before ingesting
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _should_skip(filename: str, text_preview: str) -> bool:
    """Return True when a file is a template/placeholder that must be skipped."""
    name = filename.lower()
    if name.startswith("s_"):
        return True
    if "do not use this" in text_preview.lower():
        return True
    return False


def run_ingestion(reset: bool = False) -> None:
    """Ingest all papers from PAPERS_DIR into ChromaDB."""
    from langchain_core.documents import Document

    print("Initializing ChromaDB and embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if reset and os.path.exists(DB_PATH):
        import shutil

        shutil.rmtree(DB_PATH)
        print(f"  -> Reset: removed existing index at {DB_PATH}")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    if not os.path.isdir(PAPERS_DIR):
        print(f"ERROR: papers directory not found: {PAPERS_DIR}")
        sys.exit(1)

    files = sorted(
        f
        for f in os.listdir(PAPERS_DIR)
        if f.lower().endswith((".pdf", ".txt"))
    )
    print(f"Found {len(files)} files in '{PAPERS_DIR}/'. Starting ingestion...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    skipped = 0

    for filename in files:
        file_path = os.path.join(PAPERS_DIR, filename)
        print(f"Processing: {filename}")

        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents: List[Document] = loader.load()
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
                documents = [Document(page_content=raw, metadata={"source": filename})]

            # Attach source metadata to every loaded document
            for doc in documents:
                doc.metadata["source"] = filename

            # Skip template / placeholder files
            preview = " ".join(d.page_content for d in documents[:3])[:500]
            if _should_skip(filename, preview):
                print(f"  -> Skipped: template/placeholder file '{filename}'")
                skipped += 1
                continue

            if not documents or not any(d.page_content.strip() for d in documents):
                print(f"  -> Warning: No text extracted from {filename}. Skipping.")
                skipped += 1
                continue

            chunks = text_splitter.split_documents(documents)
            chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

            if not chunks:
                print(f"  -> Warning: No usable chunks from {filename}. Skipping.")
                skipped += 1
                continue

            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  -> Stored {len(chunks)} chunks from {filename}.")

        except Exception as exc:  # pragma: no cover - defensive
            print(f"  -> Error processing {filename}: {exc}")

    print(f"\nIngestion complete! {total_chunks} chunks indexed, {skipped} skipped.")
    print(f"Index location: {DB_PATH} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NASA combustion papers into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB index before ingesting",
    )
    args = parser.parse_args()
    run_ingestion(reset=args.reset)
