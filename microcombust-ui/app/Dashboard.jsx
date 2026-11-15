'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowDown, ArrowUp, ChevronDown, ChevronUp, Database,
  Download, ExternalLink, FileText, Flame, Loader2, Play, Send, ShieldCheck,
  Sparkles, User, Wind, X,
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
/* The FastAPI backend binds the first free port (8000 is usually taken by a
   Windows service), so probe a short list instead of hard-coding one. */
const API_CANDIDATES = (
  process.env.NEXT_PUBLIC_API_URL ||
  'http://127.0.0.1:8000,http://127.0.0.1:8080,http://127.0.0.1:8081,http://127.0.0.1:8001'
)
  .split(',')
  .map((base) => base.trim().replace(/\/+$/, ''))
  .filter(Boolean);

// Port that answered on the last successful call, so we skip dead ones after.
let activeApi = null;

const isJson = (res) => String(res.headers.get('content-type') || '').includes('application/json');
const apiAddressLabel = (base) => {
  try {
    const url = new URL(base);
    const host = url.hostname === 'localhost' ? '127.0.0.1' : url.hostname;
    return `${host}${url.port ? `:${url.port}` : ''}`;
  } catch {
    return base;
  }
};
const aiModelLabel = (model) =>
  String(model || '').toLowerCase().includes('llama-3.3-70b')
    ? 'Llama-3.3-70b'
    : model;

class BackendConnectionError extends Error {}
class BackendResponseError extends Error {}

/** POST JSON to `path`, trying each configured base until one answers. */
async function apiFetch(path, body) {
  const bases = activeApi
    ? [activeApi, ...API_CANDIDATES.filter((base) => base !== activeApi)]
    : [...API_CANDIDATES];

  let lastError = null;
  let lastResponse = null;

  for (const base of bases) {
    try {
      const res = await fetch(`${base}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        lastResponse = res;
        if (res.status !== 404) break;
        continue;
      }
      if (res.ok && isJson(res)) {
        activeApi = base;
        return res;
      }
      lastResponse = res;
    } catch (err) {
      lastError = err;
    }
  }

  if (lastResponse) {
    let message = lastResponse.ok
      ? `Backend returned HTTP ${lastResponse.status} with a non-JSON response`
      : `Backend returned HTTP ${lastResponse.status}`;
    try {
      const body = await lastResponse.clone().json();
      const detail = body?.detail || body?.message;
      if (detail) {
        message += `: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`;
      }
    } catch {
      // Keep the HTTP status when the server does not return a JSON error body.
    }
    throw new BackendResponseError(message);
  }

  if (lastError) {
    throw new BackendConnectionError(
      `Could not connect to any backend at ${bases.join(', ')}: ${String(lastError)}`,
    );
  }
  throw new BackendConnectionError(`No backend responded at ${bases.join(', ')}`);
}

const GRAVITY_MODES = [
  { value: '0g', label: '0g', sub: 'Microgravity' },
  { value: '0.16g', label: '0.16g', sub: 'Lunar gravity' },
  { value: '1g', label: '1g', sub: 'Earth gravity' },
];

const FUELS = ['PMMA (Acrylic)', 'Nomex Fabric', 'Silicone Rubber', 'Kevlar Blend'];

// Short material names used by the matrix and table
const MATERIALS = [
  { id: 'PMMA', label: 'PMMA', base: 0.8, fuel: 'PMMA (Acrylic)' },
  { id: 'Silicone', label: 'Silicone', base: 0.75, fuel: 'Silicone Rubber' },
  { id: 'Cotton', label: 'Cotton', base: 0.55, fuel: null },
  { id: 'Nomex', label: 'Nomex', base: 0.45, fuel: 'Nomex Fabric' },
  { id: 'Kevlar', label: 'Kevlar', base: 0.25, fuel: 'Kevlar Blend' },
];

// `k` feeds the --k custom property that globals.css uses to tint the
// matrix columns, bars and risk badges.
const TIERS = {
  High: { title: 'High risk', hint: 'Sustained spread likely', k: '#fb7185' },
  Medium: { title: 'Medium risk', hint: 'Conditional spread', k: '#fbbf24' },
  Low: { title: 'Low risk', hint: 'Self-extinguishing in most cases', k: '#34d399' },
};

/* ------------------------------------------------------------------ */
/* Demo data. Replace with metrics extracted into ChromaDB.            */
/* Values below are ILLUSTRATIVE placeholders, not verified NASA data. */
/* ------------------------------------------------------------------ */
const SOFIE = 'https://science.nasa.gov/biological-physical/investigations/sofie/';
const SOFIE_GLENN =
  'https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/combustion-science/solid-fuel-ignition-and-extinction-sofie/';
const PSI = 'https://science.nasa.gov/physical-sciences-informatics-psi/';
const NTRS = 'https://ntrs.nasa.gov/';

const SEED_RUNS = [
  { id: 1, fuel: 'PMMA', gravity: '0g', o2: 21, airflow: 5, spread: 0.9, risk: 'High', url: SOFIE },
  { id: 2, fuel: 'PMMA', gravity: '0g', o2: 30, airflow: 10, spread: 2.1, risk: 'High', url: SOFIE_GLENN },
  { id: 3, fuel: 'PMMA', gravity: '0.16g', o2: 21, airflow: 10, spread: 1.4, risk: 'High', url: PSI },
  { id: 4, fuel: 'PMMA', gravity: '1g', o2: 21, airflow: 10, spread: 1.1, risk: 'High', url: NTRS },
  { id: 5, fuel: 'Silicone', gravity: '0g', o2: 21, airflow: 15, spread: 1.2, risk: 'High', url: SOFIE },
  { id: 6, fuel: 'Silicone', gravity: '0.16g', o2: 25, airflow: 20, spread: 1.8, risk: 'High', url: PSI },
  { id: 7, fuel: 'Cotton', gravity: '0g', o2: 21, airflow: 10, spread: 0.6, risk: 'Medium', url: SOFIE_GLENN },
  { id: 8, fuel: 'Cotton', gravity: '1g', o2: 21, airflow: 10, spread: 0.8, risk: 'Medium', url: NTRS },
  { id: 9, fuel: 'Nomex', gravity: '0g', o2: 21, airflow: 20, spread: 0.4, risk: 'Medium', url: SOFIE },
  { id: 10, fuel: 'Nomex', gravity: '0.16g', o2: 30, airflow: 15, spread: 0.7, risk: 'Medium', url: PSI },
  { id: 11, fuel: 'Kevlar', gravity: '0g', o2: 21, airflow: 15, spread: 0.0, risk: 'Low', url: SOFIE },
  { id: 12, fuel: 'Kevlar', gravity: '1g', o2: 30, airflow: 20, spread: 0.2, risk: 'Low', url: NTRS },
];

const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  text:
    "I'm Aethon, your AI assistant for NASA fire safety and microgravity combustion research. I can search NASA Physical Sciences Informatics and NTRS records across multiple experiments and missions. Ask me about flame spread, extinction limits, material flammability, or any related topic — every answer lists its sources and a confidence score.",
  citations: [],
  confidence: null,
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const tierOf = (s) => (s >= 0.6 ? 'High' : s >= 0.35 ? 'Medium' : 'Low');
const fuelToId = (fuel) => MATERIALS.find((m) => m.fuel === fuel)?.id;

// Offline estimate so the matrix stays usable when the backend is unreachable.
// A rough heuristic for demo purposes only, not a physics model.
function localRank({ oxygen, pressure, airflow, gravity }) {
  const g = gravity === '0g' ? 1.1 : gravity === '0.16g' ? 1.0 : 0.9;
  const fO2 = 0.55 + ((oxygen - 15) / 20) * 0.9;
  const fP = clamp(0.7 + 0.3 * (pressure / 101.3), 0.7, 1.3);
  const fFlow = 0.9 + (airflow / 50) * 0.3;
  return MATERIALS.map((m) => {
    const score = clamp(m.base * fO2 * fP * fFlow * g, 0, 1);
    return { material: m.id, score, tier: tierOf(score) };
  });
}

// Accepts a few reasonable response shapes from /rank-materials
function normalizeRanking(data) {
  const list = Array.isArray(data) ? data : data?.ranking || data?.results || [];
  return list
    .map((r) => {
      const name = String(r.material || r.name || '');
      const id = MATERIALS.find((m) => name.toLowerCase().includes(m.id.toLowerCase()))?.id || name;
      const score = clamp(Number(r.score ?? r.risk_score ?? 0), 0, 1);
      const tier = ['High', 'Medium', 'Low'].find(
        (t) => t.toLowerCase() === String(r.tier || r.risk || r.risk_tier || '').toLowerCase()
      ) || tierOf(score);
      return { material: id, score, tier };
    })
    .filter((r) => r.material);
}

// Accepts { answer, citations: [{title,url,confidence}] } or flat fields
function normalizeAnswer(data) {
  const citations =
    data?.citations ||
    data?.sources ||
    (data?.citation_url ? [{ title: data.citation_title || 'NASA source', url: data.citation_url }] : []);
  const list = citations.map((c) => ({
    title: c.title || c.name || 'NASA source',
    url: c.url || c.link,
    confidence: c.confidence ?? c.score ?? null,
  }));
  const confidence =
    data?.confidence ??
    (list.length && list.every((c) => c.confidence != null)
      ? list.reduce((a, c) => a + c.confidence, 0) / list.length
      : null);
  return {
    text: data?.answer || data?.response || 'No answer returned.',
    citations: list,
    confidence,
    question: data?.question || null,
    // Full ChromaDB passages, so inline citation pills can show the chunk
    // they were drawn from instead of only linking to the document.
    sources: Array.isArray(data?.retrieved_sources) ? data.retrieved_sources : [],
  };
}

const pct = (c) => (c == null ? null : Math.round((c <= 1 ? c * 100 : c)));

/* ------------------------------------------------------------------ */
/* Small UI pieces                                                     */
/* ------------------------------------------------------------------ */
function Panel({
  title,
  icon: Icon,
  right,
  children,
  className = '',
  a = '#22d3ee',
  b = '#a78bfa',
  iconColor,
  panelRef,
}) {
  return (
    <section ref={panelRef} style={{ '--a': a, '--b': b }} className={`glass ${className}`}>
      <header className="ph">
        <h2>
          {Icon && (
            <span className="ic" style={iconColor ? { '--c': iconColor } : undefined}>
              <Icon aria-hidden />
            </span>
          )}
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  );
}

function Slider({ id, label, unit, value, min, max, step = 1, digits = 0, c1, c2, onChange }) {
  const fill = ((value - min) / (max - min)) * 100;
  const handleChange = (e) => {
    const next = Number(e.target.value);
    onChange(next);
  };
  return (
    <div>
      <div className="row">
        <label htmlFor={id} className="lbl">
          {label}
        </label>
        <span className="val" style={{ '--c1': c1 }}>
          {Number(value).toFixed(digits)}
          <i>{unit}</i>
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        className="range"
        style={{
          '--c1': c1,
          '--c2': c2,
          background: `linear-gradient(to right, ${c1} 0%, ${c2} ${fill}%, rgba(255,255,255,.12) ${fill}%)`,
        }}
      />
      <div className="rng">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function RiskChip({ tier }) {
  const t = TIERS[tier] || TIERS.Low;
  return (
    <span className="badge" style={{ '--k': t.k }}>
      {tier}
    </span>
  );
}

function Citation({ message }) {
  const [open, setOpen] = useState(false);
  const c = pct(message.confidence);
  const count = message.citations.length;
  return (
    <div className="src">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="src-toggle"
      >
        <ShieldCheck aria-hidden />
        {count} {count === 1 ? 'source' : 'sources'}
        {c != null && <b>{c}% confidence</b>}
        {open ? <ChevronUp aria-hidden /> : <ChevronDown aria-hidden />}
      </button>
      {open && (
        <ul className="src-list">
          {count === 0 && <li className="src-empty">No citations were returned for this answer.</li>}
          {message.citations.map((s, i) => (
            <li key={i}>
              <a href={s.url} target="_blank" rel="noreferrer">
                <span className="src-main">
                  <span className="src-title">{s.title}</span>
                  <span className="src-url">{s.url}</span>
                </span>
                <span className="src-meta">
                  {s.confidence != null && <span>{pct(s.confidence)}%</span>}
                  <ExternalLink aria-hidden />
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Inline citation pills + source viewer                               */
/* ------------------------------------------------------------------ */
// Matches the markers the prompt asks Gemini to emit, e.g. [19930010989.pdf p.202]
const CITATION_RE = /^\[([^\[\]]+?\.pdf)\s+(?:p|page)\.?\s*(\d+)\]$/i;
const INLINE_MARKDOWN_RE =
  /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|(\[[^\[\]]+?\.pdf\s+(?:p|page)\.?\s*\d+\])/gi;

const fileNameOf = (value) =>
  String(value || '')
    .replace(/\\/g, '/')
    .split('/')
    .pop()
    .toLowerCase();

// Resolve a marker against the passages ChromaDB returned for this answer.
function findSource(sources, filename, page) {
  if (!sources?.length) return null;
  const name = String(filename || '').trim().toLowerCase();
  const matches = sources.filter(
    (s) => fileNameOf(s.filename || s.source) === name
  );
  if (!matches.length) return null;
  return matches.find((s) => String(s.page) === String(page)) || matches[0];
}

function renderInlineMarkdown(text, sources, onOpen, keyPrefix) {
  const value = String(text || '');
  const nodes = [];
  let cursor = 0;
  let tokenIndex = 0;

  for (const match of value.matchAll(INLINE_MARKDOWN_RE)) {
    const [whole, linkLabel, linkUrl, boldStars, boldUnderscores, citationText] = match;
    const at = match.index ?? 0;
    if (at > cursor) {
      nodes.push(
        <span key={`${keyPrefix}-text-${tokenIndex++}`}>{value.slice(cursor, at)}</span>
      );
    }

    const tokenKey = `${keyPrefix}-token-${tokenIndex++}`;
    if (linkLabel && linkUrl) {
      nodes.push(
        <a key={tokenKey} href={linkUrl} target="_blank" rel="noreferrer">
          {linkLabel}
        </a>
      );
    } else if (boldStars || boldUnderscores) {
      nodes.push(<strong key={tokenKey}>{boldStars || boldUnderscores}</strong>);
    } else if (citationText) {
      const citation = citationText.match(CITATION_RE);
      if (!citation) continue;
      const [, filename, page] = citation;
      const source = findSource(sources, filename, page);
      nodes.push(
        <button
          key={tokenKey}
          type="button"
          onClick={() => onOpen(source || { filename, page, snippet: null })}
          title={
            source
              ? 'Show the passage this claim was drawn from'
              : 'No stored passage matched this citation'
          }
          className="cite"
        >
          <FileText aria-hidden />
          <span>{citationText.slice(1, -1)}</span>
        </button>
      );
    }
    cursor = at + whole.length;
  }

  if (cursor < value.length) {
    nodes.push(
      <span key={`${keyPrefix}-text-${tokenIndex}`}>{value.slice(cursor)}</span>
    );
  }
  return nodes;
}

function AnswerText({ text, sources, onOpen }) {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1;
      continue;
    }

    const heading = lines[index].match(/^\s{0,3}#{1,3}\s+(.+)$/);
    if (heading) {
      blocks.push(
        <h3 className="answer-heading" key={`heading-${index}`}>
          {renderInlineMarkdown(heading[1], sources, onOpen, `heading-${index}`)}
        </h3>
      );
      index += 1;
      continue;
    }

    const firstBullet = lines[index].match(/^\s*[-*+]\s+(.+)$/);
    if (firstBullet) {
      const items = [];
      while (index < lines.length) {
        const bullet = lines[index].match(/^\s*[-*+]\s+(.+)$/);
        if (!bullet) break;
        items.push(
          <li key={`item-${index}`}>
            {renderInlineMarkdown(bullet[1], sources, onOpen, `item-${index}`)}
          </li>
        );
        index += 1;
      }
      blocks.push(<ul key={`list-${index}`}>{items}</ul>);
      continue;
    }

    const paragraph = [];
    const paragraphStart = index;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^\s{0,3}#{1,3}\s+/.test(lines[index]) &&
      !/^\s*[-*+]\s+/.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(
      <p className="answer-paragraph" key={`paragraph-${paragraphStart}`}>
        {paragraph.map((line, lineIndex) => (
          <span key={`line-${paragraphStart}-${lineIndex}`}>
            {lineIndex > 0 && <br />}
            {renderInlineMarkdown(
              line,
              sources,
              onOpen,
              `paragraph-${paragraphStart}-${lineIndex}`
            )}
          </span>
        ))}
      </p>
    );
  }

  return <div className="answer-text">{blocks}</div>;
}

function SourceModal({ source, onClose }) {
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  if (!source) return null;

  const filename = source.filename || fileNameOf(source.source) || 'NASA source';
  const snippet = source.snippet || '';
  const confidence = pct(source.confidence);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-label={`Source passage from ${filename}`}
    >
      <div onClick={onClose} aria-hidden className="absolute inset-0 bg-black/80 backdrop-blur-sm" />
      <div className="glass modal-card" style={{ '--a': '#a78bfa', '--b': '#f472b6' }}>
        <header className="ph">
          <div className="flex min-w-0 items-center gap-3">
            <span className="ic" style={{ '--c': '#a78bfa' }}>
              <FileText aria-hidden />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{filename}</p>
              <p className="truncate text-xs" style={{ color: 'var(--m)' }}>
                Page {source.page ?? 'n/a'}
                {source.source ? `  ·  ${source.source}` : ''}
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close source viewer" className="icon-btn">
            <X aria-hidden />
          </button>
        </header>

        <div className="modal-body">
          <p className="modal-label">Source text chunk · ChromaDB</p>
          {snippet ? (
            <p className="snippet">{snippet}</p>
          ) : (
            <p className="notice">
              This citation has no stored passage attached, so there is nothing to
              show here. The citations listed under the answer still link to the
              source document.
            </p>
          )}
        </div>

        <footer className="foot modal-foot">
          <span>
            Retrieved from the NASA paper index
            {confidence != null && (
              <span style={{ marginLeft: 8, fontFamily: 'var(--mono)', color: 'var(--em)' }}>
                {confidence}% confidence
              </span>
            )}
          </span>
          {source.url && (
            <a href={source.url} target="_blank" rel="noreferrer" className="btn-ghost">
              Open full document <ExternalLink aria-hidden />
            </a>
          )}
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Export summary report                                               */
/* ------------------------------------------------------------------ */
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Report uses its own palette: it is a printable document, so it is light.
const TIER_HEX = {
  High: { fg: '#be123c', bg: 'rgba(244,63,94,.12)', bar: '#f43f5e' },
  Medium: { fg: '#b45309', bg: 'rgba(245,158,11,.14)', bar: '#f59e0b' },
  Low: { fg: '#047857', bg: 'rgba(16,185,129,.14)', bar: '#10b981' },
};

const reportStamp = (d) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
    d.getDate()
  ).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(
    d.getMinutes()
  ).padStart(2, '0')}`;

const REPORT_CSS = `
*{box-sizing:border-box}
body{margin:0;background:#eef2f7;color:#0f172a;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.sheet{max-width:840px;margin:24px auto;background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 18px 48px -28px rgba(15,23,42,.5);overflow:hidden}
.hero{padding:26px 30px 24px;background:linear-gradient(135deg,#0b1220 0%,#123047 55%,#0e7490 100%);color:#f8fafc}
.brand{font-size:22px;font-weight:800;letter-spacing:.06em}
.sub{margin-top:5px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;opacity:.8}
.meta{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;opacity:.8}
.status{display:inline-block;padding:3px 10px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25)}
.print{margin-top:18px;background:#22d3ee;color:#05202b;border:0;border-radius:9px;padding:9px 16px;font-size:13px;font-weight:700;cursor:pointer}
.print:hover{background:#67e8f9}
section{padding:24px 30px;border-top:1px solid #e2e8f0}
h2{margin:0 0 14px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#0e7490}
h3{margin:20px 0 8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#64748b}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #eef2f7;vertical-align:middle}
thead th{background:#f8fafc;color:#475569;font-size:11px;letter-spacing:.08em;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
.bar{height:8px;min-width:100px;border-radius:999px;background:#e2e8f0;overflow:hidden}
.bar i{display:block;height:100%;border-radius:999px}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;white-space:nowrap}
.kv th{width:42%;background:#f8fafc;color:#334155;font-weight:600}
.note{font-size:12px;color:#64748b;margin:0 0 12px}
.q{font-size:13.5px;color:#334155;margin:0 0 10px}
.q b{color:#0f172a}
.answer{white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;font-size:13.5px;line-height:1.7}
ul.src{margin:0;padding-left:18px;font-size:12.5px;color:#334155}
ul.src li{margin:7px 0;word-break:break-word;line-height:1.55}
ul.src a{color:#0e7490}
.empty{color:#94a3b8;font-style:italic;font-size:13.5px}
.foot{padding:18px 30px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11.5px;line-height:1.6;color:#64748b}
@media print{
  body{background:#fff}
  .sheet{margin:0;border:0;border-radius:0;box-shadow:none}
  .print{display:none}
  section,.hero,.foot,tr{break-inside:avoid}
}
`;

function buildSummaryReport(opts) {
  const {
    gravity,
    oxygen,
    pressure,
    airflow,
    fuel,
    ranking,
    rankNote,
    analysis,
    apiOnline,
    aiProvider,
    aiModel,
    generatedAt,
  } = opts;

  const mode = GRAVITY_MODES.find((g) => g.value === gravity);
  const stamp = generatedAt instanceof Date ? generatedAt : new Date();
  const rows = [...ranking].sort((a, b) => b.score - a.score);

  const settings = [
    ['Gravity mode', mode ? `${mode.label} - ${mode.sub}` : gravity],
    ['Oxygen concentration', `${oxygen} %`],
    ['Ambient pressure', `${pressure} kPa`],
    ['Forced airflow velocity', `${airflow} cm/s`],
    ['Fuel specimen', fuel],
  ]
    .map(
      ([label, value]) =>
        `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`
    )
    .join('');

  const matrix = rows
    .map((r) => {
      const hex = TIER_HEX[r.tier] || TIER_HEX.Low;
      const score = Math.round(r.score * 100);
      return (
        '<tr>' +
        `<td><b>${escapeHtml(r.material)}</b></td>` +
        `<td><span class="chip" style="background:${hex.bg};color:${hex.fg}">${escapeHtml(r.tier)} risk</span></td>` +
        `<td class="num">${score}</td>` +
        `<td><div class="bar"><i style="width:${score}%;background:${hex.bar}"></i></div></td>` +
        '</tr>'
      );
    })
    .join('');

  let analysisBlock;
  if (!analysis) {
    analysisBlock =
      '<p class="empty">No AI analysis has been produced yet. Ask a question in the ' +
      'AI agent query console, then export again.</p>';
  } else if (analysis.error) {
    analysisBlock =
      `<p class="empty">The latest assistant message was an error, so no analysis ` +
      `was captured: ${escapeHtml(analysis.text)}</p>`;
  } else {
    analysisBlock =
      (analysis.question
        ? `<p class="q"><b>Question:</b> ${escapeHtml(analysis.question)}</p>`
        : '') + `<div class="answer">${escapeHtml(analysis.text)}</div>`;
  }

  const sources = analysis?.sources?.length
    ? analysis.sources
    : analysis?.citations || [];
  const sourcesBlock = sources.length
    ? `<ul class="src">${sources
        .map((s) => {
          const bits = [escapeHtml(s.title || s.filename || 'NASA source')];
          if (s.page != null && s.page !== '') bits.push(`p. ${escapeHtml(s.page)}`);
          if (s.confidence != null) bits.push(`${pct(s.confidence)}% confidence`);
          const label = bits.join(' &middot; ');
          return s.url
            ? `<li>${label} - <a href="${escapeHtml(s.url)}" target="_blank" rel="noreferrer">${escapeHtml(s.url)}</a></li>`
            : `<li>${label}</li>`;
        })
        .join('')}</ul>`
    : '<p class="empty">No sources were attached to this analysis.</p>';

  const meta =
    `<span class="status">Gravity ${escapeHtml(gravity)}</span>` +
    `<span class="status">FastAPI ${apiOnline ? 'connected' : 'offline'}</span>` +
    `<span class="status">AI ${escapeHtml(aiModel ? `${aiProvider || 'configured'} · ${aiModel}` : 'RAG only')}</span>` +
    `<span>Generated ${escapeHtml(stamp.toLocaleString())}</span>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AETHON Executive Briefing</title>
<style>${REPORT_CSS}</style>
</head>
<body>
<div class="sheet">
  <header class="hero">
    <div class="brand">AETHON EXECUTIVE BRIEFING</div>
    <div class="sub">NASA Microgravity Combustion &amp; Fire Safety AI Platform</div>
    <div class="meta">${meta}</div>
    <button class="print" type="button" onclick="window.print()">Print / Save as PDF</button>
  </header>

  <section>
    <h2>1 &middot; Experiment settings</h2>
    <table class="kv"><tbody>${settings}</tbody></table>
  </section>

  <section>
    <h2>2 &middot; Material flammability risk matrix</h2>
    <p class="note">${escapeHtml(rankNote)}</p>
    <table>
      <thead>
        <tr>
          <th>Material</th>
          <th>Risk tier</th>
          <th style="text-align:right">Score</th>
          <th>0 - 100</th>
        </tr>
      </thead>
      <tbody>${matrix}</tbody>
    </table>
  </section>

  <section>
    <h2>3 &middot; Latest AI analysis</h2>
    ${analysisBlock}
    <h3>Retrieved sources</h3>
    ${sourcesBlock}
  </section>

  <div class="foot">
    Scores are 0-100 flammability estimates produced by the dashboard ranking
    heuristic (or the backend ranking when FastAPI is connected). They are
    illustrative, not verified NASA data. The flame behavior index table uses
    demo seed values and is not reproduced in this report.
  </div>
</div>
</body>
</html>`;
}

const DEFAULT_RANK_NOTE = 'Live estimate from the sliders. Run a simulation to query the backend.';

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */
export default function Dashboard() {
  // Global + experiment state
  const [gravity, setGravity] = useState('0g');
  const [oxygen, setOxygen] = useState(21);
  const [pressure, setPressure] = useState(101.3);
  const [airflow, setAirflow] = useState(15);
  const [fuel, setFuel] = useState(FUELS[0]);

  // Backend status and ranking
  const [apiOnline, setApiOnline] = useState(false);
  const [apiAddress, setApiAddress] = useState(null);
  const [indexReady, setIndexReady] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState('all-MiniLM-L6-v2');
  const [aiProvider, setAiProvider] = useState(null);
  const [aiModel, setAiModel] = useState(null);
  const [providerModels, setProviderModels] = useState({});
  const [running, setRunning] = useState(false);
  const [remoteRank, setRemoteRank] = useState(null);
  const [rankNote, setRankNote] = useState(DEFAULT_RANK_NOTE);
  // Last simulation error, shown as a dismissible banner in the controls panel.
  const [simError, setSimError] = useState(null);

  // Chat
  const [messages, setMessages] = useState([WELCOME]);
  const [draft, setDraft] = useState('');
  const [thinking, setThinking] = useState(false);
  const [openSource, setOpenSource] = useState(null);
  const messagesContainerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const experimentControlsRef = useRef(null);
  const chatPanelRef = useRef(null);
  // Header gains its glass bar once the page has scrolled.
  const [scrolled, setScrolled] = useState(false);
  // Animated starfield backdrop.
  const starsRef = useRef(null);
  // Automated simulation mode
  const [simulationMode, setSimulationMode] = useState(false);
  const [scenarioLabel, setScenarioLabel] = useState(null);
  const simIntervalRef = useRef(null);

  // Table
  const [showAllGravity, setShowAllGravity] = useState(false);
  const [highlightSelectedFuel, setHighlightSelectedFuel] = useState(true);
  const [riskFilter, setRiskFilter] = useState('All');
  const [sort, setSort] = useState({ key: 'spread', dir: 'desc' });

  const params = { oxygen, pressure, airflow, gravity, fuel };
  const paramsKey = `${oxygen}|${pressure}|${airflow}|${gravity}|${fuel}`;

  useEffect(() => {
    const controlsPanel = experimentControlsRef.current;
    const chatPanel = chatPanelRef.current;
    if (!controlsPanel || !chatPanel) return undefined;

    const syncChatHeight = () => {
      chatPanel.style.setProperty('--chat-panel-height', `${controlsPanel.getBoundingClientRect().height}px`);
    };
    const observer = new ResizeObserver(syncChatHeight);
    observer.observe(controlsPanel);
    window.addEventListener('resize', syncChatHeight);
    syncChatHeight();

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', syncChatHeight);
      chatPanel.style.removeProperty('--chat-panel-height');
    };
  }, []);

  // Any parameter change invalidates the last backend ranking. Done while
  // rendering rather than in an effect: same outcome, without the extra
  // cascading render flagged by react-hooks/set-state-in-effect.
  const [rankParams, setRankParams] = useState(paramsKey);
  if (rankParams !== paramsKey) {
    setRankParams(paramsKey);
    setRemoteRank(null);
    setRankNote(DEFAULT_RANK_NOTE);
  }

  // The status badge should reflect reality, so probe the backend on mount.
  // Prefer backends that report index_ready: true so RAG queries actually
  // return retrieved passages instead of "no information found" errors.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let firstReachable = null;
      let firstHealth = null;
      let indexReadyApi = null;
      let indexReadyHealth = null;

      for (const base of API_CANDIDATES) {
        try {
          const res = await fetch(`${base}/health`);
          if (res.ok && isJson(res)) {
            let health;
            try {
              health = await res.json();
            } catch {
              health = {};
            }
            if (!firstReachable) {
              firstReachable = base;
              firstHealth = health;
            }
            if (health?.index_ready === true && !indexReadyApi) {
              indexReadyApi = base;
              indexReadyHealth = health;
              break;
            }
          }
        } catch {
          /* nothing listening on this port - try the next one */
        }
      }

      if (!cancelled) {
        // Prefer an index-ready backend so RAG queries work; fall back to
        // any reachable backend (ranking still works without the index).
        activeApi = indexReadyApi || firstReachable;
        if (activeApi) {
          const health = indexReadyHealth || firstHealth || {};
          setApiOnline(true);
          setApiAddress(apiAddressLabel(activeApi));
          setIndexReady(health.index_ready === true);
          setEmbeddingModel(health.embedding_model || 'all-MiniLM-L6-v2');
          setAiProvider(
            health.ai_provider ||
              (health.groq_configured ? 'groq' : health.openrouter_configured ? 'openrouter' : null),
          );
          setAiModel(health.ai_model || health.groq_model || health.openrouter_model || null);
          setProviderModels({
            groq: health.groq_model || null,
            openrouter: health.openrouter_model || null,
          });
        } else {
          setApiOnline(false);
          setApiAddress(null);
          setIndexReady(false);
          setAiProvider(null);
          setAiModel(null);
          setProviderModels({});
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const container = messagesContainerRef.current;
    container?.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, thinking]);

  /* ---------------------------------------------------------------- */
  /* Space backdrop: twinkling stars and independent shooting stars.  */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    const starCanvas = starsRef.current;
    if (!starCanvas) return undefined;
    const context = starCanvas.getContext('2d');
    if (!context) return undefined;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const STAR_COLORS = ['#ffffff', '#cfe3ff', '#ffe7c7', '#e6d4ff', '#bfe9ff'];
    const MAX_SHOOTERS = 4;
    const TAU = Math.PI * 2;

    let W = window.innerWidth;
    let H = window.innerHeight;
    let stars = [];
    let shooters = [];
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      starCanvas.width = Math.round(W * dpr);
      starCanvas.height = Math.round(H * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      // Three depth layers: many small far stars, fewer big near ones.
      stars = Array.from({ length: Math.round((W * H) / 4200) }, () => {
        const layer = Math.random();
        const d = layer < 0.6 ? 0.15 : layer < 0.9 ? 0.45 : 0.9;
        return {
          x: Math.random() * W,
          y: Math.random() * H,
          r: (Math.random() * 1.1 + 0.25) * (0.7 + d * 0.8),
          p: Math.random() * 6.28,
          s: Math.random() * 1.8 + 0.4,
          d,
          c: STAR_COLORS[(Math.random() * STAR_COLORS.length) | 0],
        };
      });
    };

    const spawnShooter = () => {
      const dir = Math.random() < 0.7 ? 1 : -1;
      const ang = 0.45 + Math.random() * 0.5;
      const sp = 520 + Math.random() * 420;
      shooters.push({
        x: dir > 0 ? Math.random() * W * 0.8 : W * 0.2 + Math.random() * W * 0.8,
        y: -20 + Math.random() * H * 0.45,
        vx: Math.cos(ang) * sp * dir,
        vy: Math.sin(ang) * sp,
        sp,
        len: 110 + Math.random() * 130,
        l: 1,
      });
    };

    const frame = (t) => {
      const dt = Math.min((t - last) / 1000, 0.05);
      last = t;

      context.clearRect(0, 0, W, H);
      for (const s of stars) {
        s.p += dt * s.s;
        const alpha = 0.3 + 0.7 * Math.abs(Math.sin(s.p));
        context.globalAlpha = alpha;
        context.fillStyle = s.c;
        context.beginPath();
        context.arc(s.x, s.y, s.r, 0, TAU);
        context.fill();
        if (s.d > 0.8) {
          context.globalAlpha = alpha * 0.22;
          context.beginPath();
          context.arc(s.x, s.y, s.r * 3, 0, TAU);
          context.fill();
        }
      }
      context.globalAlpha = 1;

      /* falling stars: several can streak across at once */
      if (shooters.length < MAX_SHOOTERS && Math.random() < dt * 0.7) spawnShooter();
      for (let i = shooters.length - 1; i >= 0; i--) {
        const o = shooters[i];
        const tx = o.x - (o.vx / o.sp) * o.len;
        const ty = o.y - (o.vy / o.sp) * o.len;
        const g = context.createLinearGradient(o.x, o.y, tx, ty);
        g.addColorStop(0, `rgba(255,255,255,${o.l})`);
        g.addColorStop(0.35, `rgba(191,233,255,${o.l * 0.45})`);
        g.addColorStop(1, 'rgba(191,233,255,0)');
        context.strokeStyle = g;
        context.lineWidth = 1.8;
        context.lineCap = 'round';
        context.beginPath();
        context.moveTo(o.x, o.y);
        context.lineTo(tx, ty);
        context.stroke();
        context.fillStyle = `rgba(255,255,255,${o.l})`;
        context.beginPath();
        context.arc(o.x, o.y, 1.8, 0, TAU);
        context.fill();
        o.x += o.vx * dt;
        o.y += o.vy * dt;
        o.l -= dt * 0.55;
        if (o.l <= 0 || o.y > H + 40 || o.x < -200 || o.x > W + 200) shooters.splice(i, 1);
      }

      raf = requestAnimationFrame(frame);
    };

    resize();
    window.addEventListener('resize', resize);
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  // Sticky header: show the glass bar behind it once the page scrolls.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Auto Simulation (Demo Mode) cycles preset telemetry values while enabled.
  useEffect(() => {
    if (!simulationMode) {
      if (simIntervalRef.current) {
        clearInterval(simIntervalRef.current);
        simIntervalRef.current = null;
      }
      return undefined;
    }

    const conditions = [
      { oxygen: 21, pressure: 101.3, airflow: 15, label: 'Nominal cruise' },
      { oxygen: 28, pressure: 95.0, airflow: 8, label: 'High-oxygen cabin' },
      { oxygen: 19, pressure: 110.0, airflow: 3, label: 'Low-airflow quadrant' },
      { oxygen: 23, pressure: 101.3, airflow: 18, label: 'Nominal ventilation' },
      { oxygen: 30, pressure: 90.0, airflow: 6, label: 'Elevated O2 + low flow' },
    ];
    let idx = 0;
    simIntervalRef.current = setInterval(() => {
      const c = conditions[idx % conditions.length];
      setOxygen(c.oxygen);
      setPressure(c.pressure);
      setAirflow(c.airflow);
      setScenarioLabel(c.label);
      idx += 1;
    }, 4000);

    return () => {
      if (simIntervalRef.current) {
        clearInterval(simIntervalRef.current);
        simIntervalRef.current = null;
      }
    };
  }, [simulationMode]);

  // Combustion Precautions & Preventive Measures Alert System
  // Connects slider state directly to dynamic hazard warnings.
  const combustionAlert = useMemo(() => {
    const o2 = oxygen;
    const flow = airflow;

    if (o2 > 25) {
      return {
        active: true,
        level: 'critical',
        icon: '🔴',
        title: 'High Flammability Risk Alert',
        body: `Oxygen concentration at ${o2}% exceeds the 25% safety threshold. Elevated flame speed and spontaneous ignition hazard. Reduce O2, remove ignition sources, and verify material compatibility before proceeding.`,
      };
    }
    if (flow >= 0 && flow <= 5) {
      return {
        active: true,
        level: 'warning',
        icon: '🟡',
        title: 'Quenching & Radiative Extinction Warning',
        body: `Forced airflow at ${flow} cm/s is in the low convective flow hazard zone (0–5 cm/s). Flame quenching and radiative extinction risk increased; verify ventilation adequacy.`,
      };
    }
    if (flow >= 10 && flow <= 25) {
      return {
        active: true,
        level: 'nominal',
        icon: '🟢',
        title: 'Nominal Spacecraft Flow State',
        body: `Airflow at ${flow} cm/s is within nominal spacecraft ventilation parameters (10–25 cm/s). Calculated flame spread parameters remain within design limits.`,
      };
    }
    return { active: false };
  }, [oxygen, airflow]);

  const ranking = useMemo(
    () => remoteRank || localRank(params),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [remoteRank, oxygen, pressure, airflow, gravity]
  );

  const byTier = useMemo(() => {
    const out = { High: [], Medium: [], Low: [] };
    [...ranking].sort((a, b) => b.score - a.score).forEach((r) => out[r.tier]?.push(r));
    return out;
  }, [ranking]);

  // Top-risk result from the latest backend calculation (only when remoteRank is set).
  const topRisk = useMemo(
    () => (ranking.length ? [...ranking].sort((a, b) => b.score - a.score)[0] : null),
    [ranking],
  );

  const selectedId = fuelToId(fuel);

  // Latest assistant reply (excluding the welcome banner) captured in reports.
  const lastAnalysis = useMemo(
    () =>
      [...messages]
        .reverse()
        .find((m) => m.role === 'assistant' && m.id !== 'welcome'),
    [messages]
  );

  function exportSummary() {
    const now = new Date();
    const html = buildSummaryReport({
      gravity,
      oxygen,
      pressure,
      airflow,
      fuel,
      ranking,
      rankNote,
      analysis: lastAnalysis,
      apiOnline,
      aiProvider,
      aiModel,
      generatedAt: now,
    });
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `aethon-executive-briefing-${reportStamp(now)}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Revoke once the download has had a chance to start.
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function runSimulation() {
    setRunning(true);
    setSimError(null);
    let res = null;
    try {
      res = await apiFetch('/rank-materials', {
        oxygen_conc: oxygen,
        pressure,
        gravity,
        airflow,
        material: fuel,
      });
      if (!res.ok) {
        let body = '';
        try {
          body = await res.text();
        } catch {
          /* body unreadable - report status only */
        }
        console.error(`Simulation request failed: HTTP ${res.status}`, body);
        throw new Error(
          `HTTP ${res.status}${body ? ` - ${body.slice(0, 300)}` : ''}`,
        );
      }
      const parsed = normalizeRanking(await res.json());
      if (!parsed.length) throw new Error('Empty ranking');
      setRemoteRank(parsed);
      setSimError(null);
      setApiOnline(true);
      setApiAddress(activeApi ? apiAddressLabel(activeApi) : null);
      setRankNote(`Ranked by the backend at ${gravity}, ${oxygen}% O₂.`);
    } catch (err) {
      // Log to DevTools so a silent failure is never invisible, then surface
      // a banner in the UI and fall back to the local estimate.
      console.error('Simulation request failed:', err);
      setSimError(err instanceof Error ? err.message : String(err));
      setRemoteRank(null);
      if (err instanceof BackendConnectionError) setApiOnline(false);
      else if (err instanceof BackendResponseError) setApiOnline(true);
      setRankNote(
        'Backend unreachable or simulation failed. Showing the local estimate instead.',
      );
    } finally {
      setRunning(false);
    }
  }

  async function sendQuery(e) {
    e?.preventDefault();
    const query = draft.trim();
    if (!query || thinking) return;
    // Prior turns ride along with the request so the backend can replay them to
    // the AI model as conversation history. This is what lets it remember details the
    // user mentioned earlier (a name, a preference) instead of starting cold.
    // `messages` still excludes the message being sent, so the new turn is not
    // duplicated in the history.
    // `chat_history` uses `content` (matching the backend ChatMessage model) and
    // feeds the last 4 turns into the system prompt as conversation memory.
    const chatHistory = messages
      .filter((m) => m.id !== 'welcome' && !m.error && (m.text || '').trim())
      .slice(-4)
      .map((m) => ({ role: m.role === 'user' ? 'user' : 'model', content: m.text }));
    setMessages((m) => [...m, { id: `u${Date.now()}`, role: 'user', text: query }]);
    setDraft('');
    setThinking(true);
    try {
      const res = await apiFetch('/query-agent', {
        query,
        chat_history: chatHistory,
        gravity_filter: gravity,
        oxygen_conc: oxygen,
        pressure,
        airflow,
        fuel,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const a = normalizeAnswer(data);
      if (data.ai_used && providerModels[data.ai_provider]) {
        setAiProvider(data.ai_provider);
        setAiModel(providerModels[data.ai_provider]);
      }
      setApiOnline(true);
      setApiAddress(activeApi ? apiAddressLabel(activeApi) : null);
      setMessages((m) => [...m, { id: `a${Date.now()}`, role: 'assistant', ...a }]);
    } catch (err) {
      if (err instanceof BackendConnectionError) setApiOnline(false);
      else if (err instanceof BackendResponseError) setApiOnline(true);
      setMessages((m) => [
        ...m,
        {
          id: `a${Date.now()}`,
          role: 'assistant',
          error: true,
          text: `Backend request failed: ${err instanceof Error ? err.message : String(err)}`,
          citations: [],
          confidence: null,
        },
      ]);
    } finally {
      setThinking(false);
    }
  }

  const rows = useMemo(() => {
    let r = SEED_RUNS.filter(
      (x) => (showAllGravity || x.gravity === gravity) && (riskFilter === 'All' || x.risk === riskFilter)
    );
    const dir = sort.dir === 'asc' ? 1 : -1;
    r = [...r].sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      return (typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))) * dir;
    });
    return r;
  }, [gravity, showAllGravity, riskFilter, sort]);

  const columns = [
    { key: 'fuel', label: 'Fuel material' },
    { key: 'gravity', label: 'Gravity' },
    { key: 'o2', label: 'Oxygen %', num: true },
    { key: 'airflow', label: 'Airflow (cm/s)', num: true },
    { key: 'spread', label: 'Spread rate (mm/s)', num: true },
    { key: 'risk', label: 'Risk' },
    { key: null, label: 'Citation' },
  ];

  const toggleSort = (key) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }));

  return (
    <div className="aethon">
      {/* Backdrop: twinkling and falling stars */}
      <canvas ref={starsRef} id="stars" className="fx fx-stars" aria-hidden="true" />

      {/* Sticky header. The bar behind the pill is a sibling layer (not an
          ancestor) so the pill's own backdrop blur keeps working. */}
      <div className="sticky top-0 z-50 px-5 pb-3 pt-[calc(env(safe-area-inset-top,0px)_+_14px)] [@media(max-height:520px)]:static">
        <div
          aria-hidden
          className={`pointer-events-none absolute inset-0 -z-10 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md transition-opacity duration-300 ${
            scrolled ? 'opacity-100' : 'opacity-0'
          }`}
        />
        <header className="top glass mx-auto max-w-[1460px]">
          <div className="brand">
            <div className="logo" aria-label="AETHON flame orbit mark">
              <svg viewBox="0 0 64 64" aria-hidden="true" focusable="false">
                <defs>
                  <linearGradient id="aethon-flame" x1="0" x2="1" y1="1" y2="0">
                    <stop offset="0" stopColor="#F97316" />
                    <stop offset="1" stopColor="#38BDF8" />
                  </linearGradient>
                  <filter id="aethon-orbit-glow" x="-60%" y="-60%" width="220%" height="220%">
                    <feGaussianBlur stdDeviation="2.2" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <ellipse
                  cx="32"
                  cy="32"
                  rx="27"
                  ry="12"
                  transform="rotate(-32 32 32)"
                  fill="none"
                  stroke="#38BDF8"
                  strokeWidth="1.8"
                  filter="url(#aethon-orbit-glow)"
                />
                <ellipse
                  cx="32"
                  cy="32"
                  rx="22"
                  ry="9"
                  transform="rotate(42 32 32)"
                  fill="none"
                  stroke="#F97316"
                  strokeWidth="1.2"
                  opacity="0.85"
                />
                <path
                  d="M32 13c1 8-7 11-7 18 0 3 2 5 5 6-1-5 3-8 7-11 0 6 5 9 5 15a10 10 0 1 1-20 0c0-6 4-10 10-17Z"
                  fill="url(#aethon-flame)"
                  stroke="#fff"
                  strokeOpacity="0.7"
                  strokeWidth="1"
                  filter="url(#aethon-orbit-glow)"
                />
                <circle cx="7" cy="24" r="2.2" fill="#38BDF8" filter="url(#aethon-orbit-glow)" />
                <circle cx="56" cy="40" r="2.2" fill="#F97316" filter="url(#aethon-orbit-glow)" />
              </svg>
            </div>
            <div>
              <h1>AETHON</h1>
              <div className="sub">NASA Microgravity Combustion &amp; Fire Safety AI Platform</div>
            </div>
          </div>

          <div className="bar">
            <span className={`pill ${apiOnline ? 'p-em' : 'p-am'}`} title="FastAPI backend connection status">
              <span className={`dot${apiOnline ? '' : ' off'}`} aria-hidden />
              API Status: {apiOnline ? `Online · ${apiAddress || '127.0.0.1:8000'}` : 'Offline'}
            </span>

            <span
              className={`pill ${indexReady ? 'p-cy' : 'p-am'}`}
              title={indexReady ? 'ChromaDB index is ready for retrieval' : 'ChromaDB index is unavailable'}
            >
              <Database aria-hidden />
              Vector Index: {indexReady ? embeddingModel : 'Unavailable'}
            </span>

            <span
              title={
                aiProvider === 'groq'
                  ? `Groq model ${aiModel || 'configured'} with retrieval-augmented synthesis.`
                  : aiProvider === 'openrouter'
                    ? `OpenRouter model ${aiModel || 'configured'} with retrieval-augmented synthesis.`
                    : 'RAG Synthesizer is available in retrieval-only mode; configure an AI provider for generated answers.'
              }
              className={`pill ${aiProvider ? 'p-vi' : 'p-mu'}`}
            >
              <Sparkles aria-hidden />
              AI Engine: {aiModel ? `${aiModelLabel(aiModel)} / RAG Synthesizer` : 'RAG Synthesizer'}
            </span>

            <button
              type="button"
              onClick={exportSummary}
              title="Download an executive briefing with telemetry, risk rankings, latest AI analysis and citations"
              className="btn-ghost"
            >
              <Download aria-hidden />
              Export Executive Briefing
            </button>

            <div role="group" aria-label="Gravity mode" className="seg">
              {GRAVITY_MODES.map((g) => (
                <button
                  key={g.value}
                  type="button"
                  onClick={() => setGravity(g.value)}
                  aria-pressed={gravity === g.value}
                  className={gravity === g.value ? 'on' : ''}
                >
                  <b>{g.label}</b>
                  <small>{g.sub}</small>
                </button>
              ))}
            </div>
          </div>
        </header>
      </div>

      <div className="wrap pt-2">
        <main className="layout">
          {/* Left column */}
          <div className="left">
            <Panel
              title="Experiment controls"
              icon={Wind}
              a="#22d3ee"
              b="#a78bfa"
              panelRef={experimentControlsRef}
            >
              <div className="pb">
                {/* Combustion Precautions & Preventive Measures Alert Box */}
                {combustionAlert.active && (
                  <div
                    className={`alert-box ${combustionAlert.level}`}
                    role="alert"
                  >
                    <span className="alert-icon" aria-hidden>{combustionAlert.icon}</span>
                    <div className="alert-content">
                      <div className="alert-title">{combustionAlert.title}</div>
                      <div className="alert-body">{combustionAlert.body}</div>
                    </div>
                  </div>
                )}
                <Slider id="o2" label="Oxygen concentration" unit="%" value={oxygen} min={15} max={35} c1="#38BDF8" c2="#F97316" onChange={setOxygen} />
                <Slider id="p" label="Ambient pressure" unit="kPa" value={pressure} min={50} max={150} step={0.1} digits={1} c1="#a78bfa" c2="#f472b6" onChange={setPressure} />
                <Slider id="v" label="Forced airflow velocity" unit="cm/s" value={airflow} min={0} max={50} c1="#fbbf24" c2="#fb923c" onChange={setAirflow} />

                <div>
                  <label htmlFor="fuel" className="lbl mb-2 block">
                    Solid Fuel Selection
                  </label>
                  <select
                    id="fuel"
                    value={fuel}
                    onChange={(e) => setFuel(e.target.value)}
                    className="field"
                  >
                    {FUELS.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                </div>

                <div className="sim-controls">
                  <div className="sim-mode-status" role="status" aria-live="polite">
                    <span className={`dot${simulationMode ? '' : ' off'}`} aria-hidden />
                    <span>
                      <strong>Auto Simulation (Demo Mode)</strong>
                      <small>
                        {simulationMode
                          ? scenarioLabel
                            ? `Scenario: ${scenarioLabel}`
                            : 'Preparing first scenario…'
                          : scenarioLabel
                            ? `Stopped: ${scenarioLabel}`
                            : 'Ready to start scenario playback'}
                      </small>
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setScenarioLabel(null);
                      setSimulationMode((mode) => !mode);
                    }}
                    disabled={running}
                    className={`sim-toggle ${simulationMode ? 'active' : ''}`}
                    title="Cycle through preset telemetry scenarios; this demo mode does not run a physics solver."
                  >
                    <Play aria-hidden />
                    {simulationMode ? 'Stop Demo' : 'Start Demo'}
                  </button>
                  <button type="button" onClick={runSimulation} disabled={running || simulationMode} className="run">
                    {running ? <Loader2 className="animate-spin" aria-hidden /> : <Play fill="currentColor" aria-hidden />}
                    {running ? 'Calculating Risk…' : 'RUN SIMULATION & RANK RISK'}
                  </button>
                </div>
                {simError && (
                  <div className="sim-error" role="alert">
                    <AlertTriangle aria-hidden />
                    <span>Simulation failed: {simError}</span>
                    <button
                      type="button"
                      onClick={() => setSimError(null)}
                      className="sim-error-dismiss"
                    >
                      Dismiss
                    </button>
                  </div>
                )}
              </div>
            </Panel>

            {/* Backend simulation results (standalone card, between controls and matrix) */}
            {remoteRank && topRisk && (
              <Panel title="Backend simulation results" icon={Flame} a="#34d399" b="#22d3ee" iconColor="#34d399">
                <div className="pb">
                  <div className="sim-insight">
                    <div className="sim-insight-header">
                      <Flame aria-hidden />
                      Backend simulation results
                    </div>
                    <div className="sim-insight-body">
                      <span className="sim-insight-conditions">
                        Calculated at {gravity}, {oxygen}% O₂, {pressure} kPa, {airflow} cm/s airflow, fuel: {fuel}.
                      </span>
                    </div>
                    <div className="sim-insight-result">
                      <span>
                        Highest risk: <span className="material">{topRisk.material}</span> —{' '}
                        <span className="score">{Math.round(topRisk.score * 100)}</span>{' '}
                        <span className="tier">({topRisk.tier})</span>
                      </span>
                    </div>
                  </div>
                </div>
              </Panel>
            )}

            <Panel
              title="Material flammability risk matrix"
              icon={AlertTriangle}
              a="#fb7185"
              b="#fbbf24"
              iconColor="#fb7185"
              right={<span className="tag">{gravity}</span>}
            >
              <div className="pb flush">
                <div className="cols">
                  {['High', 'Medium', 'Low'].map((tier) => {
                    const t = TIERS[tier];
                    return (
                      <div key={tier} className="col" style={{ '--k': t.k }}>
                        <h3>
                          {t.title}
                          <span>{byTier[tier].length}</span>
                        </h3>
                        <p>{t.hint}</p>
                        <ul>
                          {byTier[tier].length === 0 && <li className="none">None at these settings</li>}
                          {byTier[tier].map((r) => (
                            <li key={r.material} className={r.material === selectedId ? 'sel' : ''}>
                              <div>
                                <span>{r.material}</span>
                                <em>{Math.round(r.score * 100)}</em>
                              </div>
                              <div className="bar2">
                                <span className="fill" style={{ width: `${r.score * 100}%` }} />
                              </div>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
                <p className="note">
                  <Sparkles aria-hidden />
                  {rankNote}
                </p>
              </div>
            </Panel>
          </div>

          {/* Right column: chat */}
          <Panel
            title="AI agent query console"
            icon={Sparkles}
            a="#a78bfa"
            b="#f472b6"
            iconColor="#a78bfa"
            className="chat right"
            panelRef={chatPanelRef}
            right={<span className="tag">Searching {gravity} records</span>}
          >
            <div ref={messagesContainerRef} className="msgs" aria-live="polite">
              {messages.map((m) => (
                <div key={m.id} className={`m${m.role === 'user' ? ' u' : ''}`}>
                  <span
                    className={`av ${m.role === 'user' ? 'user' : 'ai'}`}
                    aria-label={m.role === 'user' ? 'User' : 'AETHON assistant'}
                  >
                    {m.role === 'user' ? <User aria-hidden /> : <Flame aria-hidden />}
                  </span>
                  <div className={`bub${m.error ? ' err' : ''}`}>
                    <div className="message-copy">
                      {m.role === 'assistant' ? (
                        <AnswerText text={m.text} sources={m.sources} onOpen={setOpenSource} />
                      ) : (
                        <p className="whitespace-pre-wrap">{m.text}</p>
                      )}
                    </div>
                    {m.role === 'assistant' && m.id !== 'welcome' && !m.error && <Citation message={m} />}
                  </div>
                </div>
              ))}
              {thinking && (
                <div className="thinking">
                  <Loader2 className="animate-spin" aria-hidden /> Overthinking this so you don&apos;t have to…
                </div>
              )}
              <div ref={messagesEndRef} aria-hidden="true" />
            </div>

            <form onSubmit={sendQuery} className="composer">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Ask about flame spread, extinction limits or materials"
                aria-label="Ask the agent"
                autoComplete="off"
                className="field composer-input"
              />
              <button type="submit" disabled={!draft.trim() || thinking} className="send composer-send">
                <Send aria-hidden />
                Send
              </button>
            </form>
          </Panel>

          {/* Flame behavior index */}
          <Panel
            title="Flame behavior index"
            icon={Activity}
            a="#fbbf24"
            b="#22d3ee"
            iconColor="#fbbf24"
            className="full"
            right={
              <div className="tools">
                <label>
                  <input
                    type="checkbox"
                    checked={showAllGravity}
                    onChange={(e) => setShowAllGravity(e.target.checked)}
                  />
                  All gravity levels
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={highlightSelectedFuel}
                    onChange={(e) => setHighlightSelectedFuel(e.target.checked)}
                  />
                  Highlight selected fuel
                </label>
                <select
                  value={riskFilter}
                  onChange={(e) => setRiskFilter(e.target.value)}
                  aria-label="Filter by risk"
                  className="field"
                >
                  {['All', 'High', 'Medium', 'Low'].map((r) => (
                    <option key={r}>{r}</option>
                  ))}
                </select>
              </div>
            }
          >
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    {columns.map((c) => (
                      <th key={c.label} scope="col" className={c.num ? 'r' : ''}>
                        {c.key ? (
                          <button type="button" onClick={() => toggleSort(c.key)}>
                            {c.label}
                            {sort.key === c.key &&
                              (sort.dir === 'asc' ? <ArrowUp aria-hidden /> : <ArrowDown aria-hidden />)}
                          </button>
                        ) : (
                          c.label
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={7} className="empty">
                        No runs match these filters. Try “All gravity levels”.
                      </td>
                    </tr>
                  )}
                  {rows.map((r) => (
                    <tr
                      key={r.id}
                      className={highlightSelectedFuel && r.fuel === selectedId ? 'hl' : ''}
                    >
                      <td style={{ fontWeight: 500 }}>{r.fuel}</td>
                      <td className="mo">{r.gravity}</td>
                      <td className="mo r">{r.o2}</td>
                      <td className="mo r">{r.airflow}</td>
                      <td className="sp r">{r.spread.toFixed(1)}</td>
                      <td>
                        <RiskChip tier={r.risk} />
                      </td>
                      <td>
                        <a href={r.url} target="_blank" rel="noreferrer">
                          Source <ExternalLink aria-hidden />
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="foot">
              Demo seed data. Spread rates are placeholders until they are replaced with metrics extracted from the NASA reports.
            </p>
          </Panel>
        </main>
      </div>

      {openSource && (
        <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
      )}
    </div>
  );
}
