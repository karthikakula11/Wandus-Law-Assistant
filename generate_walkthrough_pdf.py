"""
Generate Wandus-Code-Walkthrough.pdf for presentations.
Run from repo root: python generate_walkthrough_pdf.py
"""
from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent


def slice_lines(rel_path: str, start: int, end: int) -> str:
    """1-based inclusive line range, with line numbers prefix."""
    path = ROOT.joinpath(*rel_path.split("/"))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    for i in range(start - 1, min(end, len(lines))):
        out.append(f"{i + 1:4}  {lines[i]}")
    return "\n".join(out)


SECTIONS: list[tuple[str, str, str, int, int]] = [
    (
        "Block 1 — App startup (lifespan, Langfuse, OpenSearch, pricing loop)",
        "Runs once at process start and cleans up on shutdown.",
        "backend/app/main.py",
        20,
        57,
    ),
    (
        "Block 2 — FastAPI app: middleware, /health, /ready, all routers",
        "Single factory registers CORS, request IDs, and every feature router.",
        "backend/app/main.py",
        60,
        111,
    ),
    (
        "Block 3 — Settings from .env (Pydantic BaseSettings)",
        "Typed configuration: DATABASE_URL, OpenAI keys, models, memory flags.",
        "backend/app/config.py",
        11,
        43,
    ),
    (
        "Block 4 — Async SQLAlchemy engine and get_session dependency",
        "Converts postgresql:// to asyncpg URL; session per HTTP request.",
        "backend/app/database.py",
        12,
        41,
    ),
    (
        "Block 4b — get_session generator",
        "",
        "backend/app/database.py",
        53,
        56,
    ),
    (
        "Block 5 — Chat router: JSON POST /chat",
        "prepare_retrieval_scope → chat_dispatch → optional auto_memory → response.",
        "backend/app/routers/chat.py",
        21,
        53,
    ),
    (
        "Block 5b — Streaming POST /chat/stream (SSE)",
        "Yields data: {json}\\n\\n events for tokens and metadata.",
        "backend/app/routers/chat.py",
        66,
        109,
    ),
    (
        "Block 6 — RAG: memory snippets, chunk count, vector retrieval",
        "embed_texts + cosine_distance on Chunk.embedding; optional document scope.",
        "backend/app/services/rag.py",
        40,
        85,
    ),
    (
        "Block 7 — Ingest router: JSON and PDF multipart",
        "Delegates to ingest_document service after PDF text extraction.",
        "backend/app/routers/ingest.py",
        12,
        55,
    ),
    (
        "Block 8 — Vite dev server: /api proxy to FastAPI",
        "Browser uses /api/...; Vite strips prefix and forwards to port 8000.",
        "frontend/vite.config.ts",
        4,
        18,
    ),
    (
        "Block 9 — Frontend SSE consumer (consumeChatSse)",
        "Reads fetch stream, splits on blank line, parses data: JSON events.",
        "frontend/src/App.tsx",
        32,
        79,
    ),
]


def build_html() -> str:
    parts: list[str] = [
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<title>Wandus Law RAG — Code walkthrough</title>
<style type="text/css">
  @page { size: a4; margin: 1.5cm; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #222; }
  h1 { font-size: 18pt; margin-bottom: 0.3em; }
  h2 { font-size: 12pt; margin-top: 1.2em; page-break-after: avoid; color: #111; }
  p.blurb { font-size: 9pt; color: #444; margin: 0.2em 0 0.6em 0; }
  p.file { font-size: 8pt; font-family: Courier, monospace; color: #666; }
  pre.code {
    font-family: Courier, monospace;
    font-size: 7pt;
    background: #f4f4f4;
    border: 1px solid #ddd;
    padding: 8px;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
  }
  .footer { font-size: 8pt; color: #888; margin-top: 2em; }
</style>
</head>
<body>
<h1>Wandus Law RAG — Block-wise code walkthrough</h1>
<p class="blurb">Generated from the chatbot-pintu-demo repository for presentations. Each block maps to one slide or section.</p>
<p class="file">Repository root: """ + escape(str(ROOT)) + """</p>
"""
    ]
    for title, blurb, rel, lo, hi in SECTIONS:
        code = slice_lines(rel, lo, hi)
        parts.append(f"<h2>{escape(title)}</h2>")
        if blurb:
            parts.append(f'<p class="blurb">{escape(blurb)}</p>')
        parts.append(f'<p class="file">{escape(rel)} (lines {lo}–{hi})</p>')
        parts.append(f'<pre class="code">{escape(code)}</pre>')
    parts.append(
        '<p class="footer">End of document. Re-run: python generate_walkthrough_pdf.py</p>'
        "</body></html>"
    )
    return "".join(parts)


def main() -> None:
    html = build_html()
    out = ROOT / "Wandus-Code-Walkthrough.pdf"
    buf = BytesIO()
    status = pisa.CreatePDF(
        BytesIO(html.encode("utf-8")),
        dest=buf,
        encoding="utf-8",
    )
    if status.err:
        raise SystemExit(f"xhtml2pdf reported errors: {status.err}")
    out.write_bytes(buf.getvalue())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
