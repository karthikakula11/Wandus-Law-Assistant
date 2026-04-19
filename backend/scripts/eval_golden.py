#!/usr/bin/env python3
"""
Evaluate retrieval / chat modes on a JSONL golden file.

Each line: JSON with `question`, optional `expected_substring`, optional `document_title`.

Modes (env-driven, re-run script between modes):
- **dense**: `HYBRID_RAG_ENABLED=false`, `USE_LANGGRAPH_AGENT=false`
- **hybrid**: `HYBRID_RAG_ENABLED=true` (+ `OPENSEARCH_URL` for BM25 leg)
- **langgraph**: `USE_LANGGRAPH_AGENT=true`

Example:

    cd backend
    export DATABASE_URL=postgresql://...
    export OPENAI_API_KEY=...
    HYBRID_RAG_ENABLED=false USE_LANGGRAPH_AGENT=false python scripts/eval_golden.py scripts/golden_sample.jsonl

Exit code 0 always; prints pass rate and per-row notes to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def _run_one(
    session,
    question: str,
    top_k: int,
    history: list | None,
    row: dict,
) -> tuple[bool, str]:
    from app.services.rag import chat_auto

    answer, cites, _src = await chat_auto(session, question, top_k, history)
    ok = True
    notes: list[str] = []

    sub = (row.get("expected_substring") or "").strip()
    if sub and sub.lower() not in answer.lower():
        ok = False
        notes.append("substring_miss")

    want_title = (row.get("document_title") or "").strip()
    if want_title:
        titles = {c.document_title for c in cites}
        if want_title not in titles:
            ok = False
            notes.append("title_miss")

    return ok, ";".join(notes) if notes else "ok"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "golden",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "golden_sample.jsonl"),
        help="Path to JSONL file",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from app.config import clear_settings_cache, get_settings
    from app.database import get_session_factory

    clear_settings_cache()
    get_settings()

    rows: list[dict] = []
    with open(args.golden, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))

    factory = get_session_factory()
    passed = 0
    async with factory() as session:
        for row in rows:
            q = row["question"]
            ok, note = await _run_one(session, q, args.top_k, None, row)
            if ok:
                passed += 1
            print(f"{'PASS' if ok else 'FAIL'}\t{q[:60]!r}\t{note}")

    print(f"\nPass rate: {passed}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
