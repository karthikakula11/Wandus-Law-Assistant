#!/usr/bin/env python3
"""Run basic vs agentic retrieval benchmark; write eval/results/latest.json."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure ``backend/`` is on path when run as ``python scripts/run_eval.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def _main() -> None:
    from sqlalchemy import select

    from app.config import get_settings
    from app.database import get_session_factory
    from app.models import Chunk
    from app.services.eval_retrieval import (
        default_benchmark_path,
        default_results_path,
        mean_metric,
        recall_hit,
        resolve_gold_chunk_ids,
        retrieval_previews_for_ids,
        run_agentic_retrieval_and_trace,
        run_basic_retrieval,
    )

    settings = get_settings()
    bench_path = default_benchmark_path()
    if not bench_path.exists():
        print(f"Missing benchmark file: {bench_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(bench_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    top_k = int(data.get("top_k", 5))

    out_dir = default_results_path().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = get_session_factory()
    async with factory() as session:
        n_chunks = (await session.execute(select(Chunk.id).limit(1))).first()
        if n_chunks is None:
            print("No chunks in DB — ingest documents first.", file=sys.stderr)
            sys.exit(2)

        per_q: list[dict] = []
        basic_hits: list[bool] = []
        agent_hits: list[bool] = []

        for item in items:
            qid = str(item.get("id", ""))
            question = (item.get("question") or "").strip()
            gold = await resolve_gold_chunk_ids(session, item)
            if not question:
                per_q.append(
                    {
                        "id": qid,
                        "question": question,
                        "skipped": True,
                        "reason": "empty question",
                    }
                )
                continue
            if not gold:
                per_q.append(
                    {
                        "id": qid,
                        "question": question,
                        "skipped": True,
                        "reason": "gold chunk not found — fix gold_document_title / gold_chunk_ids in benchmark.json to match your DB",
                    }
                )
                continue

            basic_ids = await run_basic_retrieval(
                session, question, top_k, scope_document_ids=None
            )
            agent_ids, trace = await run_agentic_retrieval_and_trace(
                session, question, top_k, scope_document_ids=None
            )

            bh = recall_hit(gold, basic_ids, k=top_k)
            ah = recall_hit(gold, agent_ids, k=top_k)
            basic_hits.append(bh)
            agent_hits.append(ah)

            basic_prev = await retrieval_previews_for_ids(session, basic_ids)
            agent_prev = await retrieval_previews_for_ids(session, agent_ids)

            per_q.append(
                {
                    "id": qid,
                    "question": question,
                    "gold_chunk_ids": [str(x) for x in gold],
                    "basic_retrieved_ids": [str(x) for x in basic_ids],
                    "basic_recall_hit": bh,
                    "basic_retrieval_chunks": basic_prev,
                    "agentic_retrieved_ids": [str(x) for x in agent_ids],
                    "agentic_recall_hit": ah,
                    "agentic_retrieval_chunks": agent_prev,
                    "agentic_trace": trace,
                }
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(bench_path.relative_to(_ROOT)),
        "settings_note": "Uses current .env (e.g. HYBRID_RAG_ENABLED, USE_LANGGRAPH_AGENT).",
        "openai_chat_model": settings.openai_chat_model,
        "top_k": top_k,
        "n_evaluated": len(basic_hits),
        "basic": {
            "recall_at_k_mean": mean_metric(basic_hits),
            "description": "Single-query dense retrieval (retrieve_chunks)",
        },
        "agentic": {
            "recall_at_k_mean": mean_metric(agent_hits),
            "description": "LangGraph agent: plan → retrieve → grade → … → generate (final retrieval)",
        },
        "per_question": per_q,
        "methodology": [
            "Recall@k: at least one gold chunk id appears in the top-k retrieved chunk ids.",
            "Basic path: one embedding of the user question; dense top-k only.",
            "Agentic path: LangGraph may plan/rewrite queries and merge retrieval before answering.",
            "Side-by-side excerpts show retrieved chunk text (context for the LLM), not the final generated answer.",
            "Agentic is not always better — compare excerpts and recall; sometimes both paths retrieve the same chunks.",
        ],
    }

    out_path = default_results_path()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    b = payload["basic"]["recall_at_k_mean"]
    a = payload["agentic"]["recall_at_k_mean"]
    print(f"Recall@k mean — basic: {b}, agentic: {a}")


if __name__ == "__main__":
    asyncio.run(_main())
