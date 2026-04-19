# Retrieval benchmark

1. Ingest at least one document in the Knowledge panel (or the seed **Sample Act** text from the UI).
2. Edit `data/benchmark.json`: each item needs a **question** and either:
   - `gold_chunk_ids`: list of chunk UUIDs, or
   - `gold_document_title` + `gold_chunk_index` (matches your DB).
3. From the `backend/` directory:

```bash
.venv/bin/python scripts/run_eval.py
```

4. Results are written to `eval/results/latest.json`. The API `GET /eval/summary` and the **Evaluation** page in the UI read that file.

**Re-run after edits:** run the script again, then refresh the Evaluation page (restart the API only if `/eval/summary` is missing).

**`gold_auto`:** items can set `"gold_auto": true` with `gold_document_title` (optional). The first matching chunk is used; if none match, the first chunk in the DB is used (demo only — set real `gold_chunk_ids` for a proper evaluation).

**Duplicate chunks:** the same PDF text may be stored under **multiple chunk rows** (different UUIDs). If your gold UUID never appears in top‑k but another row with the same text does, either list **all** duplicate UUIDs in `gold_chunk_ids` or pick the UUID that your retriever actually returns (inspect `per_question` in `latest.json` after a run).

**Metrics**

- **Basic path**: single dense retrieval (`retrieve_chunks`) on the raw question.
- **Agentic path**: LangGraph (`plan → retrieve → grade → … → generate`); retrieval uses **final** `rows` before generate.

**Recall@k** here means: at least one gold chunk id appears in the top‑k retrieved chunk ids for that path.
