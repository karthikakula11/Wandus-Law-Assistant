# Law chatbot (RAG)

FastAPI backend with **PostgreSQL + pgvector**, OpenAI embeddings and chat, optional **React** UI (Vite).

## Prerequisites

- Docker Desktop (for Postgres)
- Python 3.12+ (local backend)
- Node 20+ (frontend)
- OpenAI API key

## Setup

1. Copy [`.env.example`](.env.example) to `.env` and set `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, and `DATABASE_URL`.

2. Start the database (and optionally OpenSearch for hybrid BM25):

   ```bash
   docker compose up -d db
   # Optional — lexical BM25 + RRF (set OPENSEARCH_URL in `.env`, e.g. http://localhost:9200)
   docker compose up -d opensearch
   ```

3. Backend (from repo root):

   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   alembic upgrade head   # includes HNSW index on chunks.embedding (cosine) for dense retrieval
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Frontend (separate terminal):

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open the printed URL (e.g. `http://localhost:5173`). The UI calls the API via Vite’s `/api` proxy to `http://127.0.0.1:8000`.

## API

- `GET /health` — liveness
- `GET /ready` — DB + pgvector check
- `POST /ingest` — JSON `{ "title", "text", "source_uri?" }`
- `POST /ingest/file` — multipart form: `title`, optional `source_uri`, `file` (plain text UTF-8 **.txt** or **.pdf** text extraction via PyMuPDF)
- `POST /chat` — JSON `{ "question", "top_k", "mode", "history?" }`. Optional `history` is an array of `{ "role": "user"|"assistant", "content": "..." }` (last **24** turns sent to the model) for **conversation memory** in the LLM. The API does **not** persist chats; the **browser** saves the thread in `localStorage` so it survives refresh. Vector retrieval still uses the **current** `question` for embedding. `mode`: `auto` | `rag` | `general`. Response includes `source`: `documents` | `general`.

### Memory vs ChatGPT

| | **Wandus (this app)** | **ChatGPT** |
|--|----------------------|-------------|
| **Where history lives** | Browser `localStorage` + each request sends recent turns to the API | OpenAI servers, tied to your account |
| **Cross-device** | No (this browser only) | Yes when logged in |
| **How much context** | Last **24** message pairs sent to the model (plus token limits) | Very long threads + summarization on their side |
| **Server-side chat DB** | Not implemented (optional future work) | Yes |

Use **Clear chat** in the header to wipe saved memory in this browser.

## Agentic RAG flags (Jam-style pipeline)

| Variable | Default | Purpose |
|----------|---------|---------|
| `HYBRID_RAG_ENABLED` | `false` | When `true`, retrieval uses **dense pgvector + BM25 (OpenSearch) + RRF**. Requires `OPENSEARCH_URL` for the lexical leg; without it, the dense leg still runs inside RRF. |
| `BM25_ENABLED` | `true` | If `false`, BM25 calls are skipped (debug dense-only while hybrid is on). |
| `OPENSEARCH_URL` | unset | e.g. `http://localhost:9200`. Ingest upserts chunks here when set. |
| `RRF_K` | `60` | RRF rank constant. |
| `HYBRID_PER_LIST_CAP` | `50` | Max candidates per dense/BM25 list before fusion. |
| `USE_LANGGRAPH_AGENT` | `false` | When `true`, `/chat` uses a **LangGraph** flow (plan queries → hybrid retrieve per query → optional broaden → answer) after small-talk / empty-DB checks. |

**Cost / latency:** Hybrid adds one OpenSearch request per retrieval; LangGraph adds planner (and sometimes broaden) LLM calls. Enable incrementally: hybrid first, then LangGraph.

**Backfill search index** (after enabling OpenSearch on an existing DB):

```bash
cd backend && export DATABASE_URL=... OPENSEARCH_URL=... && python scripts/reindex_opensearch.py
```

**Golden eval** (compare modes via env before running):

```bash
cd backend
HYBRID_RAG_ENABLED=false USE_LANGGRAPH_AGENT=false python scripts/eval_golden.py scripts/golden_sample.jsonl
```

## Docker API (optional)

With DB credentials in `.env` (host must be `db` inside Compose):

```bash
docker compose --profile full up -d --build
```

Set `DATABASE_URL` in `.env` to use `db` as host when the API runs in Compose, or rely on `docker-compose.yml` `environment` override for the `api` service.

## Tests

```bash
cd backend
pytest -q -m "not integration"    # CI (no Postgres)
pytest -q                         # all tests (needs Docker DB + .env)
```

## Disclaimer

This is a demonstration system. It is **not** legal advice. Verify outputs against official sources.
