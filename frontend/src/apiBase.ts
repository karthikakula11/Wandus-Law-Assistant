/**
 * API root for fetch(). Default `/api` relies on Vite dev/preview proxy → FastAPI :8000.
 * If you still see "Failed to fetch", set in frontend/.env.development:
 *   VITE_API_BASE=http://127.0.0.1:8000
 * (no trailing slash; not .../api — paths are /health, /ingest/file, etc.)
 */
function normalizeApiBase(raw: string | undefined): string {
  const s = (raw ?? "").trim();
  if (!s) return "/api";
  return s.replace(/\/+$/, "");
}

export const API = normalizeApiBase(import.meta.env.VITE_API_BASE as string | undefined);
