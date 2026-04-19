"""
Keep prompts within a safe size so gpt-4o-mini requests stay under practical context limits.
Uses character budgets (~4 chars ≈ 1 token for Latin text).
"""

from app.models import Chunk, Document

# History: last turns only; drop oldest until under cap
MAX_HISTORY_MESSAGES = 24
MAX_HISTORY_TOTAL_CHARS = 14_000

# Retrieved legal text: cap total excerpt size
MAX_RAG_CONTEXT_CHARS = 36_000
MAX_SINGLE_CHUNK_CHARS = 4_000


def budget_history_messages(history: list[dict]) -> list[dict]:
    """Trim oldest messages until total character count fits (after per-msg cap)."""
    if not history:
        return []
    out: list[dict] = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content[:8000]})

    while out and sum(len(m["content"]) for m in out) > MAX_HISTORY_TOTAL_CHARS:
        out = out[1:]

    return out


def budget_rag_contexts(
    contexts: list[tuple[Chunk, Document]],
) -> tuple[list[tuple[Chunk, Document]], list[str]]:
    """
    Include chunks in retrieval order until MAX_RAG_CONTEXT_CHARS.
    Returns (included pairs, trimmed text per chunk for the prompt).
    """
    included: list[tuple[Chunk, Document]] = []
    texts: list[str] = []
    used = 0

    for ch, doc in contexts:
        raw = (ch.content or "").strip()
        max_this = min(MAX_SINGLE_CHUNK_CHARS, max(0, MAX_RAG_CONTEXT_CHARS - used - 120))
        if max_this < 200:
            break
        piece = raw[:max_this]
        if len(raw) > max_this:
            piece = piece.rstrip() + "…[truncated]"
        included.append((ch, doc))
        texts.append(piece)
        used += len(piece) + 100
        if used >= MAX_RAG_CONTEXT_CHARS:
            break

    return included, texts
