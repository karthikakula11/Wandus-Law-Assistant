"""Extract user-specific facts from a chat turn into ``memory_items`` (ChatGPT-style)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.monitoring.openai_usage import record_chat_completion_usage
from app.services.langfuse_tracing import openai_trace_kwargs
from app.services.long_term_memory import (
    add_memory_item,
    nearest_memory_cosine_distance,
    normalize_user_key,
)
from app.services.openai_factory import (
    get_async_ollama_openai_client,
    get_async_openai_client,
)

logger = logging.getLogger(__name__)

# Below this cosine distance, treat as duplicate (same semantic neighborhood).
AUTO_MEMORY_DUPLICATE_MAX_DISTANCE = 0.16
_MAX_EXTRACT_CHARS = 2200
_MAX_BULLETS = 2
_MAX_BULLET_CHARS = 400

_SYSTEM = """You extract durable, user-specific facts to remember across future chats for a law-study assistant.

Output strict JSON: {"memories": string[]}

Rules:
- At most 2 items. Each item one short sentence, third person or neutral ("User …" / "They …" / "Prefers …").
- Only facts **about the user** (goals, exams, language, jurisdiction, preferences, background they stated).
- Do **not** store legal rules, case holdings, statutes, or content that belongs in case notes instead of memory.
- Do **not** store greetings, thanks, or one-off chitchat with no lasting value.
- If nothing qualifies, return {"memories": []}.

Respond with JSON only, no markdown."""


def _maybe_strip_code_fence(raw: str) -> str:
    """Ollama sometimes wraps JSON in ```json fences."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() in ("```", ""):
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        else:
            break
    return "\n".join(lines).strip()


def _parse_memories_json(raw: str) -> list[str]:
    raw = _maybe_strip_code_fence(raw or "")
    if not raw:
        return []
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    arr = data.get("memories")
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for x in arr[:_MAX_BULLETS]:
        if not isinstance(x, str):
            continue
        s = re.sub(r"\s+", " ", x.strip())
        if not s:
            continue
        if len(s) > _MAX_BULLET_CHARS:
            s = s[: _MAX_BULLET_CHARS - 1] + "…"
        low = s.lower()
        if any(existing.lower() == low for existing in out):
            continue
        out.append(s)
    return out


async def _ollama_chat_extract(messages: list[dict[str, str]]) -> Any:
    """Call local Ollama via OpenAI-compatible ``/v1/chat/completions``."""
    settings = get_settings()
    client = get_async_ollama_openai_client()
    model = settings.ollama_model
    common = {
        "model": model,
        "temperature": 0,
        "max_tokens": 350,
        "messages": messages,
    }
    try:
        return await client.chat.completions.create(
            response_format={"type": "json_object"},
            **common,
        )
    except Exception as e:
        logger.warning(
            "Ollama auto-memory: json_object mode failed (%s), retrying without it",
            e,
        )
        return await client.chat.completions.create(**common)


async def extract_memory_bullets_from_turn(
    user_message: str,
    assistant_message: str,
) -> list[str]:
    """Single LLM call: 0–2 memory strings, or []."""
    settings = get_settings()
    u = (user_message or "")[:_MAX_EXTRACT_CHARS]
    a = (assistant_message or "")[:_MAX_EXTRACT_CHARS]
    if len(u.strip()) < 3 or len(a.strip()) < 10:
        return []

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"User message:\n{u}\n\nAssistant reply:\n{a}",
        },
    ]

    if settings.auto_memory_llm == "ollama":
        resp = await _ollama_chat_extract(messages)
    else:
        client = get_async_openai_client()
        resp = await client.chat.completions.create(
            model=settings.openai_chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=350,
            messages=messages,
            **openai_trace_kwargs(name="auto-memory-extract"),
        )
        await record_chat_completion_usage(resp, route="auto-memory-extract")

    content = (resp.choices[0].message.content or "").strip()
    return _parse_memories_json(content)


async def persist_auto_memory_from_turn(
    session: AsyncSession,
    memory_user_id: str | None,
    question: str,
    answer: str,
) -> int:
    """
    Extract bullets and insert non-duplicate rows. Returns number of new items stored.
    """
    uk = normalize_user_key(memory_user_id)
    if not uk:
        return 0
    try:
        bullets = await extract_memory_bullets_from_turn(question, answer)
    except Exception:
        logger.exception("auto-memory extraction failed")
        return 0
    if not bullets:
        return 0

    inserted = 0
    for text in bullets:
        try:
            d = await nearest_memory_cosine_distance(session, uk, text)
            if d is not None and d < AUTO_MEMORY_DUPLICATE_MAX_DISTANCE:
                continue
            await add_memory_item(session, uk, text)
            await session.flush()
            inserted += 1
        except Exception:
            logger.exception("auto-memory insert failed for one bullet")
    return inserted
