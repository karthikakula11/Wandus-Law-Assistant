"""LLM-generated short chat titles (ChatGPT-style sidebar)."""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.monitoring.openai_usage import record_chat_completion_usage
from app.services.langfuse_tracing import openai_trace_kwargs
from app.services.openai_factory import get_async_openai_client

router = APIRouter(prefix="/titles", tags=["titles"])

_TITLE_SYSTEM = """You name conversation threads for **Wandus**, a law-study assistant (like ChatGPT sidebar titles).

**What the title must reflect**
- The **substantive legal subject**: case / judgment / statute / issue the user is actually discussing — parties, dispute, court, remedy, limitation, title suit, etc.
- If **indexed document names** are listed below, the conversation is grounded in those files — your title MUST align with that **case or judgment topic**, not with meta phrases in the assistant text.

**Never use** vague or boilerplate titles such as:
- "Document Identification", "Identifying Documents", "Legal Assistance", "General Legal Chat", "Help with Documents", "Discussion of Materials", "Analysis Session"
- Any title that sounds like **software UI** instead of **law**.

**Do use** (pick what fits):
- Short **issue + context** from the user question and assistant (e.g. "Ayodhya Evidence & Limitation", "Limitation Under 1908 Act")
- Or a **short reference** derived from the judgment/case file name if provided (e.g. key words from the filename: year, court theme).

Output **only** the title — no quotes, no "Title:" prefix, no explanation.
**3–8 words**. No trailing period. English unless the user wrote in another language."""


class TitleSuggestBody(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=4000)
    assistant_message: str | None = Field(None, max_length=4000)
    """RAG source document titles (e.g. PDF filenames) — strongly weight the title toward these."""
    source_document_titles: Optional[list[str]] = Field(None, max_length=12)


class TitleSuggestResponse(BaseModel):
    title: str


_GENERIC_TITLE = re.compile(
    r"document\s+identif|identifying\s+document|legal\s+assist|"
    r"general\s+legal|help\s+with\s+doc|discussion\s+of\s+material|"
    r"analysis\s+session|chat\s+about\s+law|legal\s+chat$",
    re.I,
)


def _sanitize_title(raw: str) -> str:
    s = raw.strip().split("\n")[0].strip()
    s = s.strip("\"'“”")
    s = re.sub(r"^Title:\s*", "", s, flags=re.I)
    if len(s) > 72:
        s = s[:69] + "…"
    return s or "Legal chat"


def _humanize_source_filename(name: str) -> str:
    """Turn indexed PDF/title into a short sidebar label."""
    base = (name or "").strip()
    if not base:
        return "Legal chat"
    base = base.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    # Prefer text from "Judgement" / "Judgment" onward (case-style filenames)
    for needle in ("Judgement", "Judgment", "ORDER", "Order"):
        idx = base.find(needle)
        if idx >= 0:
            base = base[idx:]
            break
    else:
        # Strip long leading docket numbers (digits and short tokens)
        parts = base.split()
        while parts and (parts[0].isdigit() or len(parts[0]) <= 2):
            parts.pop(0)
        base = " ".join(parts) if parts else base
    if len(base) > 56:
        base = base[:53] + "…"
    return base[:72]


def _maybe_replace_generic(title: str, source_titles: list[str] | None) -> str:
    if not source_titles:
        return title
    clean = [t.strip() for t in source_titles if t and t.strip()]
    if not clean:
        return title
    if _GENERIC_TITLE.search(title):
        return _humanize_source_filename(clean[0])
    return title


@router.post("/suggest", response_model=TitleSuggestResponse)
async def suggest_title(body: TitleSuggestBody) -> TitleSuggestResponse:
    settings = get_settings()
    client = get_async_openai_client()
    parts = [f"User message:\n{body.user_message[:3500]}"]
    if body.assistant_message and body.assistant_message.strip():
        parts.append(
            f"Assistant reply (excerpt):\n{body.assistant_message.strip()[:2800]}"
        )
    if body.source_document_titles:
        uniq: list[str] = []
        seen: set[str] = set()
        for t in body.source_document_titles:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
            if len(uniq) >= 8:
                break
        if uniq:
            parts.append(
                "**Indexed sources cited in this reply (prioritize these for the thread name):**\n"
                + "\n".join(f"- {x}" for x in uniq)
            )
    user_block = "\n\n".join(parts)
    resp = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": _TITLE_SYSTEM},
            {"role": "user", "content": user_block},
        ],
        temperature=0.35,
        max_tokens=40,
        **openai_trace_kwargs(name="chat-thread-title"),
    )
    await record_chat_completion_usage(resp, route="chat-thread-title")
    raw = (resp.choices[0].message.content or "").strip()
    title = _sanitize_title(raw)
    title = _maybe_replace_generic(title, body.source_document_titles)
    return TitleSuggestResponse(title=title)
