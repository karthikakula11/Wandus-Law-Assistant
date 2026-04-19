from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IngestBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)
    source_uri: str | None = Field(None, max_length=2048)


class IngestResponse(BaseModel):
    document_id: UUID
    chunks_created: int


class DocumentListItem(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


class ChunkPreviewItem(BaseModel):
    chunk_index: int
    char_count: int
    excerpt: str = Field(..., description="Start of chunk text for debugging ingest quality")


class DocumentChunksPreviewResponse(BaseModel):
    document_id: UUID
    title: str
    chunks: list[ChunkPreviewItem]


class CitationOut(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    chunk_index: int
    excerpt: str


class ChatHistoryItem(BaseModel):
    """Prior turns in this chat session (client-held). Max 24 items enforced server-side."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=20)
    # Conversation memory: prior user/assistant messages (this request). Sidebar threads are stored via ``/users/chat-state``.
    history: list[ChatHistoryItem] | None = None
    # When set, retrieval is limited to this document (also inferred from history / filenames).
    document_id: UUID | None = None
    # Opaque stable id (e.g. UUID in localStorage) for server-side long-term semantic memory.
    memory_user_id: str | None = Field(
        None,
        max_length=64,
        description="If set, retrieve stored facts for this id and inject into the prompt.",
    )
    # When True (default), server may append extracted facts to long-term memory after the reply.
    auto_memory: bool = Field(
        True,
        description="If False, skip automatic conversation→memory extraction for this request.",
    )


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    # Where the reply came from (for UI)
    source: Literal["documents", "general"] = "general"


class MemoryItemCreate(BaseModel):
    memory_user_id: str = Field(..., min_length=8, max_length=64)
    content: str = Field(..., min_length=1, max_length=4000)


class MemoryItemOut(BaseModel):
    id: UUID
    content: str
    created_at: datetime


class MemoryListResponse(BaseModel):
    items: list[MemoryItemOut]


class UserOut(BaseModel):
    id: UUID
    username: str
    display_name: str | None
    memory_user_key: str
    created_at: datetime


class ChatStoredMessageIn(BaseModel):
    """One message in a client-held thread (matches frontend ``StoredMessage``)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., max_length=128)
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=120_000)
    citations: list | None = None
    source: Literal["documents", "general"] | None = None


class ChatThreadIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., max_length=128)
    title: str = Field(..., max_length=512)
    updatedAt: str = Field(..., max_length=80)
    messages: list[ChatStoredMessageIn] = Field(default_factory=list, max_length=150)
    activeDocumentId: str | None = Field(None, max_length=128)


class ChatThreadsStateIn(BaseModel):
    """Full sidebar state (``v: 2``) synced from the browser."""

    v: Literal[2]
    threads: list[ChatThreadIn] = Field(..., max_length=50)
    activeThreadId: str = Field(..., max_length=128)

    @model_validator(mode="after")
    def active_id_matches(self) -> "ChatThreadsStateIn":
        ids = {t.id for t in self.threads}
        if self.activeThreadId not in ids:
            raise ValueError("activeThreadId must match a thread id")
        return self
