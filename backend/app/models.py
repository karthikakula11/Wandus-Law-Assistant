import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector

from app.embedding_dims import CHUNK_EMBED_DIM, MEMORY_ITEM_EMBED_DIM

# Backwards compat name for chunk embeddings
EMBED_DIM = CHUNK_EMBED_DIM
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    """
    Registered app user (minimal). ``memory_user_key`` matches ``memory_items.user_key``
    for long-term memory; no password/auth in this phase.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    memory_user_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(CHUNK_EMBED_DIM), nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class MemoryItem(Base):
    """
    Per-browser or per-user long-term facts. ``user_key`` is an opaque id from the client
    (e.g. UUID in localStorage). Retrieval uses the same embedding model as document chunks.
    """

    __tablename__ = "memory_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(MEMORY_ITEM_EMBED_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserChatState(Base):
    """
    Full sidebar state (threads + messages) per ``memory_user_key``, synced from the client.
    Replaces reliance on browser-only ``localStorage`` for the same key.
    """

    __tablename__ = "user_chat_states"

    memory_user_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RetrievalDriftSample(Base):
    """
    One row per document-grounded chat turn: retrieval match strength (confidence proxy).
    Used for distribution drift (KS) vs a prior window — no gold labels required.
    """

    __tablename__ = "retrieval_drift_samples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)


class LLMUsageLog(Base):
    """
    One row per completed LLM call with token counts; ``cost_usd`` from local pricing cache
    (Langfuse model catalog) or backfilled from Langfuse trace ``totalCost``.
    """

    __tablename__ = "llm_usage_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    route: Mapped[str | None] = mapped_column(String(128), nullable=True)
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
