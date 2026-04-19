from functools import lru_cache
from typing import Literal

MemoryEmbeddingProvider = Literal["openai", "sentence_transformers"]
AutoMemoryLlmProvider = Literal["openai", "ollama"]

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(validation_alias="DATABASE_URL")
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_CHAT_MODEL",
    )
    embedding_dimensions: int = Field(
        default=1536,
        validation_alias="EMBEDDING_DIMENSIONS",
    )

    # Long-term memory vectors: size is ``app.embedding_dims.MEMORY_ITEM_EMBED_DIM`` (384);
    # use OpenAI ``text-embedding-3-*`` with reduced dimensions, or free local Sentence-Transformers.
    memory_embedding_provider: MemoryEmbeddingProvider = Field(
        default="openai",
        validation_alias="MEMORY_EMBEDDING_PROVIDER",
    )
    sentence_transformer_memory_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="SENTENCE_TRANSFORMER_MEMORY_MODEL",
    )

    # After each reply, extract 0–2 user-specific facts into ``memory_items`` (LLM call).
    auto_memory_from_conversation: bool = Field(
        default=True,
        validation_alias="AUTO_MEMORY_FROM_CONVERSATION",
    )
    # Which LLM runs that extraction: OpenAI (paid) or local Ollama (free, OpenAI-compatible API).
    auto_memory_llm: AutoMemoryLlmProvider = Field(
        default="openai",
        validation_alias="AUTO_MEMORY_LLM",
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        validation_alias="OLLAMA_MODEL",
    )

    # PDF — Jam course uses Docling (``pdf_parser/docling.py``). Falls back to PyMuPDF if off/unavailable.
    use_docling_for_pdf: bool = Field(
        default=True,
        validation_alias="USE_DOCLING_FOR_PDF",
    )
    docling_max_pages: int = Field(
        default=100,
        ge=1,
        le=500,
        validation_alias="DOCLING_MAX_PAGES",
    )

    # Chunking — defaults match course ``TextChunker`` (text_chunker.py: 600 words / 100 overlap)
    chunking_strategy: Literal["structure", "word", "structure_word"] = Field(
        default="structure",
        validation_alias="CHUNKING_STRATEGY",
    )
    chunk_words: int = Field(
        default=600,
        ge=50,
        le=4000,
        validation_alias="CHUNK_WORDS",
    )
    chunk_overlap_words: int = Field(
        default=100,
        ge=0,
        le=2000,
        validation_alias="CHUNK_OVERLAP_WORDS",
    )
    chunk_min_words: int = Field(
        default=100,
        ge=1,
        le=500,
        validation_alias="CHUNK_MIN_WORDS",
    )
    chunk_size_chars: int = Field(
        default=1200,
        ge=200,
        le=16000,
        validation_alias="CHUNK_SIZE_CHARS",
    )
    chunk_overlap_chars: int = Field(
        default=200,
        ge=0,
        le=4000,
        validation_alias="CHUNK_OVERLAP_CHARS",
    )

    # Jam with AI–style RAG: use top-k retrieved chunks whenever any exist (no cosine-distance fallback).
    # Set RAG_DISTANCE_GATE_ENABLED=true to restore the old behavior (skip excerpts when best_dense > threshold).
    rag_distance_gate_enabled: bool = Field(
        default=False,
        validation_alias="RAG_DISTANCE_GATE_ENABLED",
    )
    rag_distance_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        validation_alias="RAG_DISTANCE_THRESHOLD",
    )

    # Hybrid RAG: dense (pgvector) + BM25 (OpenSearch) + RRF
    hybrid_rag_enabled: bool = Field(
        default=False,
        validation_alias="HYBRID_RAG_ENABLED",
    )
    bm25_enabled: bool = Field(
        default=True,
        validation_alias="BM25_ENABLED",
    )
    opensearch_url: str | None = Field(
        default=None,
        validation_alias="OPENSEARCH_URL",
    )
    # Index name; use a **new** name (e.g. law_chunks_v2) when enabling unified hybrid on an existing cluster.
    opensearch_index_name: str = Field(
        default="law_chunks",
        validation_alias="OPENSEARCH_INDEX_NAME",
    )
    # Jam course: native hybrid + RRF in OpenSearch (``client.py`` ``search_unified``). Requires knn mapping + re-ingest.
    opensearch_unified_hybrid: bool = Field(
        default=False,
        validation_alias="OPENSEARCH_UNIFIED_HYBRID",
    )
    rrf_k: int = Field(default=60, validation_alias="RRF_K")
    hybrid_per_list_cap: int = Field(
        default=50,
        validation_alias="HYBRID_PER_LIST_CAP",
    )

    # LangGraph agent: plan → retrieve → grade (Jam-style) → rewrite loop → optional broaden → generate
    use_langgraph_agent: bool = Field(
        default=True,
        validation_alias="USE_LANGGRAPH_AGENT",
    )
    # LLM grades retrieved chunks before answering; if not relevant, rewrite queries and re-retrieve (bounded).
    rag_agentic_grade_enabled: bool = Field(
        default=True,
        validation_alias="RAG_AGENTIC_GRADE_ENABLED",
    )
    # Max rewrite→retrieve rounds after the first retrieval (0 = no rewrite loop).
    rag_agent_max_rewrite_rounds: int = Field(
        default=2,
        ge=0,
        le=8,
        validation_alias="RAG_AGENT_MAX_REWRITE_ROUNDS",
    )
    # Skip the plan LLM and search with the raw user question only (saves ~1 LLM round-trip).
    rag_agent_skip_plan_llm: bool = Field(
        default=False,
        validation_alias="RAG_AGENT_SKIP_PLAN_LLM",
    )

    # Multi-document shortlist (custom; Jam-style default = search full corpus like one OpenSearch index)
    document_router_enabled: bool = Field(
        default=False,
        validation_alias="DOCUMENT_ROUTER_ENABLED",
    )
    document_router_pool_chunks: int = Field(
        default=500,
        ge=50,
        le=5000,
        validation_alias="DOCUMENT_ROUTER_POOL_CHUNKS",
    )
    document_router_max_candidate_docs: int = Field(
        default=250,
        ge=1,
        le=5000,
        validation_alias="DOCUMENT_ROUTER_MAX_CANDIDATE_DOCS",
    )

    # Langfuse (optional): SDK tracing + monitoring (pricing cache, optional HTTP ingestion)
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )
    # Hours between ``GET /api/public/models`` refreshes (local cost estimate in LLMUsageLog).
    langfuse_pricing_refresh_hours: float = Field(
        default=6.0,
        ge=0.25,
        le=168.0,
        validation_alias="LANGFUSE_PRICING_REFRESH_HOURS",
    )
    # Extra POST /api/public/ingestion mirror (usually off — SDK already traces). Avoid duplicate Langfuse rows.
    langfuse_manual_ingestion: bool = Field(
        default=False,
        validation_alias="LANGFUSE_MANUAL_INGESTION",
    )

    # Retrieval drift (KS on live confidence samples) — lower = faster demos; raise for production.
    drift_min_samples: int = Field(
        default=5,
        ge=3,
        le=50,
        validation_alias="DRIFT_MIN_SAMPLES",
    )
    drift_split_min_total: int = Field(
        default=10,
        ge=6,
        le=200,
        validation_alias="DRIFT_SPLIT_MIN_TOTAL",
    )

    @model_validator(mode="after")
    def _chunk_overlap_sane(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_words:
            object.__setattr__(
                self,
                "chunk_overlap_words",
                max(0, self.chunk_words // 6),
            )
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            object.__setattr__(
                self,
                "chunk_overlap_chars",
                max(0, self.chunk_size_chars // 5),
            )
        return self


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    from app.services.langfuse_env import apply_langfuse_env_from_settings

    apply_langfuse_env_from_settings(s)
    return s


def clear_settings_cache() -> None:
    get_settings.cache_clear()
    try:
        from app.services.openai_factory import clear_openai_client_cache

        clear_openai_client_cache()
    except ImportError:
        pass
