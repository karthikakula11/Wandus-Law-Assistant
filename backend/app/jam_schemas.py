"""
API models aligned with jamwithai/production-agentic-rag-course (Ask + hybrid search).
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering (course-compatible)."""

    query: str = Field(..., description="User's question", min_length=1, max_length=4000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=20)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector) when configured")
    model: str = Field(
        "gpt-4o-mini",
        description="OpenAI chat model (replaces Ollama in this deployment)",
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Reserved; course used arXiv categories — not used for law corpus",
    )


class AskResponse(BaseModel):
    """Response model for RAG question answering (course-compatible)."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="Source URIs or document references used")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")


class HybridSearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., min_length=1, max_length=2000)
    size: int = Field(10, ge=1, le=100)
    from_: int = Field(0, ge=0, alias="from", description="Offset for pagination")
    categories: Optional[List[str]] = Field(None, description="Unused in law deployment")
    latest_papers: bool = Field(False, description="Unused in law deployment")
    use_hybrid: bool = Field(True, description="Hybrid (BM25 + vector) when configured")
    min_score: float = Field(0.0, ge=0.0)


class SearchHit(BaseModel):
    arxiv_id: str = ""
    title: str = ""
    authors: Optional[str] = None
    abstract: Optional[str] = None
    published_date: Optional[str] = None
    pdf_url: Optional[str] = None
    score: float = 0.0
    highlights: Optional[dict[str, Any]] = None
    chunk_text: Optional[str] = None
    chunk_id: Optional[str] = None
    section_name: Optional[str] = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str
    total: int
    hits: List[SearchHit]
    size: int = Field(description="Number of results requested")
    from_: int = Field(alias="from", description="Offset used for pagination")
    search_mode: Optional[str] = Field(None, description="bm25, vector, or hybrid")
    error: Optional[str] = None
