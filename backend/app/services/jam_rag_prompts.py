"""
Jam with AI course–compatible RAG prompt builder.

**Reference (cloned repo)**:
``reference/production-agentic-rag-course/src/services/ollama/prompts.py`` — class ``RAGPromptBuilder``,
method ``create_rag_prompt`` (system file ``prompts/rag_system.txt``, ``### Context from Papers``,
numbered ``[i. arXiv:…]``, ``### Question``, ``### Answer``).

**This app**: ``app/prompts/rag_system.txt`` + same section shape; citations use document title when
no arXiv id (law corpus). Generation uses OpenAI instead of Ollama — see ``jam_ask_service.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class JamRAGPromptBuilder:
    """Builder class for creating RAG prompts (same shape as course Ollama path)."""

    def __init__(self) -> None:
        self.prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_file = self.prompts_dir / "rag_system.txt"
        if not prompt_file.exists():
            return (
                "You are an AI assistant specialized in answering questions about "
                "the user's indexed materials. Base your answer STRICTLY on the provided excerpts."
            )
        return prompt_file.read_text(encoding="utf-8").strip()

    def create_rag_prompt(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """Create a single prompt string (course: passed to Ollama /api/generate)."""
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")

            # Course format: [i. arXiv:…] — for law corpus we still use arxiv_id key if set;
            # otherwise show document title so the label is never empty.
            if arxiv_id:
                prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            else:
                title = chunk.get("document_title", "document")
                prompt += f"[{i}. {title}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += (
            "### Answer:\n"
            "Provide a natural, conversational response (not JSON) and cite sources using "
            "[1], [2] or [arXiv:id] format when applicable.\n\n"
        )

        return prompt
