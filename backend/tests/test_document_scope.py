"""Document scope resolution for RAG (no DB)."""

from uuid import uuid4

from app.services.document_scope import _find_doc_in_text


def test_find_doc_prefers_longest_title_match():
    a = uuid4()
    b = uuid4()
    docs = [
        (a, "Act"),
        (b, "law_test_document.pdf"),
    ]
    text = "what is in law_test_document.pdf please"
    assert _find_doc_in_text(text, docs) == b


def test_find_doc_sample_act():
    a = uuid4()
    docs = [(a, "Sample Act")]
    assert _find_doc_in_text("Summarize Sample Act section 1", docs) == a
