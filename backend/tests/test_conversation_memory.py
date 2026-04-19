"""Auto memory extraction (parsing + guards)."""

from app.services.conversation_memory import _parse_memories_json


def test_parse_memories_json_valid():
    assert _parse_memories_json('{"memories": ["Prefers Indian law", "Studying for CLAT"]}') == [
        "Prefers Indian law",
        "Studying for CLAT",
    ]


def test_parse_memories_json_empty():
    assert _parse_memories_json('{"memories": []}') == []
    assert _parse_memories_json("") == []
    assert _parse_memories_json("not json") == []


def test_parse_memories_json_caps_at_two():
    raw = '{"memories": ["a", "b", "c"]}'
    assert len(_parse_memories_json(raw)) == 2


def test_parse_memories_json_dedupes_case_insensitive_strings():
    raw = '{"memories": ["Same", "same"]}'
    out = _parse_memories_json(raw)
    assert len(out) == 1


def test_parse_memories_json_strips_markdown_fence():
    raw = '```json\n{"memories": ["From Ollama"]}\n```'
    assert _parse_memories_json(raw) == ["From Ollama"]
