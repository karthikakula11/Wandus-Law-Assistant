"""Chat sidebar persistence API (Pydantic schema)."""

import pytest

from app.schemas import ChatThreadsStateIn


def test_chat_threads_state_schema_active_must_match():
    with pytest.raises(Exception):
        ChatThreadsStateIn.model_validate(
            {
                "v": 2,
                "threads": [
                    {
                        "id": "th1",
                        "title": "T",
                        "updatedAt": "2026-01-01T00:00:00.000Z",
                        "messages": [{"id": "m1", "role": "user", "content": "x"}],
                    }
                ],
                "activeThreadId": "other",
            }
        )


def test_chat_threads_state_schema_ok():
    st = ChatThreadsStateIn.model_validate(
        {
            "v": 2,
            "threads": [
                {
                    "id": "th1",
                    "title": "T",
                    "updatedAt": "2026-01-01T00:00:00.000Z",
                    "messages": [{"id": "m1", "role": "user", "content": "hello"}],
                }
            ],
            "activeThreadId": "th1",
        }
    )
    assert st.v == 2
    assert st.threads[0].id == "th1"
