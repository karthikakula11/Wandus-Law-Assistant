import pytest

from app.services.small_talk import is_small_talk


@pytest.mark.parametrize(
    "q,expected",
    [
        ("hi who r u", True),
        ("Hi, who are you?", True),
        ("hey what is your name", True),
        ("hi whats ur name", True),
        ("hello", True),
        ("thanks", True),
        ("What is section 5 of the IPC?", False),
        ("Section 2 definitions", False),
        ("negligence under tort law", False),
    ],
)
def test_is_small_talk(q, expected):
    assert is_small_talk(q) is expected
