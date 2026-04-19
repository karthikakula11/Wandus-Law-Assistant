"""
Route short conversational / meta questions to general chat so Wandus feels like a bot,
not a strict legal search over random retrieved chunks.
"""

import re

# If present, user is likely asking about law content — do not treat as small talk.
_LEGAL_HINT = re.compile(
    r"\b(section|article|clause|subsection|act|statute|ordinance|rule|order|"
    r"ipc|crpc|constitution|petition|plaintiff|defendant|jurisdiction|appeal|"
    r"penalty|fine|imprisonment|bail|offence|offense)\b",
    re.I,
)


def is_small_talk(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q or len(q) > 320:
        return False
    if _LEGAL_HINT.search(q):
        return False

    # Identity / meta (includes "who r u", "whats ur name", casual spellings)
    if re.search(
        r"\b(who\s+(are|r)\s+y?o?u|wh(o\'?s|os)\s+y?o?u|what\s+(are|r)\s+y?o?u|"
        r"what\s*('?s| is)\s+your\s+name|"
        r"(whats|what'?s)\s+ur\s+name|what\s+ur\s+name|"
        r"introduce\s+yourself|tell\s+me\s+about\s+yourself)\b",
        q,
        re.I,
    ):
        return True

    # Thanks / bye
    if re.match(r"^(thanks|thank you|thx|ty|bye|goodbye|see you|cya)\b", q, re.I):
        return True

    # Pure greeting line
    if re.match(
        r"^(hi|hello|hey|hiya|yo|sup|good\s+(morning|afternoon|evening))[\s!.,?]*$",
        q,
        re.I,
    ):
        return True

    # Short "how are you"
    if re.match(r"^how\s+(are|r)\s+y?o?u[\s?!.]*$", q, re.I):
        return True

    # Greeting + identity in one short message (e.g. "hi, who are you", "hi whats ur name")
    if len(q) <= 100 and re.match(r"^(hi|hello|hey|hiya)\b", q, re.I):
        if re.search(r"\b(who|what|whats)\b", q) and not _LEGAL_HINT.search(q):
            return True

    return False
