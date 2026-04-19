"""Ensure OpenAI SDK is only imported via ``openai_factory`` (Langfuse-wrapped path)."""

from __future__ import annotations

import re
from pathlib import Path


def test_app_sources_do_not_import_openai_sdk_directly() -> None:
    app_root = Path(__file__).resolve().parent.parent / "app"
    allowed = app_root / "services" / "openai_factory.py"
    # Block direct `import openai` / `from openai import ...` (not `openai_usage`, etc.)
    pattern = re.compile(r"(^|\n)\s*(from openai import|import openai\s)", re.MULTILINE)
    offenders: list[str] = []
    for path in sorted(app_root.rglob("*.py")):
        if path == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(app_root)))
    assert not offenders, (
        "Import OpenAI only through app.services.openai_factory.get_async_openai_client; "
        f"found direct openai SDK imports in: {offenders}"
    )
