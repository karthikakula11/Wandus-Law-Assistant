"""Sync Langfuse credentials from Settings into ``os.environ`` for the Langfuse SDK.

The Langfuse OpenAI wrapper calls ``get_client()`` → ``Langfuse()``, which reads
``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_BASE_URL`` from the
process environment. Pydantic Settings loads ``.env`` into the model only; it does not
set those variables on ``os.environ``, so tracing could be disabled while local usage
logging (via ``get_settings()``) still worked.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def apply_langfuse_env_from_settings(settings: Any) -> None:
    pk = (getattr(settings, "langfuse_public_key", None) or "").strip()
    sk = (getattr(settings, "langfuse_secret_key", None) or "").strip()
    if not pk or not sk:
        return

    host = (getattr(settings, "langfuse_host", None) or "").strip().rstrip("/")
    if not host:
        host = "https://cloud.langfuse.com"

    os.environ["LANGFUSE_PUBLIC_KEY"] = pk
    os.environ["LANGFUSE_SECRET_KEY"] = sk
    os.environ["LANGFUSE_BASE_URL"] = host
    os.environ.setdefault("LANGFUSE_HOST", host)
    # Default SDK batching waits up to 5s; lower backup interval if a code path omits flush().
    os.environ.setdefault("LANGFUSE_FLUSH_INTERVAL", "2")

    logger.debug(
        "Langfuse env synced from settings (base_url=%s, public_key_prefix=%s…)",
        host,
        pk[:8],
    )
