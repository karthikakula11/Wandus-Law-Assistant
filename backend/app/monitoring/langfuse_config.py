"""
Credentials and enablement (Basic auth for Langfuse HTTP APIs).

``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST`` / ``LANGFUSE_BASE_URL``
must all be non-empty for ``langfuse_enabled()`` to be true.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str
    secret_key: str
    base_url: str  # no trailing slash

    def basic_auth_header(self) -> str:
        raw = f"{self.public_key}:{self.secret_key}"
        return base64.b64encode(raw.encode("utf-8")).decode("ascii")


def get_langfuse_config() -> LangfuseConfig | None:
    s = get_settings()
    pk = (s.langfuse_public_key or "").strip()
    sk = (s.langfuse_secret_key or "").strip()
    if not pk or not sk:
        return None
    host = (s.langfuse_host or "").strip().rstrip("/") or "https://cloud.langfuse.com"
    return LangfuseConfig(public_key=pk, secret_key=sk, base_url=host)


def langfuse_enabled() -> bool:
    return get_langfuse_config() is not None
