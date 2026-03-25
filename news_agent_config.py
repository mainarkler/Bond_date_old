from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    gnews_key: str = os.getenv("GNEWS_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    request_timeout_seconds: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))


settings = Settings()
