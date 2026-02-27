from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "analytics-clean-arch"
    app_env: str = "local"
    log_level: str = "INFO"

    moex_base_url: str = "https://iss.moex.com/iss"
    cbr_base_url: str = "https://www.cbr.ru"

    http_timeout_seconds: int = 20
    http_retry_total: int = 5
    http_retry_backoff: float = 0.7


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
