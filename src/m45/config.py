from typing import Literal

from pydantic import PositiveInt, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: PostgresDsn
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    personal_context_message_limit: PositiveInt = 20
    personal_context_character_limit: PositiveInt = 12_000


def load_settings() -> Settings:
    """Load and validate settings from environment sources."""
    return Settings()  # pyright: ignore[reportCallIssue]
