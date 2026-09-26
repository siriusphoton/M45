from typing import Literal

from pydantic import AnyHttpUrl, PositiveInt, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: PostgresDsn
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    agent_server_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:2024")
    agent_server_assistant_id: str = "m45"

    discord_bot_token: SecretStr | None = None
    discord_allowed_user_id: PositiveInt | None = None
    model_provider: Literal["google_genai", "ollama"] = "google_genai"
    model_name_google: str = "gemma-4-31b-it"
    model_name_ollama: str = "gemma4:31b-cloud"

    google_api_key: SecretStr | None = None
    ollama_api_key: SecretStr | None = None
    ollama_base_url: str = "https://ollama.com"

    personal_context_message_limit: PositiveInt = 20
    personal_context_character_limit: PositiveInt = 12_000


def load_settings() -> Settings:
    """Load and validate settings from environment sources."""
    return Settings()  # pyright: ignore[reportCallIssue]
