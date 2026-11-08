"""Settings, read from environment variables or a local .env file (never committed)."""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # AI provider (Google Gemini API, free tier)
    gemini_api_key: SecretStr | None = None
    chat_model: str = "gemini-3.5-flash-lite"
    fallback_chat_model: str | None = "gemini-3.8-flash"  # used when the main model is overloaded (503/429)
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768

    # Knowledge base and vector store
    knowledge_dir: Path = ROOT / "knowledge"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "gym_kb"
    top_k: int = 4
    min_similarity: float = 0.55  # measured: real matches score 0.67-0.77, unrelated 0.51-0.61

    # Agent: gym database (schedule, bookings, tickets) and conversation memory
    gym_db_path: Path = ROOT / "data" / "gym.db"
    memory_db_path: Path = ROOT / "data" / "memory.db"

    # API safety limits
    max_question_chars: int = 500
    rate_limit_per_minute: int = 20

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
