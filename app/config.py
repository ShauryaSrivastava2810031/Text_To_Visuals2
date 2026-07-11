import os
from typing import ClassVar

from dotenv import load_dotenv

# Load variables from a local .env file (never committed).
load_dotenv()


class Config:
    """Application configuration sourced from environment variables."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-change-me")

    # Logging verbosity for the "t2v" loggers: INFO (default) or DEBUG
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM — provider-agnostic (google | openai | anthropic | openrouter)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    # Per-provider API keys (only the active provider's key is required)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/your_db"
    )

    # Caching
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 600  # 10 minutes

    # Uploads
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {"csv", "xlsx"}

    # Rendering
    MAX_TABLE_ROWS = 100  # above this, offer a CSV download instead of a table
