import os

from dotenv import load_dotenv

# Load variables from a local .env file (never committed).
load_dotenv()


class Config:
    """Application configuration sourced from environment variables."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-change-me")

    # LLM / Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/your_db"
    )

    # Caching
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 600  # 10 minutes

    # Uploads
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS = {"csv", "xlsx"}

    # Rendering
    MAX_TABLE_ROWS = 100  # above this, offer a CSV download instead of a table
