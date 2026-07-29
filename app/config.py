"""
Application configuration.

Uses environment-based configuration classes so the same codebase behaves
correctly in development, testing, and production without code changes --
only the config *name* passed to create_app() changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class BaseConfig:
    """Settings shared by every environment."""

    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

    UPLOAD_FOLDER: Path = BASE_DIR / os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH: int = (
        int(os.environ.get("MAX_CONTENT_LENGTH_MB", 16)) * 1024 * 1024
    )
    ALLOWED_EXTENSIONS: set[str] = {
        ext.strip().lower()
        for ext in os.environ.get("ALLOWED_EXTENSIONS", "pdf,docx,txt").split(",")
    }

    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    DEBUG: bool = False
    TESTING: bool = False


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER: Path = BASE_DIR / "tests" / "tmp_uploads"


class ProductionConfig(BaseConfig):
    DEBUG = False
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")


CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}