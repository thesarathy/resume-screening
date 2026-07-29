"""
Application factory for the Resume Screening & Candidate Ranking System.
"""

import logging
from pathlib import Path

from flask import Flask

from app.config import CONFIG_MAP


def create_app(config_name: str = "default") -> Flask:
    app = Flask(__name__)

    config_class = CONFIG_MAP.get(config_name, CONFIG_MAP["default"])
    app.config.from_object(config_class)

    _configure_logging(app)
    _ensure_upload_folder_exists(app)
    _register_routes(app)

    app.logger.info("App created with config=%s", config_name)
    return app


def _configure_logging(app: Flask) -> None:
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    app.logger.setLevel(log_level)


def _ensure_upload_folder_exists(app: Flask) -> None:
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)


def _register_routes(app: Flask) -> None:
    @app.route("/health")
    def health() -> tuple[dict, int]:
        return {"status": "ok", "service": "resume-screening"}, 200