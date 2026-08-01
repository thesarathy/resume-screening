"""
Application factory for the Resume Screening & Candidate Ranking System.
"""

import csv
import io
import logging
from pathlib import Path

from flask import Flask, Response, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.config import CONFIG_MAP
from app.factory import METHOD_TFIDF, VALID_METHODS
from app.parser import ParsingError, ResumeParser
from app.ranking import ResumeRanker

logger = logging.getLogger(__name__)

# Holds the most recent ranking so /export.csv can re-emit it without
# forcing the recruiter to re-upload. A module-level store is fine for a
# dev tool; swap for a session/cache in a multi-user deployment.
_last_ranking: dict | None = None


def create_app(config_name: str = "default") -> Flask:
    app = Flask(__name__)

    config_class = CONFIG_MAP.get(config_name, CONFIG_MAP["default"])
    app.config.from_object(config_class)

    _configure_logging(app)
    _ensure_upload_folder_exists(app)

    from app.factory import ScorerFactory

    # One shared spaCy pipeline + scorer factory for the whole app.
    app.extensions["scorer_factory"] = ScorerFactory()

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

    @app.route("/")
    def index():
        return render_template("index.html", valid_methods=VALID_METHODS)

    @app.route("/rank", methods=["POST"])
    def rank():
        job_description = request.form.get("job_description", "").strip()
        if not job_description:
            flash("Please provide a job description.", "error")
            return redirect(url_for("index"))

        method = request.form.get("method", METHOD_TFIDF)
        if method not in VALID_METHODS:
            flash(f"Unknown ranking method: {method!r}", "error")
            return redirect(url_for("index"))

        files = request.files.getlist("resumes")
        if not files or all(f.filename == "" for f in files):
            flash("Please upload at least one resume.", "error")
            return redirect(url_for("index"))

        factory = app.extensions["scorer_factory"]
        parser = ResumeParser()

        # Save uploads to disk so the parser (which reads paths) can open
        # them, then parse each -- a file that fails to parse is flagged
        # and skipped rather than failing the whole batch.
        saved_paths: list[Path] = []
        resumes: list[tuple[str, str]] = []
        errors: list[str] = []

        try:
            for upload in files:
                filename = upload.filename or ""
                if not filename or not _allowed_file(filename, app):
                    errors.append(f"Skipped '{filename or '(unnamed)'}': unsupported file type.")
                    continue
                path = _save_upload(upload, app)
                saved_paths.append(path)
                try:
                    text = parser.parse(path)
                except ParsingError as exc:
                    errors.append(str(exc))
                    continue
                resumes.append((path.name, text))

            if not resumes:
                flash(
                    "No valid resumes could be processed: " + "; ".join(errors),
                    "error",
                )
                return redirect(url_for("index"))

            scorer = factory.build_scorer(method)
            ranker = ResumeRanker(nlp=factory.nlp, scorer=scorer)
            candidates = ranker.rank(job_description, resumes)

            # Stash for the CSV export route.
            global _last_ranking
            _last_ranking = {
                "job_description": job_description,
                "method": method,
                "candidates": [c.as_dict() for c in candidates],
            }

            return render_template(
                "results.html",
                job_description=job_description,
                method=method,
                candidates=candidates,
                errors=errors,
            )
        finally:
            for path in saved_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @app.route("/export.csv")
    def export_csv():
        if not _last_ranking:
            flash("Nothing to export yet -- run a ranking first.", "error")
            return redirect(url_for("index"))
        return _csv_response(_last_ranking)


def _allowed_file(filename: str, app: Flask) -> bool:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return extension in app.config["ALLOWED_EXTENSIONS"]


def _save_upload(upload, app: Flask) -> Path:
    """Save an uploaded file to the app's upload folder with a unique name."""
    from uuid import uuid4

    base = secure_filename(upload.filename)
    unique_name = f"{uuid4().hex}_{base}"
    path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
    upload.save(path)
    return path


def _csv_response(ranking: dict) -> Response:
    """Build a UTF-8 (with BOM, for Excel) CSV response from a ranking dict."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "filename", "score", "name", "email", "phone",
                     "education", "experience_years", "skills"])

    for rank, cand in enumerate(ranking["candidates"], start=1):
        writer.writerow([
            rank,
            cand.get("filename", ""),
            f"{cand.get('score', 0.0):.2f}",
            cand.get("name") or "",
            cand.get("email") or "",
            cand.get("phone") or "",
            "; ".join(cand.get("education", [])),
            cand.get("experience_years") or "",
            "; ".join(cand.get("skills", [])),
        ])

    csv_text = buffer.getvalue()
    return Response(
        "﻿" + csv_text,  # utf-8 BOM so Excel opens it correctly
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=ranking.csv"},
    )