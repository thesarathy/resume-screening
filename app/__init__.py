"""
Application factory for the Resume Screening & Candidate Ranking System.

Every ranking is persisted to the database, which powers the ranking
history and candidate detail features -- and keeps the app ready to host
(Render/Railway) where the database is PostgreSQL via DATABASE_URL.
"""

import csv
import io
import json
import logging
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from app.config import CONFIG_MAP
from app.factory import METHOD_TFIDF, VALID_METHODS
from app.models import Candidate, Ranking, db
from app.parser import ParsingError, ResumeParser
from app.ranking import ResumeRanker

logger = logging.getLogger(__name__)


def create_app(config_name: str = "default") -> Flask:
    app = Flask(__name__)

    config_class = CONFIG_MAP.get(config_name, CONFIG_MAP["default"])
    app.config.from_object(config_class)

    _configure_logging(app)
    _ensure_upload_folder_exists(app)
    _ensure_db_folder_exists(app)

    from app.factory import ScorerFactory

    # One shared spaCy pipeline + scorer factory for the whole app.
    app.extensions["scorer_factory"] = ScorerFactory()

    db.init_app(app)
    with app.app_context():
        db.create_all()

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
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)


def _ensure_db_folder_exists(app: Flask) -> None:
    """Make sure the SQLite file's parent directory exists for local dev.

    PostgreSQL (production) ignores this since its URI has no file path.
    """
    uri = app.config["SQLALCHEMY_DATABASE_URI"]
    if uri.startswith("sqlite:///"):
        db_path = uri.replace("sqlite:///", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


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

        saved_paths: list[Path] = []
        resumes: list[tuple[str, str]] = []
        errors: list[str] = []

        try:
            for upload in files:
                filename = upload.filename or ""
                if not filename or not _allowed_file(filename, app):
                    errors.append(
                        f"Skipped '{filename or '(unnamed)'}': unsupported file type."
                    )
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

            ranking_id = _persist_ranking(job_description, method, candidates)

            for error in errors:
                flash(error, "error")

            return redirect(url_for("ranking_detail", ranking_id=ranking_id))
        finally:
            for path in saved_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @app.route("/rankings")
    def rankings():
        all_rankings = Ranking.query.order_by(Ranking.created_at.desc()).all()
        return render_template("rankings.html", rankings=all_rankings)

    @app.route("/rankings/<int:ranking_id>")
    def ranking_detail(ranking_id: int):
        ranking = db.get_or_404(Ranking, ranking_id)
        candidates = [c.as_dict() for c in ranking.candidates]
        return render_template("results.html", ranking=ranking, candidates=candidates)

    @app.route("/rankings/<int:ranking_id>/candidates/<int:candidate_id>")
    def candidate_detail(ranking_id: int, candidate_id: int):
        candidate = db.session.get(Candidate, candidate_id)
        if candidate is None or candidate.ranking_id != ranking_id:
            abort(404)
        return render_template(
            "candidate_detail.html",
            ranking_id=ranking_id,
            candidate=candidate.as_dict(),
        )

    @app.route("/rankings/<int:ranking_id>/export.csv")
    def ranking_export(ranking_id: int):
        ranking = db.get_or_404(Ranking, ranking_id)
        return _csv_response(ranking)


def _persist_ranking(
    job_description: str, method: str, candidates: list
) -> int:
    """Save a ranking and its candidates, returning the new Ranking id."""
    ranking = Ranking(job_description=job_description, method=method)
    for rank, c in enumerate(candidates, start=1):
        ranking.candidates.append(
            Candidate(
                rank=rank,
                filename=c.filename,
                name=c.name,
                email=c.email,
                phone=c.phone,
                experience_years=c.experience_years,
                score=c.score,
                education=json.dumps(c.education),
                skills=json.dumps(c.skills),
                candidate_skills=json.dumps(c.candidate_skills),
                text_preview=c.text_preview,
            )
        )
    db.session.add(ranking)
    db.session.commit()
    return ranking.id


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


def _csv_response(ranking: Ranking) -> Response:
    """Build a UTF-8 (with BOM, for Excel) CSV response from a Ranking."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "filename", "score", "name", "email", "phone",
                     "education", "experience_years", "skills"])

    for candidate in ranking.candidates:
        c = candidate.as_dict()
        writer.writerow([
            c["rank"],
            c["filename"],
            f"{c['score']:.2f}",
            c["name"] or "",
            c["email"] or "",
            c["phone"] or "",
            "; ".join(c["education"]),
            c["experience_years"] or "",
            "; ".join(c["skills"]),
        ])

    csv_text = buffer.getvalue()
    return Response(
        "﻿" + csv_text,  # utf-8 BOM so Excel opens it correctly
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=ranking.csv"},
    )
