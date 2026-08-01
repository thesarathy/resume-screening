"""
Database models for persisted rankings and their candidates.

Every time a recruiter ranks a batch of resumes, we store one Ranking row
and one Candidate row per resume. That's what powers the two new features:

- **Ranking history**: re-open and compare past rankings (`GET /rankings`).
- **Candidate detail view**: a page showing a single candidate's full facts
  plus a preview of their resume text, saved here at ranking time.

List-valued fields (education, skills, candidate_skills) are stored as
JSON strings so the schema stays portable across SQLite (dev) and
PostgreSQL (prod) without needing an array column type.
"""

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ranking(db.Model):
    """One screening run: a job description scored against a batch."""

    __tablename__ = "rankings"

    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    method = db.Column(db.String(32), nullable=False, default="tfidf")
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    candidates = db.relationship(
        "Candidate",
        back_populates="ranking",
        cascade="all, delete-orphan",
        order_by="Candidate.rank",
    )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "job_description": self.job_description,
            "method": self.method,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "candidate_count": len(self.candidates),
        }


class Candidate(db.Model):
    """One screened resume within a ranking."""

    __tablename__ = "candidates"

    id = db.Column(db.Integer, primary_key=True)
    ranking_id = db.Column(
        db.Integer, db.ForeignKey("rankings.id"), nullable=False, index=True
    )
    rank = db.Column(db.Integer, nullable=False, default=0)
    filename = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(255), nullable=True)
    phone = db.Column(db.String(64), nullable=True)
    experience_years = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, nullable=False, default=0.0)
    education = db.Column(db.Text, nullable=False, default="[]")
    skills = db.Column(db.Text, nullable=False, default="[]")
    candidate_skills = db.Column(db.Text, nullable=False, default="[]")
    text_preview = db.Column(db.Text, nullable=False, default="")

    ranking = db.relationship("Ranking", back_populates="candidates")

    @property
    def match_pct(self) -> int:
        """The similarity score as an integer percentage for display."""
        return round(self.score * 100)

    @staticmethod
    def _load_list(raw: str) -> list[str]:
        try:
            value = json.loads(raw or "[]")
            return value if isinstance(value, list) else []
        except (TypeError, ValueError):
            return []

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "rank": self.rank,
            "filename": self.filename,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "experience_years": self.experience_years,
            "score": self.score,
            "match_pct": round(self.score * 100),
            "education": self._load_list(self.education),
            "skills": self._load_list(self.skills),
            "candidate_skills": self._load_list(self.candidate_skills),
            "text_preview": self.text_preview,
        }
