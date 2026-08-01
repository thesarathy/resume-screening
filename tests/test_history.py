"""Tests for ranking persistence (history) and the database models."""

import pytest

from app import create_app
from app.models import Candidate, Ranking, db


@pytest.fixture(scope="module")
def app():
    app = create_app("testing")
    with app.app_context():
        db.drop_all()
        db.create_all()
        _seed(app)
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


def _seed(app) -> None:
    ranking = Ranking(job_description="Need a Python backend engineer", method="tfidf")
    ranking.candidates.append(
        Candidate(
            rank=1,
            filename="alice.txt",
            name="Alice Johnson",
            email="alice@example.com",
            score=0.8,
            skills='["Python", "SQL"]',
            education='["Bachelor of Science"]',
            text_preview="Alice Johnson\nPython engineer...",
        )
    )
    ranking.candidates.append(
        Candidate(rank=2, filename="bob.txt", name="Bob Smith", score=0.4, skills="[]")
    )
    db.session.add(ranking)
    db.session.commit()


@pytest.fixture()
def client(app):
    with app.test_client() as test_client:
        yield test_client


def test_ranking_persists_with_candidates(app):
    with app.app_context():
        ranking = Ranking.query.one()
        assert ranking.method == "tfidf"
        assert len(ranking.candidates) == 2
        assert ranking.candidates[0].rank == 1
        assert ranking.candidates[0].name == "Alice Johnson"


def test_candidate_as_dict_decodes_json_lists(app):
    with app.app_context():
        candidate = Candidate.query.filter_by(name="Alice Johnson").one()
        data = candidate.as_dict()
        assert data["skills"] == ["Python", "SQL"]
        assert data["education"] == ["Bachelor of Science"]
        assert data["match_pct"] == 80


def test_rankings_history_page_lists_ranking(client):
    response = client.get("/rankings")
    assert response.status_code == 200
    assert b"Python backend engineer" in response.data
    assert b"Alice Johnson" not in response.data  # results, not names, on history


def test_ranking_detail_renders_candidates(client):
    with client.application.app_context():
        ranking_id = Ranking.query.one().id
    response = client.get(f"/rankings/{ranking_id}")
    assert response.status_code == 200
    assert b"Alice Johnson" in response.data
    assert b"Download CSV" in response.data


def test_ranking_export_csv_matches_persisted_candidates(client):
    with client.application.app_context():
        ranking_id = Ranking.query.one().id
    response = client.get(f"/rankings/{ranking_id}/export.csv")
    assert response.status_code == 200
    body = response.data.decode("utf-8-sig")
    assert "Alice Johnson" in body
    assert "Bob Smith" in body
    # Sorted by rank: alice (1) before bob (2).
    assert body.index("Alice Johnson") < body.index("Bob Smith")
