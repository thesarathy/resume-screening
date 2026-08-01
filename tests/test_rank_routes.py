"""Integration tests for the ranking web routes (dashboard, upload, history,
candidate detail, CSV) backed by the test SQLite database."""

import io

import pytest

from app import create_app
from app.models import db

RESUME_TXT = (
    "Alice Johnson\n"
    "alice@example.com | 555-123-4567\n"
    "Software Engineer with 5 years of experience in Python and SQL.\n"
    "Bachelor of Science in Computer Science.\n"
)

JD = "Software Engineer with experience in Python, SQL and cloud."  # noqa: E501


@pytest.fixture(scope="module")
def app():
    app = create_app("testing")
    with app.app_context():
        db.drop_all()
        db.create_all()
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    with app.test_client() as test_client:
        yield test_client


def _post_rank(client, resumes, jd=JD, method="tfidf"):
    data = {"job_description": jd, "method": method}
    for name, content in resumes:
        if isinstance(content, str):
            content = content.encode()
        data["resumes"] = (io.BytesIO(content), name)
    return client.post("/rank", data=data, content_type="multipart/form-data")


def _ranking_id(response) -> int:
    """Parse the ranking id out of a /rank redirect Location."""
    location = response.headers["Location"]
    return int(location.rstrip("/").rsplit("/", 1)[-1])


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Rank candidates" in response.data


def test_rank_redirects_to_ranking_and_shows_results(client):
    response = _post_rank(client, [("alice.txt", RESUME_TXT)])
    assert response.status_code == 302
    assert "/rankings/" in response.headers["Location"]

    results = client.get(response.headers["Location"])
    assert results.status_code == 200
    assert b"Ranked candidates" in results.data
    assert b"alice@example.com" in results.data


def test_rank_requires_job_description(client):
    response = client.post(
        "/rank",
        data={"method": "tfidf", "resumes": (io.BytesIO(b"x"), "a.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    assert "/" == response.headers["Location"]


def test_rank_rejects_bad_extension(client):
    response = _post_rank(client, [("evil.exe", b"not a resume")])
    assert response.status_code == 302
    assert "/" == response.headers["Location"]


def test_rank_requires_at_least_one_resume(client):
    response = client.post(
        "/rank",
        data={"job_description": JD, "method": "tfidf"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    assert "/" == response.headers["Location"]


def test_rankings_history_lists_saved_ranking(client):
    created = _post_rank(client, [("alice.txt", RESUME_TXT)])
    ranking_id = _ranking_id(created)

    response = client.get("/rankings")
    assert response.status_code == 200
    assert b"Ranking history" in response.data
    assert f"rankings/{ranking_id}".encode() in response.data


def test_ranking_detail_404_for_unknown_id(client):
    assert client.get("/rankings/999999").status_code == 404


def test_candidate_detail_shows_resume_preview(client):
    import re

    created = _post_rank(client, [("alice.txt", RESUME_TXT)])
    ranking_id = _ranking_id(created)

    results = client.get(f"/rankings/{ranking_id}")
    assert results.status_code == 200

    # Pull the first candidate id out of a detail link on the results page.
    match = re.search(
        rf"/rankings/{ranking_id}/candidates/(\d+)",
        results.data.decode(),
    )
    assert match is not None
    candidate_id = int(match.group(1))

    detail = client.get(f"/rankings/{ranking_id}/candidates/{candidate_id}")
    assert detail.status_code == 200
    assert b"Alice Johnson" in detail.data
    assert b"Resume text" in detail.data


def test_export_csv_after_ranking(client):
    created = _post_rank(client, [("alice.txt", RESUME_TXT)])
    ranking_id = _ranking_id(created)

    response = client.get(f"/rankings/{ranking_id}/export.csv")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert "attachment" in response.headers.get("Content-Disposition", "")
    body = response.data.decode("utf-8-sig")
    rows = [r for r in body.splitlines() if r.strip()]
    assert rows[0].startswith("rank,filename,score")
    assert any("alice@example.com" in r for r in rows)
