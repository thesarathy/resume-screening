"""Integration tests for the ranking web routes (dashboard, upload, CSV).

These exercise the real Flask app + TF-IDF path (fast, no model download)
end-to-end. Uploaded files are written under the testing config's
tests/tmp_uploads folder and cleaned up by the route handler.
"""

import io

import pytest

from app import create_app

RESUME_TXT = (
    "Alice Johnson\n"
    "alice@example.com | 555-123-4567\n"
    "Software Engineer with 5 years of experience in Python and SQL.\n"
    "Bachelor of Science in Computer Science.\n"
)

JD = "Software Engineer with experience in Python, SQL and cloud."  # noqa: E501


@pytest.fixture(scope="module")
def client():
    app = create_app("testing")
    with app.test_client() as test_client:
        yield test_client


def _post_rank(client, resumes, jd=JD, method="tfidf"):
    data = {"job_description": jd, "method": method}
    for name, content in resumes:
        if isinstance(content, str):
            content = content.encode()
        data["resumes"] = (io.BytesIO(content), name)
    return client.post("/rank", data=data, content_type="multipart/form-data")


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Rank candidates" in response.data


def test_rank_returns_results_page(client):
    response = _post_rank(client, [("alice.txt", RESUME_TXT)])
    assert response.status_code == 200
    assert b"Ranked candidates" in response.data
    assert b"alice@example.com" in response.data


def test_rank_requires_job_description(client):
    response = client.post(
        "/rank",
        data={"method": "tfidf", "resumes0": (io.BytesIO(b"x"), "a.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    assert "/" in response.headers["Location"]


def test_rank_rejects_bad_extension(client):
    response = _post_rank(client, [("evil.exe", b"not a resume")])
    # No valid resumes -> redirect back to the form with a flash error.
    assert response.status_code == 302
    assert "/" in response.headers["Location"]


def test_rank_requires_at_least_one_resume(client):
    response = client.post(
        "/rank",
        data={"job_description": JD, "method": "tfidf"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    assert "/" in response.headers["Location"]


def test_export_csv_after_ranking(client):
    _post_rank(client, [("alice.txt", RESUME_TXT)])
    response = client.get("/export.csv")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert "attachment" in response.headers.get("Content-Disposition", "")
    body = response.data.decode("utf-8-sig")
    rows = [r for r in body.splitlines() if r.strip()]
    assert rows[0].startswith("rank,filename,score")
    assert any("alice@example.com" in r for r in rows)


def test_export_csv_without_ranking_redirects():
    # The last-ranking store is a module global; reset it so this test is
    # independent of test order.
    import app as app_module
    app_module._last_ranking = None

    fresh = create_app("testing").test_client()
    response = fresh.get("/export.csv")
    assert response.status_code == 302
    assert "/" in response.headers["Location"]
