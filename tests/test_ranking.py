"""Unit tests for the ResumeRanker orchestration layer.

Uses the real extractors (they need spaCy, which the suite already loads)
but a fake scorer, so we test ranking logic in isolation from TF-IDF /
SentenceTransformer behavior (which has its own tests).
"""

import pytest

from app.ranking import ResumeRanker
from app.preprocessing import load_spacy_model

RESUME_ALICE = (
    "Alice Johnson\n"
    "alice@example.com | 555-123-4567\n"
    "Software Engineer with 5 years of experience in Python and SQL.\n"
    "Bachelor of Science in Computer Science.\n"
)

RESUME_BOB = (
    "Bob Smith\n"
    "bob@example.com\n"
    "Data Analyst, 2 years of experience. Skilled in Python, ETL and pandas.\n"
    "Master of Science in Data Science.\n"
)


class FakeScorer:
    """Returns a fixed score per resume based on its index."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score(self, job_description: str, resumes: list[str]) -> list[float]:
        # Pad/truncate to match the number of resumes given.
        return [self.scores[i] for i in range(len(resumes))]


@pytest.fixture(scope="module")
def nlp():
    return load_spacy_model()


def test_rank_sorts_candidates_by_score_descending(nlp):
    scorer = FakeScorer([0.3, 0.9])
    ranker = ResumeRanker(nlp=nlp, scorer=scorer)
    resumes = [("alice.txt", RESUME_ALICE), ("bob.txt", RESUME_BOB)]

    result = ranker.rank("Software Engineer JD", resumes)

    assert [c.filename for c in result] == ["bob.txt", "alice.txt"]
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.3)


def test_rank_populates_candidate_fields(nlp):
    scorer = FakeScorer([0.8])
    ranker = ResumeRanker(nlp=nlp, scorer=scorer)

    result = ranker.rank("JD", [("alice.txt", RESUME_ALICE)])
    candidate = result[0]

    assert candidate.name == "Alice Johnson"
    assert candidate.email == "alice@example.com"
    assert candidate.phone == "555-123-4567"
    assert candidate.experience_years == 5.0
    assert "Python" in candidate.skills
    assert any("Bachelor" in edu for edu in candidate.education)
    assert candidate.match_pct == 80


def test_rank_returns_empty_list_for_no_resumes(nlp):
    ranker = ResumeRanker(nlp=nlp, scorer=FakeScorer([]))
    assert ranker.rank("JD", []) == []


def test_rank_preserves_all_candidates_even_when_scores_tie(nlp):
    scorer = FakeScorer([0.5, 0.5])
    ranker = ResumeRanker(nlp=nlp, scorer=scorer)

    result = ranker.rank("JD", [("a.txt", RESUME_ALICE), ("b.txt", RESUME_BOB)])

    # Both present; a stable sort keeps original order for ties.
    assert {c.filename for c in result} == {"a.txt", "b.txt"}
    assert result[0].score == result[1].score == pytest.approx(0.5)
