"""Tests for app/similarity.py -- TfidfSimilarityScorer."""

import pytest

from app.preprocessing import TextPreprocessor, load_spacy_model
from app.similarity import TfidfSimilarityScorer


@pytest.fixture(scope="module")
def scorer() -> TfidfSimilarityScorer:
    # Loaded once per test module -- spaCy model loading is expensive.
    nlp = load_spacy_model()
    preprocessor = TextPreprocessor(nlp=nlp)
    return TfidfSimilarityScorer(preprocessor)


def test_returns_one_score_per_resume(scorer):
    jd = "Looking for a Python developer with Docker experience"
    resumes = ["Python developer skilled in Docker", "Chef with baking experience"]

    scores = scorer.score(jd, resumes)

    assert len(scores) == 2


def test_scores_are_within_valid_range(scorer):
    jd = "Looking for a Python developer with Docker experience"
    resumes = ["Python developer skilled in Docker", "Chef with baking experience"]

    scores = scorer.score(jd, resumes)

    for score in scores:
        assert 0.0 <= score <= 1.0


def test_relevant_resume_scores_higher_than_irrelevant(scorer):
    jd = "Looking for a backend engineer experienced in Python, Docker, and PostgreSQL"
    relevant_resume = "Backend engineer with 5 years in Python, Docker, and PostgreSQL"
    irrelevant_resume = "Professional chef specializing in pastry and baking techniques"

    scores = scorer.score(jd, [relevant_resume, irrelevant_resume])

    assert scores[0] > scores[1]


def test_identical_text_scores_highest(scorer):
    jd = "Senior Python developer with Docker and Kubernetes experience"

    scores = scorer.score(jd, [jd, "Completely unrelated content about gardening"])

    assert scores[0] > scores[1]
    assert scores[0] > 0.9


def test_empty_resume_list_returns_empty_list(scorer):
    scores = scorer.score("Some job description", [])
    assert scores == []


def test_all_empty_documents_return_zero_without_crashing(scorer):
    scores = scorer.score("", ["", ""])
    assert scores == [0.0, 0.0]