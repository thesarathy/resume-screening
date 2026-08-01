"""Tests for app/similarity.py -- SentenceTransformerSimilarityScorer.

Uses a lightweight fake embedding model (word-presence vectors) instead
of the real ~90MB pretrained model. This tests the scorer's own logic --
cosine similarity, score ordering, edge cases -- deterministically and
without a network call on every test run. The real model is loaded the
same way in the running app (see load_sentence_transformer_model in
app/similarity.py); that integration is worth a one-off manual check
(see the note at the end of this step), not something every automated
test run should depend on downloading.
"""

import numpy as np
import pytest

from app.similarity import SentenceTransformerSimilarityScorer


class FakeEmbeddingModel:
    """Encodes text as a word-presence vector over a shared vocabulary.

    Not semantically meaningful the way a real model is -- it exists
    purely so the scorer's surrounding logic can be tested without the
    real dependency or a network call.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        vocab = sorted({word for text in texts for word in text.lower().split()})
        vectors = [
            [1.0 if word in set(text.lower().split()) else 0.0 for word in vocab]
            for text in texts
        ]
        return np.array(vectors)


@pytest.fixture
def scorer() -> SentenceTransformerSimilarityScorer:
    return SentenceTransformerSimilarityScorer(FakeEmbeddingModel())


def test_returns_one_score_per_resume(scorer):
    jd = "python developer docker experience"
    resumes = ["python developer docker skilled", "chef baking experience"]

    scores = scorer.score(jd, resumes)

    assert len(scores) == 2


def test_scores_are_within_valid_range(scorer):
    jd = "python developer docker experience"
    resumes = ["python developer docker skilled", "chef baking experience"]

    scores = scorer.score(jd, resumes)

    for score in scores:
        assert 0.0 <= score <= 1.0


def test_relevant_resume_scores_higher_than_irrelevant(scorer):
    jd = "backend engineer python docker postgresql"
    relevant_resume = "backend engineer python docker postgresql experience"
    irrelevant_resume = "professional chef pastry baking techniques"

    scores = scorer.score(jd, [relevant_resume, irrelevant_resume])

    assert scores[0] > scores[1]


def test_identical_text_scores_highest(scorer):
    jd = "senior python developer docker kubernetes experience"

    scores = scorer.score(jd, [jd, "completely unrelated content gardening"])

    assert scores[0] > scores[1]
    assert scores[0] > 0.9


def test_empty_resume_list_returns_empty_list(scorer):
    scores = scorer.score("some job description", [])
    assert scores == []