"""Tests for app/preprocessing.py."""

import pytest

from app.preprocessing import TextPreprocessor, clean_text


def test_clean_text_removes_bullets_and_collapses_whitespace():
    raw = "• Managed   a team\n•Built  APIs"

    result = clean_text(raw)

    assert "•" not in result
    assert "  " not in result


@pytest.fixture(scope="module")
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


def test_normalize_lowercases_and_lemmatizes(preprocessor):
    result = preprocessor.normalize("Managing Teams and Building APIs")

    assert "manage" in result
    assert "team" in result
    assert "managing" not in result


def test_normalize_removes_stopwords(preprocessor):
    result = preprocessor.normalize("I am the best candidate for this job")

    for stopword in ("i", "am", "the", "for", "this"):
        assert stopword not in result.split()


def test_normalize_strips_bullets_and_whitespace(preprocessor):
    result = preprocessor.normalize("• Python\n• SQL")

    assert "•" not in result
    assert "python" in result
    assert "sql" in result


def test_raises_for_unknown_model():
    with pytest.raises(RuntimeError, match="not installed"):
        TextPreprocessor(model_name="not_a_real_model")