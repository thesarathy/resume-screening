"""Tests for app/entity_extractor.py."""

import pytest

from app.entity_extractor import EntityExtractor
from app.preprocessing import load_spacy_model


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    # Loaded once per test module and shared across all tests in this
    # file -- spaCy model loading is expensive, so we pay that cost once.
    nlp = load_spacy_model()
    return EntityExtractor(nlp)


def test_extracts_email(extractor):
    result = extractor.extract("Contact me at jane.doe@example.com for details")
    assert result.email == "jane.doe@example.com"


def test_returns_none_when_no_email(extractor):
    result = extractor.extract("No contact info here")
    assert result.email is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Call me at 555-123-4567", "555-123-4567"),
        ("Phone: (555) 123-4567", "(555) 123-4567"),
        ("+1 555.123.4567 is my number", "+1 555.123.4567"),
    ],
)
def test_extracts_phone(extractor, text, expected):
    result = extractor.extract(text)
    assert result.phone == expected


def test_returns_none_when_no_phone(extractor):
    result = extractor.extract("No phone number in this text")
    assert result.phone is None


@pytest.mark.parametrize(
    "header",
    [
        "Jane Doe\nSoftware Engineer\njane@example.com",
        "JANE DOE\nSoftware Engineer\njane@example.com",
        "ALEX RIVERA - Data Scientist\nalex@example.com",
        "Priya Sharma | Backend Developer\npriya@example.com",
        "Curriculum Vitae\nMichael Chen\nmichael@example.com",
    ],
)
def test_extracts_name_from_varied_headers(extractor, header):
    result = extractor.extract(header)
    assert result.name is not None
    # The extracted name should be one of the real name tokens, not a
    # job title or the "Curriculum Vitae" label.
    assert "engineer" not in result.name.lower()
    assert "curriculum" not in result.name.lower()


def test_extracts_education(extractor):
    result = extractor.extract("Bachelor of Science in Computer Science, MIT, 2020")
    assert result.education
    assert "Bachelor of Science" in result.education[0]


def test_extracts_multiple_education_entries_without_duplicates(extractor):
    text = "B.Tech in Computer Science\nMBA from Wharton\nB.Tech in Computer Science"
    result = extractor.extract(text)
    assert len(result.education) == 2


def test_extracts_experience_years(extractor):
    result = extractor.extract("5+ years of experience in backend development")
    assert result.experience_years == 5.0


def test_returns_none_when_no_experience_mentioned(extractor):
    result = extractor.extract("A short resume with no experience statement")
    assert result.experience_years is None