"""Tests for app/skill_extractor.py."""

import pytest

from app.preprocessing import load_spacy_model
from app.skill_extractor import SkillExtractor


@pytest.fixture(scope="module")
def nlp():
    # Loaded once per test module -- spaCy model loading is expensive.
    return load_spacy_model()


@pytest.fixture
def extractor(nlp):
    return SkillExtractor(nlp)


def test_extracts_basic_skills(extractor):
    result = extractor.extract("Experienced in Python, Django, and Docker.")
    assert set(result.skills) == {"Python", "Django", "Docker"}


def test_go_language_is_detected(extractor):
    result = extractor.extract("Backend services written in Go and deployed on AWS.")
    assert "Go" in result.skills


def test_go_verb_is_not_detected(extractor):
    result = extractor.extract("Go above and beyond to meet client deadlines.")
    assert "Go" not in result.skills


def test_lowercase_go_is_not_matched(extractor):
    result = extractor.extract("We need to go over the requirements again.")
    assert "Go" not in result.skills


def test_r_language_detected_with_exact_case(extractor):
    result = extractor.extract("Statistical modeling using R and RStudio.")
    assert "R" in result.skills


def test_lowercase_r_not_matched(extractor):
    result = extractor.extract("Our r&d team files new patents every year.")
    assert "R" not in result.skills


def test_aliases_resolve_to_canonical(extractor):
    result = extractor.extract("5 years in ML and NLP, comfortable with JS and Golang.")
    assert set(result.skills) == {
        "Machine Learning", "Natural Language Processing", "JavaScript", "Go"
    }


def test_no_duplicates_from_alias_and_canonical(extractor):
    result = extractor.extract("Skilled in Go and Golang microservices.")
    # "Go" (exact-case match) and "Golang" (alias match) both resolve to
    # the same canonical skill -- it should appear exactly once, not twice.
    assert result.skills.count("Go") == 1
    assert "Go" in result.skills


def test_empty_text_returns_no_skills(extractor):
    result = extractor.extract("")
    assert result.skills == []


def test_by_category_grouping(extractor):
    result = extractor.extract("Python, Docker, and AWS experience.")
    grouped = result.by_category()
    assert grouped["language"] == ["Python"]
    assert grouped["tool"] == ["Docker"]
    assert grouped["cloud"] == ["AWS"]


def test_candidate_skills_finds_undocumented_acronym(extractor):
    result = extractor.extract_candidate_skills("Built ETL pipelines for reporting")
    assert "ETL" in result


def test_candidate_skills_merges_adjacent_technical_tokens(extractor):
    result = extractor.extract_candidate_skills("Familiar with ABC DEF systems")
    assert "ABC DEF" in result


def test_candidate_skills_excludes_known_vocabulary(extractor):
    result = extractor.extract_candidate_skills("Experience with Python and AWS")
    assert "Python" not in result
    assert "AWS" not in result