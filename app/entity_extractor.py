"""
Entity extraction for resumes: name, email, phone, education, and a rough
years-of-experience estimate.

Skill extraction lives in its own module (skill_extractor.py, next step)
since it uses a different technique (a curated skill vocabulary + NER)
and is conceptually a separate concern from "who is this person and
what's their background."

Design note on name detection: it combines spaCy's PERSON NER with a
regex fallback, and deliberately only looks at the first few lines of
the resume rather than the whole document. Two reasons:

1. A resume mentions plenty of PERSON-shaped entities that aren't the
   candidate -- references, former managers, professional contacts.
   The candidate's own name is reliably one of the first things on the
   page, so narrowing the search space avoids most false positives
   outright, rather than trying to make the NER model smarter than it is.

2. spaCy's small English model (en_core_web_sm) is trained on ordinary
   grammatical sentences, and resume headers are anything but -- short
   fragments, ALL CAPS, separated by pipes/dashes instead of punctuation.
   In testing, that model correctly tags "Jane Doe" and "JANE DOE" as
   PERSON, but misses names like "ALEX RIVERA" when they're immediately
   followed by a job title. So NER is tried first (it's the more
   accurate signal when it works), and a shape-based regex heuristic
   (2-4 capitalized words, no digits, not a known job-title/section
   keyword) is used only as a fallback when NER finds nothing. Neither
   layer is perfect -- this is a known hard subproblem, and even
   production ATS tools often ask candidates to confirm or type their
   name directly rather than relying purely on extraction.
"""

import logging
import re
from dataclasses import dataclass, field

from spacy.language import Language

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")

# Matches common phone formats: +1 (555) 123-4567, 555-123-4567,
# 555.123.4567, 5551234567 -- optional country code, flexible separators.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)

_DEGREE_KEYWORDS = (
    r"Bachelor(?:'s)?(?:\s+of\s+\w+)?",
    r"Master(?:'s)?(?:\s+of\s+\w+)?",
    r"MBA",
    r"Ph\.?D\.?",
    r"B\.?Tech\.?",
    r"M\.?Tech\.?",
    r"B\.?Sc\.?",
    r"M\.?Sc\.?",
    r"B\.?E\.?",
    r"M\.?E\.?",
    r"Associate(?:'s)?\s+Degree",
    r"Diploma",
)
_EDUCATION_RE = re.compile(
    r"(?:" + "|".join(_DEGREE_KEYWORDS) + r")[^\n]{0,80}",
    re.IGNORECASE,
)

_EXPERIENCE_YEARS_RE = re.compile(
    r"(\d{1,2}\+?)\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
    re.IGNORECASE,
)

_HEADER_LINE_COUNT = 5
_LINE_SPLIT_RE = re.compile(r"\s*[-|•,]\s*")
_NAME_PATTERN_RE = re.compile(r"^[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,3}$")
_TITLE_KEYWORDS = {
    "resume", "cv", "curriculum", "vitae", "objective", "summary", "profile",
    "contact", "email", "phone", "address", "engineer", "developer",
    "manager", "scientist", "analyst", "designer", "consultant",
    "specialist", "intern", "university", "college", "bachelor", "master",
    "skills", "experience", "education", "projects", "certifications",
    "references", "software", "backend", "frontend", "full", "stack",
    "data", "senior", "junior", "lead",
}


@dataclass
class ExtractedEntities:
    """Structured facts pulled out of a resume's raw text."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    education: list[str] = field(default_factory=list)
    experience_years: float | None = None


class EntityExtractor:
    """Extracts structured contact/background info from resume text.

    Email and phone use regex -- they follow strict, well-defined formats
    that regex handles reliably and far faster than a full NLP pass.
    Education and experience-years use keyword/regex matching, since
    resumes describe degrees and tenure in fairly formulaic phrases
    ("B.Tech in...", "5+ years of experience"). Name uses spaCy NER with
    a regex fallback, restricted to the resume header -- see the module
    docstring for why.
    """

    def __init__(
        self,
        nlp: Language,
    ) -> None:
        self._nlp = nlp

    def extract(self, text: str) -> ExtractedEntities:
        return ExtractedEntities(
            name=self._extract_name(text),
            email=self._extract_email(text),
            phone=self._extract_phone(text),
            education=self._extract_education(text),
            experience_years=self._extract_experience_years(text),
        )

    @staticmethod
    def _extract_email(text: str) -> str | None:
        match = _EMAIL_RE.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        match = _PHONE_RE.search(text)
        return match.group(0).strip() if match else None

    def _extract_name(self, text: str) -> str | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header_lines = lines[:_HEADER_LINE_COUNT]

        # Pass 1: spaCy NER on each header line, and on segments split by
        # common name/title separators (e.g. "Jane Doe | Engineer") --
        # splitting first avoids the model merging a name and a job title
        # into one confusing span.
        for line in header_lines:
            candidate = self._first_person_entity(line)
            if candidate:
                return candidate
            for segment in _LINE_SPLIT_RE.split(line):
                candidate = self._first_person_entity(segment)
                if candidate:
                    return candidate

        # Pass 2: regex fallback for the cases NER misses (see module
        # docstring). Looks for a short, title-cased/all-caps segment
        # that isn't a known job-title or resume-section keyword.
        for line in header_lines:
            candidate = self._regex_name_candidate(line)
            if candidate:
                return candidate
            for segment in _LINE_SPLIT_RE.split(line):
                candidate = self._regex_name_candidate(segment)
                if candidate:
                    return candidate

        return None

    def _first_person_entity(self, segment: str) -> str | None:
        segment = segment.strip()
        if not segment:
            return None
        doc = self._nlp(segment)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    @staticmethod
    def _regex_name_candidate(segment: str) -> str | None:
        segment = segment.strip()
        if not _NAME_PATTERN_RE.match(segment):
            return None
        words = {w.lower() for w in segment.split()}
        if words & _TITLE_KEYWORDS:
            return None
        return segment

    @staticmethod
    def _extract_education(text: str) -> list[str]:
        matches = [m.group(0).strip() for m in _EDUCATION_RE.finditer(text)]
        seen: set[str] = set()
        unique: list[str] = []
        for match in matches:
            key = match.lower()
            if key not in seen:
                seen.add(key)
                unique.append(match)
        return unique

    @staticmethod
    def _extract_experience_years(text: str) -> float | None:
        match = _EXPERIENCE_YEARS_RE.search(text)
        if not match:
            return None
        return float(match.group(1).rstrip("+"))