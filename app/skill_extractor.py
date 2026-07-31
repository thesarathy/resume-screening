"""
Skill extraction: matches resume/job-description text against a curated
skill vocabulary, plus a lower-confidence "candidate skill" pass for
terms that look technical but aren't in the vocabulary yet.

The core matching strategy runs TWO passes, because a single alias list
can't safely handle every skill name:

1. Alias matching (attr="LOWER", case-insensitive). Most skill names are
   unambiguous, so app/data/skills.json lists their common lowercase
   aliases -- e.g. "Machine Learning" -> ["machine learning", "ml"] --
   and any of those aliases match regardless of how they're capitalized
   in the resume.

2. Exact-case canonical-name matching (attr="ORTH", case-sensitive),
   used for every skill's canonical name itself, not just its aliases.
   This exists because a handful of skill names are also ordinary
   English words -- "Go" (the language) vs. "go" (the verb), "R" (the
   language) vs. the letter "r". For these, the taxonomy deliberately
   omits a bare lowercase self-alias (see "Go": ["golang"], no plain
   "go"), so the *only* way "Go" is recognized at all is this exact-case
   pass. But capitalization alone isn't quite enough: "Go above and
   beyond..." also capitalizes "Go" purely because it starts the
   sentence, not because it's the language. So exact-case matches at the
   very start of a sentence are discarded -- a token's capitalization is
   only trusted as a signal when it isn't explained by sentence position.

Skill candidate discovery (extract_candidate_skills) is a separate,
lower-confidence pass: no fixed vocabulary keeps up with every tool or
acronym a candidate might list, so we additionally scan for short,
technical-looking tokens (ALL-CAPS acronyms like "ETL", or letter+digit
mixes like "S3") not already covered by the vocabulary, merging adjacent
technical tokens into one candidate (e.g. "ABC" + "DEF" -> "ABC DEF").
These are suggestions worth a human's attention, not asserted skills.
"""

import json
import logging
from pathlib import Path

from spacy.language import Language
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent / "data" / "skills.json"
_MIN_TOKEN_LENGTH = 2


def load_skill_taxonomy(path: Path = _DEFAULT_SKILLS_PATH) -> dict[str, dict]:
    """Load the curated skill vocabulary from disk.

    Kept as a JSON data file rather than a Python dict in source code so
    the skill list -- its aliases and categories -- can be extended
    without touching any logic; a non-engineer on a team could safely
    edit it. Each entry has the shape:
        "Canonical Name": {"aliases": [...], "category": "..."}
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SkillExtractionResult:
    """Result of extracting skills from one piece of text.

    `skills` is the flat, deduplicated, sorted list of canonical skill
    names found. `by_category()` groups those same names by taxonomy
    category (e.g. "language", "tool", "cloud"), computed on demand
    rather than stored twice.
    """

    def __init__(self, skills: list[str], category_of: dict[str, str]) -> None:
        self.skills = skills
        self._category_of = category_of

    def by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for skill in self.skills:
            category = self._category_of.get(skill, "other")
            grouped.setdefault(category, []).append(skill)
        return {category: sorted(names) for category, names in grouped.items()}

    def __repr__(self) -> str:
        return f"SkillExtractionResult(skills={self.skills!r})"


class SkillExtractor:
    """Extracts skills from text using a curated alias vocabulary +
    PhraseMatcher, with an optional secondary pass for undocumented
    technical terms.
    """

    def __init__(
        self,
        nlp: Language,
        taxonomy: dict[str, dict] | None = None,
    ) -> None:
        self._nlp = nlp
        self._taxonomy = taxonomy if taxonomy is not None else load_skill_taxonomy()
        self._known_aliases: set[str] = set()
        self._category_of: dict[str, str] = {}
        self._alias_matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._exact_matcher = PhraseMatcher(self._nlp.vocab, attr="ORTH")
        self._build_matchers()

    def _build_matchers(self) -> None:
        for canonical_name, info in self._taxonomy.items():
            aliases = info["aliases"]
            self._category_of[canonical_name] = info.get("category", "other")

            alias_patterns = [self._nlp.make_doc(alias) for alias in aliases]
            self._alias_matcher.add(canonical_name, alias_patterns)
            self._known_aliases.update(alias.lower() for alias in aliases)

            exact_pattern = [self._nlp.make_doc(canonical_name)]
            self._exact_matcher.add(canonical_name, exact_pattern)

    def extract(self, text: str) -> SkillExtractionResult:
        doc = self._nlp(text)
        found: set[str] = set()

        for match_id, _, _ in self._alias_matcher(doc):
            found.add(self._nlp.vocab.strings[match_id])

        for match_id, start, _ in self._exact_matcher(doc):
            if doc[start].is_sent_start:
                continue
            found.add(self._nlp.vocab.strings[match_id])

        return SkillExtractionResult(
            skills=sorted(found),
            category_of=self._category_of,
        )

    def extract_candidate_skills(self, text: str) -> list[str]:
        """Surface short, technical-looking tokens not already covered by
        the vocabulary -- lower-confidence suggestions, not asserted skills.
        """
        doc = self._nlp(text)
        candidates: dict[str, str] = {}
        current_span: list[str] = []

        def flush() -> None:
            if current_span:
                phrase = " ".join(current_span)
                if phrase.lower() not in self._known_aliases:
                    candidates[phrase.lower()] = phrase
                current_span.clear()

        for token in doc:
            if self._looks_technical(token.text):
                current_span.append(token.text)
            else:
                flush()
        flush()

        return sorted(candidates.values())

    @staticmethod
    def _looks_technical(token_text: str) -> bool:
        """A token "looks technical" if it's an all-caps acronym (e.g.
        "ETL", "CRM") or mixes letters and digits (e.g. "S3", "Python3"),
        rather than an ordinary English word.
        """
        if len(token_text) < _MIN_TOKEN_LENGTH:
            return False
        if token_text.isupper() and token_text.isalpha():
            return True
        has_letter = any(c.isalpha() for c in token_text)
        has_digit = any(c.isdigit() for c in token_text)
        return has_letter and has_digit