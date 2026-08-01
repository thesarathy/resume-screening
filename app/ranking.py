"""
Candidate ranking: the orchestration layer that ties the rest of the
engine together.

Given a job description and a batch of resume texts, a ResumeRanker:
  1. extracts structured facts per resume (entities + skills),
  2. asks a similarity scorer for a match score per resume,
  3. pairs the scores back onto the candidates and returns them sorted
     by score, best first.

It deliberately knows nothing about *which* scoring method is used. It
only talks to a BaseSimilarityScorer (the Strategy pattern used across
this project), so the app can swap TF-IDF for Sentence-Transformer -- or
add a third method later -- without changing a line here. That decision
lives in app/factory.py, which maps a method name to a concrete scorer.
"""

import logging
from dataclasses import asdict, dataclass, field

from spacy.language import Language

from app.entity_extractor import EntityExtractor, ExtractedEntities
from app.parser import ResumeParser
from app.similarity import BaseSimilarityScorer
from app.skill_extractor import SkillExtractor

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single screened candidate, with the facts we surfaced plus their
    match score against the job description.

    `score` is the raw similarity score in [0, 1]. `match_pct` is the same
    value presented as a recruiter-friendly 0-100 percentage -- computed
    on demand so it can never drift out of sync with the raw score.
    """

    filename: str
    score: float
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    education: list[str] = field(default_factory=list)
    experience_years: float | None = None
    skills: list[str] = field(default_factory=list)
    candidate_skills: list[str] = field(default_factory=list)

    @property
    def match_pct(self) -> int:
        """The similarity score as an integer percentage for display."""
        return round(self.score * 100)

    def as_dict(self) -> dict:
        """Plain dict for rendering to JSON or CSV."""
        return asdict(self)


class ResumeRanker:
    """Scores and ranks a batch of resumes against one job description.

    The extractors are injected (defaulting to real ones built from the
    shared `nlp` pipeline), so tests can substitute fakes -- or reuse the
    already-shared pipeline in the running app rather than loading spaCy
    models more than once.
    """

    def __init__(
        self,
        nlp: Language,
        scorer: BaseSimilarityScorer,
        parser: ResumeParser | None = None,
        entity_extractor: EntityExtractor | None = None,
        skill_extractor: SkillExtractor | None = None,
    ) -> None:
        self._nlp = nlp
        self._scorer = scorer
        self._parser = parser if parser is not None else ResumeParser()
        self._entity_extractor = (
            entity_extractor if entity_extractor is not None else EntityExtractor(nlp)
        )
        self._skill_extractor = (
            skill_extractor if skill_extractor is not None else SkillExtractor(nlp)
        )

    def rank(
        self,
        job_description: str,
        resumes: list[tuple[str, str]],
    ) -> list[Candidate]:
        """Score every resume against the job description and return the
        candidates sorted by score, best first.

        Args:
            job_description: The raw job posting text.
            resumes: A list of ``(filename, resume_text)`` pairs.

        Returns:
            A list of :class:`Candidate`, highest score first. If no
            resumes are supplied, returns an empty list.
        """
        if not resumes:
            return []

        # Extract candidate facts first (needs per-resume text), then ask
        # the scorer for one similarity score per resume in the same order.
        candidates: list[Candidate] = []
        resume_texts: list[str] = []

        for filename, text in resumes:
            entities = self._entity_extractor.extract(text)
            skills = self._skill_extractor.extract(text).skills
            candidate_skills = self._skill_extractor.extract_candidate_skills(text)
            candidates.append(
                self._to_candidate(
                    filename, entities, skills, candidate_skills, score=0.0
                )
            )
            resume_texts.append(text)

        scores = self._scorer.score(job_description, resume_texts)

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        return sorted(candidates, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _to_candidate(
        filename: str,
        entities: ExtractedEntities,
        skills: list[str],
        candidate_skills: list[str],
        score: float,
    ) -> Candidate:
        return Candidate(
            filename=filename,
            name=entities.name,
            email=entities.email,
            phone=entities.phone,
            education=entities.education,
            experience_years=entities.experience_years,
            skills=skills,
            candidate_skills=candidate_skills,
            score=score,
        )
