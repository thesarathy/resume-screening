"""
Dependency construction: builds the shared spaCy pipeline and maps a
ranking-method name to a concrete similarity scorer.

Centralizing this here (instead of scattering it through the route
handlers) means the app loads the spaCy model exactly once and reuses
that single pipeline for every extractor -- TextPreprocessor,
EntityExtractor, and SkillExtractor all share it. It also keeps the
method-swap decision in one place, per the Strategy pattern used
throughout the engine (see app/ranking.py's module docstring).
"""

import logging

from app.preprocessing import TextPreprocessor, load_spacy_model
from app.similarity import (
    BaseSimilarityScorer,
    SentenceTransformerSimilarityScorer,
    TfidfSimilarityScorer,
)

logger = logging.getLogger(__name__)

# Method identifiers as sent from the UI form.
METHOD_TFIDF = "tfidf"
METHOD_EMBEDDINGS = "embeddings"

VALID_METHODS = (METHOD_TFIDF, METHOD_EMBEDDINGS)


class ScorerFactory:
    """Builds similarity scorers from a method name.

    Both heavy models are loaded lazily rather than at app startup:

    - The spaCy pipeline is built on first access and cached, so a
      memory-constrained host doesn't tip over at boot (see nlp).
    - The SentenceTransformer model is only built when the embeddings
      method is actually requested -- it pulls in PyTorch and a ~90MB
      pretrained model download, which would be wasted (and heavy) at
      startup. Requests for the default TF-IDF method never pay that cost.
    """

    def __init__(self, nlp=None) -> None:
        # Loaded lazily on first access (see nlp); a caller may inject a
        # prebuilt pipeline (e.g. a lighter fake in tests) to skip loading.
        self._nlp = nlp
        self._embedding_scorer: BaseSimilarityScorer | None = None

    @property
    def nlp(self):
        """The single shared spaCy pipeline used across all extractors,
        loaded on first access and cached for the lifetime of the process.
        """
        if self._nlp is None:
            self._nlp = load_spacy_model()
        return self._nlp

    def build_scorer(self, method: str = METHOD_TFIDF) -> BaseSimilarityScorer:
        """Return a scorer for the given method name.

        Raises:
            ValueError: if `method` is not one of VALID_METHODS.
        """
        if method == METHOD_TFIDF:
            return TfidfSimilarityScorer(TextPreprocessor(nlp=self.nlp))
        if method == METHOD_EMBEDDINGS:
            # Build once, then cache and reuse for subsequent requests in
            # this process -- the model is too heavy to reload per request.
            if self._embedding_scorer is None:
                logger.info("Loading SentenceTransformer model (first use)...")
                from app.similarity import load_sentence_transformer_model

                self._embedding_scorer = SentenceTransformerSimilarityScorer(
                    load_sentence_transformer_model()
                )
            return self._embedding_scorer
        raise ValueError(f"Unknown ranking method: {method!r}")
