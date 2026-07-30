"""
Text preprocessing for resumes and job descriptions.

Two preprocessing "strengths" are provided because the two similarity
methods used later in this project want different things:

- TF-IDF (Method 1) is a bag-of-words model -- it only cares which words
  appear and how often. It benefits from aggressive normalization:
  lowercasing, stopword removal, lemmatization. This collapses "manages",
  "managing", "managed" into one shared signal instead of three separate
  ones.

- Sentence-Transformer embeddings (Method 2) are trained on natural,
  grammatical sentences. Stripping stopwords and lemmatizing *hurts*
  their quality -- e.g. "not experienced with Python" and "experienced
  with Python" collapse into nearly the same bag of words once "not" is
  removed as a stopword, but an embedding model relies on that word to
  get the meaning right. So embeddings should only get light cleaning,
  not full normalization.

clean_text()                -> light cleaning, safe for embeddings.
TextPreprocessor.normalize() -> full pipeline (clean + tokenize + lemmatize
                                 + stopword removal), for TF-IDF.
"""

import logging
import re

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

_BULLET_CHARS = "•◦▪●■‣∙·"
_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(rf"^[{re.escape(_BULLET_CHARS)}]\s*", re.MULTILINE)


def clean_text(text: str) -> str:
    """Light cleanup that is safe for both similarity methods.

    Strips leading bullet characters and collapses repeated whitespace,
    but preserves casing, punctuation, and stopwords -- sentence
    embeddings need those to understand meaning and negation correctly.
    """
    text = _BULLET_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


class TextPreprocessor:
    """Full NLP normalization pipeline, built on spaCy.

    Loading a spaCy model takes real time (tens to hundreds of ms), so
    this class should be instantiated once per process (e.g. once at
    app startup) and reused for every resume, not re-created per file.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp: Language = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

    def normalize(self, text: str) -> str:
        """Full normalization pipeline for TF-IDF.

        clean -> tokenize -> lowercase -> drop stopwords/punctuation ->
        lemmatize. Returns a single space-joined string of lemmas, ready
        to hand to scikit-learn's TfidfVectorizer.
        """
        text = clean_text(text)
        doc = self._nlp(text)

        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
        ]
        return " ".join(lemmas)