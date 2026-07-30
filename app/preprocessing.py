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


def load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load a spaCy pipeline, with a clear error if it isn't installed.

    Pulled out as a standalone function (rather than kept private inside
    TextPreprocessor) so other modules -- like EntityExtractor, added in
    the next step -- can share the exact same loading/error-handling
    logic. More importantly, it lets the app load the model **once** at
    startup and inject that single Language object into every class that
    needs it, instead of each class silently loading its own copy.
    Loading a spaCy model takes real time (tens to hundreds of ms) and
    real memory, so sharing one instance across the app matters once
    there's more than one consumer of it.
    """
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


class TextPreprocessor:
    """Full NLP normalization pipeline, built on spaCy.

    Accepts an already-loaded spaCy pipeline via `nlp` (preferred in the
    running app, so it can share one pipeline with EntityExtractor), or
    falls back to loading its own by `model_name` if none is given (handy
    for quick scripts, notebooks, or tests that only need this one class).
    """

    def __init__(
        self,
        nlp: Language | None = None,
        model_name: str = "en_core_web_sm",
    ) -> None:
        self._nlp: Language = nlp if nlp is not None else load_spacy_model(model_name)

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