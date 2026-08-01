"""
Similarity scoring: compares a job description against one or more
resumes and returns a similarity score for each.

Two independent implementations live here -- TF-IDF and
Sentence-Transformer embeddings -- both behind the same
BaseSimilarityScorer interface (the Strategy pattern, same approach
used for resume parsing in app/parser.py). This lets the app expose a
"ranking method" toggle without ranking.py needing to know or care
which concrete scorer it's talking to.

Design note on preprocessing: each scorer decides its own preprocessing
internally rather than the caller deciding for it. TF-IDF is a
bag-of-words model, so it gets the FULLY normalized text
(TextPreprocessor.normalize() -- lowercased, lemmatized, stopwords
removed). Sentence-Transformer embeddings want the opposite -- natural,
un-stripped sentences -- see app/preprocessing.py's module docstring
for why. Encapsulating that choice inside each scorer is what keeps the
two methods truly swappable: the caller always passes raw resume/JD
text and gets a score back, regardless of which method is selected.
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import TextPreprocessor, clean_text

logger = logging.getLogger(__name__)


class BaseSimilarityScorer(ABC):
    """Common interface every similarity method must implement."""

    @abstractmethod
    def score(self, job_description: str, resumes: list[str]) -> list[float]:
        """Return a similarity score in [0, 1] for each resume against the
        job description, in the same order as `resumes`.
        """
        raise NotImplementedError


class TfidfSimilarityScorer(BaseSimilarityScorer):
    """Similarity via TF-IDF vectors + cosine similarity.

    TF-IDF represents each document as a vector of word-importance
    scores: a word counts for more the more often it appears in *this*
    document, but less the more common it is *across all documents* --
    so shared filler words contribute little, while distinctive terms
    (both documents mentioning "Kubernetes", say) contribute a lot.
    Cosine similarity then measures the angle between the job
    description's vector and each resume's vector -- 1.0 means identical
    direction (perfect topical overlap), 0.0 means no shared vocabulary
    at all.

    Important implementation detail: the TfidfVectorizer is fit fresh
    inside every call to score(), not built once and reused. Unlike a
    pretrained embedding model (the sentence-transformer scorer below),
    TF-IDF's vocabulary and IDF weights are derived entirely from
    whatever document set is passed in -- refitting per batch is correct,
    not wasteful, since a vectorizer fit on one job description's resume
    pool would give meaningless IDF weights for a completely different
    job posting.

    Trade-off vs. Sentence-Transformer embeddings: TF-IDF only "sees"
    exact vocabulary overlap -- a resume saying "constructed REST
    services" won't score well against a JD saying "built APIs", even
    though they mean the same thing, because the words themselves don't
    match. It's fast, has zero model-download cost, and is fully
    interpretable (you can inspect exactly which words drove the score)
    -- but it has no notion of meaning, only word overlap.
    """

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self._preprocessor = preprocessor

    def score(self, job_description: str, resumes: list[str]) -> list[float]:
        if not resumes:
            return []

        documents = [self._preprocessor.normalize(job_description)] + [
            self._preprocessor.normalize(resume) for resume in resumes
        ]

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
        except ValueError:
            # Raised by scikit-learn when every document is empty after
            # normalization (e.g. all stopwords, or blank text) -- there's
            # no vocabulary to vectorize, so nothing can be compared.
            logger.warning(
                "TF-IDF vocabulary is empty after normalization; "
                "returning zero similarity for all resumes."
            )
            return [0.0] * len(resumes)

        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(jd_vector, resume_vectors)[0]
        return [float(s) for s in similarities]


class EmbeddingModel(Protocol):
    """Structural type for anything that can encode text into vectors.

    Defined as a Protocol (duck typing) rather than a base class, so a
    real SentenceTransformer instance satisfies it automatically with no
    inheritance needed -- and, importantly, so tests can substitute a
    tiny fake object implementing just this one method, without needing
    the real ~90MB pretrained model or its network download at all.
    """

    def encode(self, texts: list[str]):
        ...


def load_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Load a pretrained Sentence-Transformer model.

    The import is deferred to inside this function, rather than at the
    top of the module, because sentence-transformers pulls in PyTorch --
    a large dependency that only the embedding-based scorer needs. Code
    that only uses TfidfSimilarityScorer (or tests that only exercise
    it) never has to import torch at all.

    Downloads the model from Hugging Face Hub the first time it's called
    on a given machine (cached locally afterward), so this call needs an
    internet connection at least once.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerSimilarityScorer(BaseSimilarityScorer):
    """Similarity via Sentence-Transformer embeddings + cosine similarity.

    Where TF-IDF only sees exact vocabulary overlap, a Sentence-Transformer
    model (here, "all-MiniLM-L6-v2") encodes each piece of text into a
    dense vector that captures *meaning*, learned from training on
    hundreds of millions of sentence pairs. Two sentences phrased
    completely differently -- "built REST APIs" and "developed web
    services" -- land close together in this vector space, because the
    model has learned they mean similar things. Cosine similarity between
    these embeddings then reflects semantic closeness, not just word
    overlap.

    Design note on preprocessing: this scorer calls the LIGHT clean_text()
    function, not TextPreprocessor.normalize(). Embedding models are
    trained on natural, grammatical sentences -- stripping stopwords or
    lemmatizing would remove exactly the structure the model relies on to
    understand meaning (see app/preprocessing.py's module docstring).

    Trade-off vs. TF-IDF: far better at matching paraphrased or
    differently-worded skills/experience, at the cost of a ~90MB
    pretrained model download on first use, slower inference, and much
    less interpretability -- there's no simple way to point at "this
    word" as the reason for a given score, the way there is with TF-IDF.
    """

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    def score(self, job_description: str, resumes: list[str]) -> list[float]:
        if not resumes:
            return []

        documents = [clean_text(job_description)] + [
            clean_text(resume) for resume in resumes
        ]
        embeddings = self._model.encode(documents)

        jd_embedding = embeddings[0:1]
        resume_embeddings = embeddings[1:]

        similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]
        # Embedding cosine similarity can theoretically be negative
        # (opposite-direction vectors), unlike TF-IDF's non-negative
        # vectors -- clip to the same [0, 1] contract every scorer promises.
        return [max(0.0, float(s)) for s in similarities]
