"""Embedding backends.

    sentence-transformers  all-MiniLM-L6-v2, 384-d, L2-normalised, cosine (proposal)
    hashing                deterministic 384-d hashing vectors for tests / offline dev.
                           Not semantic - never report results produced with it.

NOTE (WP-03/05): all-MiniLM-L6-v2 truncates input at 256 word-pieces, but the
proposal's fixed chunks are ~500 tokens, so the tail of long chunks is not
embedded. ``build_index`` reports how many chunks exceed the limit. Options:
smaller chunks, or a 512-token 384-d model such as BAAI/bge-small-en-v1.5.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from core.config import Settings, get_settings


class Embedder(Protocol):
    name: str
    dimension: int
    normalized: bool
    max_tokens: int | None

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    normalized = True

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.max_tokens = int(getattr(self.model, "max_seq_length", 0) or 0) or None

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))


class HashingEmbedder:
    normalized = True
    max_tokens = None

    def __init__(self, dimension: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dimension}"
        self.dimension = dimension
        self.vectorizer = HashingVectorizer(
            n_features=dimension, alternate_sign=False, norm="l2", ngram_range=(1, 2)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()


def chunk_embedding_text(chunk: dict) -> str:
    """The exact text embedded for a chunk. Index build and lazy indexing
    MUST use this same function, or vectors become inconsistent."""
    return f"{chunk.get('title', '')}\n{chunk.get('text', '')}"


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)


@lru_cache(maxsize=2)
def get_embedder() -> Embedder:
    return build_embedder()
