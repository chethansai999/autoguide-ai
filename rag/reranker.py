"""Cross-encoder reranking for AutoGuide AI.

A bi-encoder (used for dense search) embeds the question and the chunk
separately, which is fast but approximate. A cross-encoder reads the
question and the chunk together and scores how well the chunk answers
the question. It is too slow to run on every chunk, so we only run it
on the 20+ hybrid candidates and keep the best 5.
"""

import os

from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# MiniLM cross-encoders accept at most 512 tokens per pair.
MAX_LENGTH = 512


class CrossEncoderReranker:

    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=MAX_LENGTH)

    def rerank(self, query, candidates, top_k=5):
        """Score each (query, chunk) pair and return the best top_k.

        Adds "rerank_score" (higher is better; raw logit) and
        "rerank_rank" to each candidate. The original retrieval
        rank is kept so the trace can show before/after.
        """

        if not candidates:
            return []

        pairs = [
            (query, f"{candidate['title']}\n{candidate['text']}")
            for candidate in candidates
        ]

        scores = self.model.predict(pairs, show_progress_bar=False)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate["rerank_score"],
            reverse=True
        )

        for rank, candidate in enumerate(ranked, start=1):
            candidate["rerank_rank"] = rank

        return ranked[:top_k]


_reranker = None


def get_reranker():
    """Cross-encoder unless RERANKER_BACKEND=none (offline tests)."""

    global _reranker

    from core.config import get_settings

    if get_settings().reranker_backend == "none":
        return None

    if _reranker is None:
        _reranker = CrossEncoderReranker(get_settings().reranker_model)

    return _reranker
