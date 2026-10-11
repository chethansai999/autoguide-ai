"""Evidence sufficiency check: retrieve, rerank, decide answer vs. refuse.

Contract (used by rag/chatbot.py and agent/orchestrator.py):

    EvidenceChecker().check_evidence(question, vehicle_filter=None, ...) -> {
        "sufficient": bool,
        "reason": "ok" | "no_evidence" | "low_relevance" | "unsupported_prediction"
                  | "index_unavailable",
        "evidence": [result dicts, best first, <= FINAL_K],
        "candidates": [all candidates before rerank, with scores],
        "top_score": float | None,
        "threshold": float,
    }

Thresholds come from .env (MIN_RERANK_SCORE / MIN_RETRIEVAL_SCORE) and must
be tuned on the DEV split only (WP-05).
"""

from __future__ import annotations

import logging

from agent.lexicon import PREDICTION
from core.config import Settings, get_settings
from rag.filters import matches_scope

log = logging.getLogger(__name__)


class EvidenceChecker:
    def __init__(self, retriever=None, settings: Settings | None = None):
        self._retriever = retriever
        self.settings = settings or get_settings()

    def retriever_for(self, strategy: str | None):
        if self._retriever is not None:
            return self._retriever
        from rag.retrieval import get_retriever

        return get_retriever(strategy)

    def check_evidence(
        self,
        question: str,
        vehicle_filter: dict | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        strategy: str | None = None,
        source_types: list[str] | None = None,
    ) -> dict:
        settings = self.settings
        empty = {"evidence": [], "candidates": [], "top_score": None, "threshold": None}

        if PREDICTION.search(question):
            return {"sufficient": False, "reason": "unsupported_prediction", **empty}

        try:
            retriever = self.retriever_for(strategy)
        except ValueError as error:  # no chunks yet
            log.warning("Retriever unavailable: %s (build it: python ingest.py crawl, "
                        "extract-autoguide, chunk, index)", error)
            return {"sufficient": False, "reason": "index_unavailable", **empty}

        use_rerank = settings.reranker_backend != "none" if rerank is None else rerank
        filters = {}
        if vehicle_filter is not None or source_types is not None:
            filters = {"vehicle_filter": vehicle_filter, "source_types": source_types}
        results, candidates = retriever.retrieve_with_trace(
            question,
            top_k=settings.final_k,
            mode=mode or settings.retrieval_mode,
            candidate_k=settings.candidate_k,
            rerank=use_rerank,
            **filters,
        )

        if filters:
            results = [r for r in results if matches_scope(r.get("chunk", r), vehicle_filter, source_types)]
            candidates = [r for r in candidates if matches_scope(r.get("chunk", r), vehicle_filter, source_types)]

        if not results:
            return {"sufficient": False, "reason": "no_evidence", **empty, "candidates": candidates}

        reranked = use_rerank and getattr(retriever, "reranker", None) is not None
        top = results[0]
        if reranked and top.get("rerank_score") is not None:
            score, threshold = top["rerank_score"], settings.min_rerank_score
        elif top.get("dense_score") is not None:
            score = max(r.get("dense_score") or 0.0 for r in results)
            threshold = settings.min_dense_score
        else:  # BM25 only: any keyword overlap counts
            score, threshold = top.get("bm25_score") or 0.0, 1e-9

        if score < threshold:
            return {
                "sufficient": False,
                "reason": "low_relevance",
                "evidence": results,
                "candidates": candidates,
                "top_score": score,
                "threshold": threshold,
            }

        return {
            "sufficient": True,
            "reason": "ok",
            "evidence": results,
            "candidates": candidates,
            "top_score": score,
            "threshold": threshold,
        }
