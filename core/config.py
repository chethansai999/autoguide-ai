"""Central configuration for every AutoGuide component.

All tunables live here and are read from environment variables or `.env`.
Field names map to env vars case-insensitively (e.g. ``llm_provider`` <-
``LLM_PROVIDER``). Import ``get_settings()``; never read ``os.environ``
directly in new code.

Offline / no-key development profile (used by the test-suite):
    LLM_PROVIDER=mock  EMBEDDING_BACKEND=hashing  RERANKER_BACKEND=none
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

ChunkStrategy = Literal["fixed", "structure"]
RetrievalMode = Literal["hybrid", "bm25", "dense"]

PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-...", "changeme"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = ROOT / "data"
    safercar_csv: Path = ROOT / "data" / "structured" / "Safercar_data.csv"
    models_dir: Path = ROOT / "models"

    # -------------------------------------------------------------- databases
    # SQLite by default (zero setup). Postgres in docker-compose:
    #   DATABASE_URL=postgresql+psycopg://autoguide:autoguide@db:5432/autoguide
    database_url: str = f"sqlite:///{(ROOT / 'data' / 'autoguide.db').as_posix()}"
    chroma_db_path: Path = ROOT / "data" / "chroma_db"
    # Optional hard override; normally derived by collection_name().
    chroma_collection: str | None = None

    # -------------------------------------------------------------- retrieval
    embedding_backend: Literal["sentence-transformers", "hashing"] = (
        "sentence-transformers"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_backend: Literal["cross-encoder", "none"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_strategy: ChunkStrategy = "fixed"   # switch to "structure" once WP-03 lands (B3)
    chunk_version: str = "v1"
    fixed_chunk_tokens: int = 500
    fixed_chunk_overlap: int = 100

    retrieval_mode: RetrievalMode = "hybrid"
    candidate_k: int = 20          # proposal: >= 20 candidates before rerank
    final_k: int = 5               # proposal: <= 5 chunks to the LLM
    # Evidence sufficiency thresholds (tune on the DEV split only - WP-05).
    # RRF scores are rank-based and say nothing about relevance, so the
    # check uses the cross-encoder logit when reranking, else the top
    # chunk's dense cosine similarity, else any BM25 keyword overlap.
    min_rerank_score: float = -2.0       # ms-marco MiniLM logit
    min_dense_score: float = 0.30        # cosine similarity

    # -------------------------------------------------------------------- llm
    # "auto" -> openai if OPENAI_API_KEY is set, else anthropic if
    # ANTHROPIC_API_KEY is set, else gemini if GEMINI_API_KEY is set, else
    # mock (deterministic, offline).
    llm_provider: Literal["auto", "openai", "anthropic", "gemini", "mock"] = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    # Google Gemini (AI Studio key, including the newer "AQ." keys). Sent in the
    # x-goog-api-key header to the native generateContent endpoint.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    llm_timeout_s: float = 30.0
    # Retries for transient network errors; the SDKs back off between tries.
    llm_max_retries: int = 2
    # Fail fast on a broken network: a host whose DNS lookup takes longer than
    # this is treated as unreachable and skipped for the cool-down period
    # (core/netcheck.py). Applies to the LLM and the NHTSA Recalls API.
    net_dns_timeout_s: float = 4.0
    net_unreachable_cooldown_s: float = 60.0
    # If the LLM can't be reached (DNS, connection or timeout; never a bad key
    # or quota), answer from the evidence extractively and say so in the
    # answer. Evaluation runs always turn this off.
    llm_fallback_on_connection_error: bool = True
    # USD per 1M tokens, used for the cost-per-query metric. Verify against
    # your provider's current price sheet before reporting.
    llm_price_input_per_mtok: float = 0.15
    llm_price_output_per_mtok: float = 0.60

    # ------------------------------------------------------------------ agent
    nlu_backend: Literal["rules", "hf", "llm"] = "rules"
    nlu_model_dir: Path = ROOT / "models" / "clarifier"
    # NLU_BACKEND=llm: model used for NLU, e.g. an OpenAI fine-tune id
    # "ft:gpt-4o-mini-...:autoguide-nlu:..." printed by finetune/openai_finetune.py.
    # Empty -> the generation model is reused with a prompt.
    nlu_llm_model: str | None = None
    max_clarification_turns: int = 3
    history_turns: int = 6

    # -------------------------------------------------------------------- api
    # When set, the Streamlit UI calls this HTTP API; otherwise it runs the
    # ChatService in-process (simplest for demos).
    autoguide_api_url: str | None = None
    cors_origins: list[str] = ["http://localhost:8501"]

    # --------------------------------------------------------------- external
    nhtsa_api_base: str = "https://api.nhtsa.gov"
    enable_live_recalls_api: bool = True
    recalls_cache_ttl_hours: int = 24
    ratings_citation_url: str = "https://www.nhtsa.gov/ratings"
    crawler_contact_email: str | None = None

    # ---------------------------------------------------------------- helpers
    def chunks_dir(self, strategy: str | None = None) -> Path:
        return self.data_dir / "chunks" / (strategy or self.chunk_strategy)

    def collection_name(self, strategy: str | None = None) -> str:
        """Chroma collection for a (strategy, embedding model, version).

        One collection per configuration lets B2 (fixed) and B3 (structure)
        coexist and makes re-chunking a build-then-activate swap.
        """
        if self.chroma_collection:
            return self.chroma_collection
        model = "hashing" if self.embedding_backend == "hashing" else self.embedding_model
        model_slug = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower())
        name = f"nhtsa_{strategy or self.chunk_strategy}_{model_slug}_{self.chunk_version}"
        return name[:63].strip("-_")

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if (self.openai_api_key or "").strip() not in PLACEHOLDER_KEYS:
            return "openai"
        if (self.anthropic_api_key or "").strip() not in PLACEHOLDER_KEYS:
            return "anthropic"
        if (self.gemini_api_key or "").strip() not in PLACEHOLDER_KEYS:
            return "gemini"
        return "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
