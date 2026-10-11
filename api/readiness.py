"""Inspect local deployment artifacts without loading models or calling APIs."""

from __future__ import annotations

import json
import logging
from collections import Counter
from importlib.util import find_spec
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from core.config import PLACEHOLDER_KEYS
from db.models import IndexRegistry, Vehicle

if TYPE_CHECKING:
    from agent.orchestrator import ChatService
    from core.config import Settings

log = logging.getLogger(__name__)


def _vector_ids(settings: Settings, collection: str) -> set[str]:
    # PersistentClient creates a database if one is missing. A readiness probe
    # must report that missing artifact instead of creating an empty index.
    if not (settings.chroma_db_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError("Chroma database is missing")
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    index = client.get_collection(collection, embedding_function=None)
    return set(index.get(include=[])["ids"])


def inspect_readiness(service: ChatService) -> dict:
    """Check configuration and artifact consistency, not external availability.

    Never instantiate NLU, embedding, reranking or generation models here.
    A successful probe is not a measured answer-quality result.
    """
    settings = service.settings
    provider = settings.resolved_llm_provider()
    collection = settings.collection_name()
    with service.session_factory() as db:
        vehicles = db.scalar(select(func.count()).select_from(Vehicle)) or 0
        active = db.scalars(
            select(IndexRegistry).where(IndexRegistry.is_active.is_(True))
            .order_by(IndexRegistry.strategy, IndexRegistry.collection)
        ).all()
    selected = next((index for index in active if index.collection == collection), None)
    checks = {"vehicles": vehicles > 0}
    blockers: list[dict[str, str]] = []

    def check(name: str, passed: bool, message: str) -> None:
        checks[name] = passed
        if not passed:
            blockers.append({"code": name, "message": message})

    if not checks["vehicles"]:
        blockers.append({"code": "vehicles", "message": "Load the vehicle CSV: python ingest.py load-safercar"})
    expected_model = "hashing-384" if settings.embedding_backend == "hashing" else settings.embedding_model
    check(
        "active_index",
        bool(selected and selected.strategy == settings.chunk_strategy
             and selected.version == settings.chunk_version
             and selected.embedding_model == expected_model and selected.chunk_count > 0),
        f"Build an index for the configured model and strategy: python ingest.py index --strategy {settings.chunk_strategy}",
    )

    chunk_ids: list[str] = []
    chunks_valid = False
    indexed_sources: dict[str, int] = {}
    try:
        path = settings.chunks_dir() / "chunks.jsonl"
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        chunks_valid = bool(records) and all(
            isinstance(row, dict) and isinstance(row.get("id"), str)
            and bool(row["id"]) and isinstance(row.get("text"), str) and bool(row["text"].strip())
            for row in records
        )
        if chunks_valid:
            chunk_ids = [row["id"] for row in records]
            chunks_valid = len(set(chunk_ids)) == len(chunk_ids)
            if chunks_valid:
                indexed_sources = dict(sorted(Counter(row.get("source_type") or "unknown" for row in records).items()))
    except (OSError, ValueError):
        log.debug("Chunk file is missing or unreadable", exc_info=True)
    check("chunks", chunks_valid,
          f"Build valid chunks: python ingest.py chunk --strategy {settings.chunk_strategy}")

    vectors_valid = False
    if checks["active_index"] and chunks_valid:
        try:
            vectors_valid = (selected.chunk_count == len(chunk_ids)
                             and _vector_ids(settings, collection) == set(chunk_ids))
        except Exception:
            log.warning("Configured vector collection is unavailable", exc_info=True)
    check("vector_index", vectors_valid,
          f"The vector collection must match the chunk file; rebuild: python ingest.py index --strategy {settings.chunk_strategy}")

    key = (getattr(settings, f"{provider}_api_key", None) or "").strip()
    check("generation_configured", provider != "mock" and key not in PLACEHOLDER_KEYS,
          "Configure a supported LLM provider and its API key in .env; mock output is for development only.")
    check("semantic_embeddings", settings.embedding_backend != "hashing",
          "Use sentence-transformers embeddings for a real deployment; hashing is for offline tests.")

    nlu_present = True
    if settings.nlu_backend == "hf":
        nlu_present = find_spec("transformers") is not None and all(
            (settings.nlu_model_dir / head / "config.json").is_file()
            and (settings.nlu_model_dir / head / "tokenizer_config.json").is_file()
            and any((settings.nlu_model_dir / head / name).is_file()
                    for name in ("model.safetensors", "pytorch_model.bin"))
            for head in ("intent", "slots")
        )
    check("nlu_artifacts", nlu_present,
          "Install the fine-tuning dependencies and train both NLU heads, or set NLU_BACKEND=rules.")

    return {
        "status": "ready" if not blockers else "degraded",
        "ready": not blockers,
        "scope": "local_artifacts_and_configuration",
        "external_services": "not_checked",
        "vehicles_loaded": vehicles,
        "active_indexes": {i.strategy: {"collection": i.collection, "chunks": i.chunk_count} for i in active},
        "chunk_strategy": settings.chunk_strategy,
        "chunks_loaded": len(chunk_ids),
        "indexed_sources": indexed_sources,
        "llm_provider": provider,
        "llm_model": getattr(settings, f"{provider}_model", "mock"),
        "nlu_backend": settings.nlu_backend,
        "embedding_backend": settings.embedding_backend,
        "reranker_backend": settings.reranker_backend,
        "demo": provider == "mock" or settings.embedding_backend == "hashing",
        "checks": checks,
        "blockers": blockers,
        "warnings": ["Provider connectivity, model downloads, live recalls and answer quality require separate smoke checks."],
    }
