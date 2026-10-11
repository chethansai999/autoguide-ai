"""ChromaDB index management (build, activate, delete, verify).

One collection per (strategy, embedding model, version), e.g.
``nhtsa_structure_all-minilm-l6-v2_v1``. Rebuild procedure (proposal s.5):

    1. python ingest.py chunk --strategy structure      (new chunks.jsonl)
    2. python ingest.py index --strategy structure      (fresh collection)
    3. build_index verifies: count == chunks, no orphan IDs, no missing IDs
    4. the collection is registered + marked active in index_registry
    5. evaluate old vs new on the SAME dev questions before switching

    python ingest.py index --strategy fixed
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from db.models import ChunkRow, IndexRegistry
from rag.embeddings import Embedder, chunk_embedding_text, get_embedder

log = logging.getLogger(__name__)

BATCH = 64


def _flat_metadata(record: dict) -> dict:
    """Chroma metadata must be flat str/int/float/bool."""
    meta = {}
    for key, value in record.items():
        if key in ("id", "text", "chunk_id"):
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    meta[f"{key}_{sub_key}"] = sub_value
    return meta


class ChromaVectorStore:
    def __init__(self, collection_name: str, embedder: Embedder | None = None, path: Path | None = None):
        import chromadb

        self.settings = get_settings()
        self.embedder = embedder or get_embedder()
        self.client = chromadb.PersistentClient(path=str(path or self.settings.chroma_db_path))
        self.name = collection_name

    def collection(self, create: bool = True):
        metadata = {
            "hnsw:space": "cosine",
            "embedding_model": self.embedder.name,
            "dimension": self.embedder.dimension,
            "normalized": self.embedder.normalized,
        }
        if create:
            return self.client.get_or_create_collection(self.name, metadata=metadata)
        return self.client.get_collection(self.name)

    def drop(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass

    def upsert(self, records: list[dict]) -> None:
        collection = self.collection()
        for start in range(0, len(records), BATCH):
            batch = records[start:start + BATCH]
            collection.upsert(
                ids=[r["id"] for r in batch],
                documents=[r["text"] for r in batch],
                embeddings=self.embedder.embed([chunk_embedding_text(r) for r in batch]),
                metadatas=[_flat_metadata(r) for r in batch],
            )

    def ids(self) -> set[str]:
        return set(self.collection().get(include=[])["ids"])

    def count(self) -> int:
        return self.collection().count()

    def delete_document(self, doc_id: str) -> None:
        self.collection().delete(where={"doc_id": doc_id})


def read_chunk_records(strategy: str, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    path = settings.chunks_dir(strategy) / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python ingest.py chunk --strategy {strategy}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(session: Session, strategy: str, activate: bool = True, settings: Settings | None = None) -> dict:
    """Build a fresh collection for ``strategy`` and verify it."""
    settings = settings or get_settings()
    records = read_chunk_records(strategy, settings)
    name = settings.collection_name(strategy)
    store = ChromaVectorStore(name, path=settings.chroma_db_path)
    store.drop()
    store.upsert(records)

    expected = {r["id"] for r in records}
    actual = store.ids()
    orphans, missing = actual - expected, expected - actual
    if orphans or missing:
        raise RuntimeError(f"Index verification failed: {len(orphans)} orphans, {len(missing)} missing")

    over_limit = None
    count_tokens = getattr(store.embedder, "count_tokens", None)
    if count_tokens and store.embedder.max_tokens:
        over_limit = sum(count_tokens(chunk_embedding_text(r)) > store.embedder.max_tokens for r in records)

    session.execute(delete(ChunkRow).where(ChunkRow.strategy == strategy))
    session.add_all(
        ChunkRow(
            chunk_id=r["id"],
            doc_id=r.get("doc_id", ""),
            version=r.get("version", settings.chunk_version),
            strategy=strategy,
            chunk_index=r.get("chunk_index", 0),
            token_count=r.get("token_count", 0),
            text_hash=hashlib.sha256(r["text"].encode()).hexdigest(),
            collection=name,
        )
        for r in records
    )
    registry = session.get(IndexRegistry, name) or IndexRegistry(collection=name)
    registry.strategy = strategy
    registry.version = settings.chunk_version
    registry.embedding_model = store.embedder.name
    registry.dimension = store.embedder.dimension
    registry.metric = "cosine"
    registry.normalized = store.embedder.normalized
    registry.chunk_count = len(records)
    session.add(registry)
    if activate:
        session.execute(
            update(IndexRegistry).where(IndexRegistry.strategy == strategy).values(is_active=False)
        )
        registry.is_active = True
    session.commit()

    report = {
        "collection": name,
        "chunks": len(records),
        "embedding_model": store.embedder.name,
        "dimension": store.embedder.dimension,
        "metric": "cosine",
        "normalized": store.embedder.normalized,
        "orphans": 0,
        "missing": 0,
        "chunks_over_embedding_limit": over_limit,
    }
    log.info("Index built: %s", report)
    return report
