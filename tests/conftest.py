"""Test configuration: fully offline and deterministic.

Every test runs against a temporary data directory with:
  LLM_PROVIDER=mock, EMBEDDING_BACKEND=hashing, RERANKER_BACKEND=none,
  a 39-row slice of the real Safercar CSV, and 4 labelled fixture documents.
No network, no model downloads, no API keys.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))


def _reset_caches() -> None:
    from core import config, llm
    from db import session
    from rag import embeddings, reranker, retrieval

    config.get_settings.cache_clear()
    llm.get_llm.cache_clear()
    session.get_engine.cache_clear()
    embeddings.get_embedder.cache_clear()
    retrieval._retrievers.clear()
    reranker._reranker = None


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES / "processed", data / "processed")
    env = {
        "DATA_DIR": str(data),
        "DATABASE_URL": f"sqlite:///{(data / 'test.db').as_posix()}",
        "CHROMA_DB_PATH": str(data / "chroma"),
        "SAFERCAR_CSV": str(FIXTURES / "safercar_sample.csv"),
        "LLM_PROVIDER": "mock",
        "EMBEDDING_BACKEND": "hashing",
        "RERANKER_BACKEND": "none",
        "NLU_BACKEND": "rules",
        "CHUNK_STRATEGY": "fixed",
        "ENABLE_LIVE_RECALLS_API": "false",
        "MIN_DENSE_SCORE": "0.0",
        "AUTOGUIDE_API_URL": "",
    }
    os.environ.update(env)
    _reset_caches()
    return data


@pytest.fixture(scope="session")
def loaded(data_dir):
    """Vehicles loaded + fixed chunks indexed (session-wide, built once)."""
    from db.session import init_db, session_scope
    from rag.chunking import chunk_corpus
    from rag.vector_store import build_index
    from structured.safercar import load_safercar

    init_db()
    with session_scope() as db:
        load_safercar(db)
    chunk_corpus("fixed")
    with session_scope() as db:
        build_index(db, "fixed")
    return data_dir


@pytest.fixture(scope="session")
def vocab(loaded):
    from db.session import get_sessionmaker
    from structured.vocab import VehicleVocab

    with get_sessionmaker()() as db:
        return VehicleVocab.from_session(db)


@pytest.fixture(scope="session")
def service(loaded):
    from agent.orchestrator import ChatService

    return ChatService()
