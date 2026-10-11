"""Chunking strategies (proposal section 5).

    fixed      ~500-token windows with 100-token overlap (baseline, B1/B2)
    structure  heading/section-aware chunks with heading_path (B3) - WP-03

Chunk IDs are deterministic: ``{doc_id}:{version}:{strategy}:{NNN}``
(e.g. ``NHTSA001:v1:fixed:001``), so re-running chunking on unchanged text
produces identical IDs and the index can be diffed / replaced safely.

    python ingest.py chunk --strategy fixed
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol

from core.config import get_settings
from core.schemas import Chunk, Document

log = logging.getLogger(__name__)

WORD = re.compile(r"\S+")


# ---------------------------------------------------------------- tokens

@lru_cache(maxsize=1)
def _encoding():
    if get_settings().embedding_backend == "hashing":
        # Offline fixtures must never download tiktoken's encoding table.
        return None
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as error:  # offline: tiktoken downloads its BPE file once
        log.warning("tiktoken unavailable (%s); using word-based token estimate.", error)
        return None


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return int(len(WORD.findall(text)) * 1.3)


def make_chunk_id(doc_id: str, version: str, strategy: str, index: int) -> str:
    return f"{doc_id}:{version}:{strategy}:{index:03d}"


# ---------------------------------------------------------------- strategies

class Chunker(Protocol):
    strategy: str

    def chunk(self, document: Document) -> list[Chunk]: ...


class FixedSizeChunker:
    """Token windows (tiktoken cl100k) with overlap. Falls back to words."""

    strategy = "fixed"

    def __init__(self, size: int | None = None, overlap: int | None = None, version: str | None = None):
        settings = get_settings()
        self.size = size or settings.fixed_chunk_tokens
        self.overlap = settings.fixed_chunk_overlap if overlap is None else overlap
        self.version = version or settings.chunk_version
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than chunk size")

    def _windows(self, text: str) -> list[str]:
        encoding = _encoding()
        step = self.size - self.overlap
        if encoding is not None:
            tokens = encoding.encode(text)
            windows = []
            for start in range(0, max(len(tokens), 1), step):
                windows.append(encoding.decode(tokens[start:start + self.size]))
                if start + self.size >= len(tokens):
                    break
            return windows
        # Fallback: word windows that slice the original text, so newlines
        # and '## Heading' markers survive.
        spans = [m.span() for m in WORD.finditer(text)]
        size_w, step_w = max(1, int(self.size / 1.3)), max(1, int(step / 1.3))
        windows = []
        for start in range(0, max(len(spans), 1), step_w):
            group = spans[start:start + size_w]
            if not group:
                break
            windows.append(text[group[0][0]:group[-1][1]])
            if start + size_w >= len(spans):
                break
        return windows

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []
        chunks = []
        for index, window in enumerate(self._windows(text), start=1):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, self.version, self.strategy, index),
                    doc_id=document.doc_id,
                    version=self.version,
                    strategy=self.strategy,
                    chunk_index=index,
                    text=window,
                    title=document.title,
                    source_url=document.source_url,
                    source_type=document.source_type,
                    page_type=document.page_type,
                    token_count=count_tokens(window),
                    vehicle=document.vehicle,
                    attribution=document.attribution,
                )
            )
        return chunks


class StructureAwareChunker:
    """Heading/section-aware chunking. OWNER: WP-03 (Chunking & Database).

    Acceptance (tests/test_chunking.py::test_structure_*):
      * splits at heading boundaries; never merges text across two H2 sections
      * every chunk carries heading_path, e.g. ["Air Bags", "Frontal Air Bags"]
      * sections longer than FIXED_CHUNK_TOKENS are sub-split with overlap
      * tables stay intact inside one chunk where they fit
      * deterministic chunk IDs with strategy "structure"
    Prerequisite: extraction must keep heading positions in the text
    (WP-02 emits '## Heading' markers like Soujanya's ingest did).
    """

    strategy = "structure"

    def __init__(self, version: str | None = None):
        self.version = version or get_settings().chunk_version

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("WP-03: implement structure-aware chunking")


def get_chunker(strategy: str) -> Chunker:
    if strategy == "fixed":
        return FixedSizeChunker()
    if strategy == "structure":
        return StructureAwareChunker()
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------- corpus

def iter_documents(dirs: Iterable[Path]) -> Iterable[Document]:
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.json")):
            try:
                yield Document.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception as error:
                log.warning("Skipping %s: %s", path, error)


def chunk_corpus(strategy: str, processed_dirs: list[Path] | None = None, out_dir: Path | None = None) -> dict:
    """Chunk every processed document; write chunks.jsonl + stats."""
    settings = get_settings()
    processed_dirs = processed_dirs or [
        settings.data_dir / "processed",
        settings.data_dir / "processed" / "api",
        settings.data_dir / "processed" / "autoguide",
        settings.data_dir / "processed" / "reference",   # team-supplied PDFs (ADR-009)
        settings.data_dir / "processed" / "wikipedia",
        settings.data_dir / "processed" / "fueleconomy",
    ]
    out_dir = out_dir or settings.chunks_dir(strategy)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunker = get_chunker(strategy)

    chunks: list[Chunk] = []
    documents = 0
    for document in iter_documents(processed_dirs):
        documents += 1
        chunks.extend(chunker.chunk(document))

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")

    lengths = [c.token_count for c in chunks] or [0]
    stats = {
        "strategy": strategy,
        "version": chunker.version,
        "documents": documents,
        "chunks": len(chunks),
        "tokens_mean": round(statistics.mean(lengths), 1),
        "tokens_median": statistics.median(lengths),
        "tokens_p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "tokens_max": max(lengths),
        "content_hash": hashlib.sha256("".join(c.chunk_id for c in chunks).encode()).hexdigest()[:16],
    }
    (out_dir.parent / f"{strategy}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Chunked %d documents into %d %s chunks", documents, len(chunks), strategy)
    return stats
