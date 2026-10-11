# WP-04: Indexing, re-chunk swap, orphan checks

**Owner:** Rahul (Chunking & Database) · **Phase:** P0 · **Report:** `reports/02_chunking.md` (index section)

## Already working
`python ingest.py index --strategy X` builds a fresh collection named `nhtsa_<strategy>_<model>_<version>` and verifies it has 0 orphans and 0 missing IDs. It writes the `chunks` and `index_registry` rows (model, 384 dimensions, cosine, normalised) and marks the collection active.

## Tasks
1. Add `delete_document(doc_id)` coverage. When a document is removed or changed, delete its chunks from Chroma and from the `chunks` table, then re-verify that no orphans remain. Add a test using the hashing embedder.
2. Write the re-chunk procedure into the report, with real output: bump `CHUNK_VERSION`, chunk, index, evaluate old vs new on dev, then activate.
3. Add metadata filtering by `page_type` in `ChromaVectorStore` (proposal §4) and a test for it.
4. Make `rag/retrieval.py` read the **active** collection from `index_registry`, not only the derived name. Coordinate with WP-05.

## Files you own
`rag/vector_store.py`, `rag/embeddings.py`, `tests/test_vector_store.py`

## Acceptance
- Tests cover delete → no orphans, the page-type filter, and activation switching.
- The report shows the index registry for both strategies.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-04-indexing.md. In rag/vector_store.py add document deletion that also removes ChunkRow rows and verifies no orphans, plus a page_type metadata filter for queries. Write tests/test_vector_store.py using EMBEDDING_BACKEND=hashing fixtures from tests/conftest.py. Run python -m pytest -q.
