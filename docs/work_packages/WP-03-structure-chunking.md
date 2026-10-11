# WP-03: Structure-aware chunking and the comparison

**Owner:** Rahul (Chunking & Database) · **Phase:** P1 (needed for B3) · **Report:** `reports/02_chunking.md`

## Goal
Implement `StructureAwareChunker` in `rag/chunking.py` and compare it with fixed chunking.

## Spec
- Split at `## Heading` boundaries from WP-02. Never merge text from two sections into one chunk.
- `heading_path` holds the heading hierarchy, for example `["Air Bags", "Frontal Air Bags"]`.
- Sections longer than `FIXED_CHUNK_TOKENS` are sub-split with `FIXED_CHUNK_OVERLAP`.
- Keep a table intact in one chunk if it fits.
- IDs are `doc:version:structure:NNN` and deterministic.
- Consider prepending the heading path to the chunk text before embedding. Measure the effect.

## Tasks
1. Implement the chunker and remove the two `xfail` markers in `tests/test_chunking.py`.
2. Run `python ingest.py chunk --strategy both`. The stats land in `data/chunks/{fixed,structure}_stats.json`.
3. Report chunk counts and token-length distributions (mean, median, p95, max), plus a histogram.
4. ⚠️ Report `chunks_over_embedding_limit` from `ingest.py index`. MiniLM truncates at 256 word-pieces (ARCHITECTURE §10). Decide, and record why, whether to keep 500-token chunks.

## Files you own
`rag/chunking.py`, `tests/test_chunking.py`, `reports/02_chunking.md`

## Acceptance
- `tests/test_chunking.py` passes with no xfails.
- The report has a measured comparison table and a retrieval comparison on the dev split (via WP-09).

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-03-structure-chunking.md. Implement StructureAwareChunker.chunk() in rag/chunking.py per the spec, reusing FixedSizeChunker for oversized sections. Remove the xfail markers in tests/test_chunking.py and make those tests pass; add a test for a table staying intact. Do not change the Chunk contract. Run python -m pytest -q.
