# WP-11: Curated vehicle web sources

Owner-requested integration, October 10, 2026. See ADR-010. This is one feature across ingestion, retrieval, chat and presentation; it explicitly spans the earlier package boundaries. Existing edits and the concurrent ADR-009 reference-PDF work must be preserved.

## Files you own

- `ingestion/web_sources.py`, `ingestion/web_extractor.py`, `config/vehicle_web_sources.json`, `ingest.py`
- `core/schemas.py` (additive enum members and optional attribution only)
- `rag/chunking.py`, `rag/vector_store.py`, `rag/retrieval.py`, `rag/filters.py`, `rag/evidence.py`, `rag/chatbot.py`, `rag/prompts.py`
- `agent/lexicon.py`, `agent/nlu.py`, `agent/orchestrator.py`, `finetune/synth_data.py`
- `api/client.py`, `api/readiness.py` (additive source counts only), `app.py`
- `tests/test_web_sources.py`, `tests/test_contracts.py`, `tests/test_retrieval.py`, `tests/test_ui.py`
- `AGENTS.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/work_packages/README.md`, this package, `reports/01_ingestion.md`, `reports/05_generation.md`

## Acceptance

1. `crawl-web --dry-run` validates/prints the manifest with no network or contact required. Actual collection requires contact, robots permission and one-second-or-slower pacing. Stop codes, unsafe URLs and cross-host redirects fail closed.
2. Saved HTML extracts offline into typed documents with headings/tables, source and capture provenance. Wikipedia attribution survives chunking, retrieval, chat citations, API session history and UI.
3. Both BM25 and dense retrieval filter before top-k; missing or mismatched vehicle/year/variant metadata cannot fall back to unrelated content.
4. History/specification/efficiency questions use `vehicle_info`; missing required details prompt clarification. Recalls, safety ratings and guards retain their existing routes.
5. Optional synthetic NLU generation includes `vehicle_info`, without contradictory MPG/horsepower out-of-scope labels. Do not run training or read held-out evaluation questions as part of implementation.
6. `python -m pytest -q` passes offline. The two existing WP-03 structure-chunking expected failures remain outside this feature. Update ingestion and generation reports with reproducible implementation checks; never report mock quality numbers.

## Remaining live validation

Local smoke testing also exposed overlapping UI/API conversations failing with SQLite's writer lock. The orchestrator now closes its read transaction before retrieval/generation and persists the completed turn in one short transaction; recall caching runs before that transaction. Regression checks in `tests/test_web_sources.py` cover new and existing conversations while a second caller saves, with no network or real model required. No schema or citation/grounding rules changed.

Collection was skipped by the owner. The starter manifest contains three yearless model articles and two general efficiency guides, not verified EPA vehicle IDs. A real contact and reviewed vehicle pages are required before claiming model-specific MPG coverage. Verify FuelEconomy selectors against permitted captured HTML, rebuild the index, restart the app, and evaluate real provider answers on DEV before claiming answer quality.

## Owner-requested publication preparation

On October 10, 2026 the owner explicitly requested publishing all current code to a new public GitHub repository named `autoguide-ai`. This is a separate repository snapshot; the existing team repository is retained. The publication includes concurrent changes at the owner's request and excludes local credentials, databases, chat history, model weights and generated corpora. The additional `.gitignore` change ignores filesystem `.fuse_hidden*` artifacts and is a publication-specific ownership exception.

A pre-publication regression run exposed intermittent misses in approximate vector search with a single eligible vehicle document. Scoped dense retrieval now ranks only eligible stored vectors with exact cosine similarity and stable ID ordering for ties. Unscoped retrieval retains Chroma's approximate search. Regression checks cover avoiding approximate scoped queries and avoiding all vector access for an empty scope; vehicle/year/source filtering and grounding requirements remain intact.
