# WP-02: Extraction, cleaning, tables, heading markers

**Owner:** Vishnu (Data Processing) · **Phase:** P0 · **Report:** `reports/01_ingestion.md` (extraction section)

## Goal
Every processed document validates as `core.schemas.Document` and keeps its structure, so that WP-03 can chunk by section.

## Tasks
1. In `ingestion/extractor.py`, emit headings **inline** as `## Heading` lines in reading order. This is the approach from Soujanya's `pipeline/ingest.py`. Today the headings are a separate list and their positions are lost.
2. Tables: render each table as pipe-delimited rows inside the text where it appears, and keep `tables` as structured data. Use pdfplumber for PDF tables.
3. Fill `title`, `effective_date` (from page metadata or "Last updated" text, when present), `source_type` and `page_type`.
4. Remove navigation, footer and duplicate boilerplate, and record what was removed in the report.
5. Redact any PII found (emails, phone numbers of individuals).

## Files you own
`ingestion/extractor.py`, `ingestion/cleaner.py`, `ingestion/api_normalizer.py`

## Contract
Output files in `data/processed/*.json` must satisfy `Document.model_validate(json)`. Check with:
`python -c "from rag.chunking import iter_documents; from pathlib import Path; print(sum(1 for _ in iter_documents([Path('data/processed')])))"`

## Acceptance
- A new `tests/test_extraction.py` covers a tiny HTML page with an h2, a table and a nav, and asserts that the `## ` markers, the table rows and the nav removal all happen.
- At least 3 processed documents have `table_count > 0`.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-02-extraction.md. Change ingestion/extractor.py so headings are emitted inline as '## Heading' lines in reading order and tables are rendered as pipe-delimited rows in place, keeping the existing output keys. Add tests/test_extraction.py with an inline HTML fixture (no network). Run python -m pytest -q.
