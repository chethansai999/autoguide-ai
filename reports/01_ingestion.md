# 01 — Data Ingestion Report

## Data Source

National Highway Traffic Safety Administration (NHTSA)

Website: https://www.nhtsa.gov/

## Initial Crawling Attempt

Date: October 10, 2026

The team implemented a Python crawler using Requests
and BeautifulSoup.

The crawler includes:
- robots.txt verification
- Request rate limiting
- Descriptive User-Agent
- Raw document storage
- SHA-256 content hashing
- Duplicate detection

## Access Restriction

During initial testing, NHTSA returned HTTP 403 Forbidden
for both the homepage and robots.txt.

The crawler stopped because crawling permissions could
not be verified.

## Resolution Status

The team is investigating official NHTSA APIs and
other authorized public data sources.

No restricted pages were collected.
## Reference documents (ADR-009)

The project owner supplied five IIHS PDFs, which `python ingest.py add-docs FILE.pdf ...` extracts with `ingestion/reference_docs.py`. Nothing is scraped. Extraction:
- reads the title, version and date from the cover;
- reads section headings from the contents page, then drops that page;
- removes running headers and footers;
- marks headings as `##`/`###` in contents order;
- re-flows wrapped lines into paragraphs.

| Document | Version | Words | Sections |
|---|---|---|---|
| Small Overlap Frontal Crashworthiness Evaluation: Crash Test Protocol | VIII, May 2025 | 10,610 | 26 |
| Moderate Overlap Frontal Crashworthiness Evaluation 2.0: Crash Test Protocol | II, May 2025 | 6,828 | 25 |
| Moderate Overlap 2.0 Rating Guidelines | III, February 2026 | 8,481 | 33 |
| Side Impact Crashworthiness Evaluation 2.0: Crash Test Protocol | III, May 2025 | 6,339 | 21 |
| Side Impact Rating Guidelines 2.0 | IV, April 2024 | 4,472 | 12 |

Known limitation: tables come out as one line of cell text, not as rows. Raw PDFs are not committed.

## Curated vehicle websites (WP-11 / ADR-010, October 10, 2026)

The owner requested the attached proposal's Wikipedia and FuelEconomy.gov sources. The implementation adds bounded manifest collection, robots/contact/rate/stop safeguards, raw capture metadata, source-specific offline HTML extraction, inline headings/tables and Wikipedia attribution. Existing AutoGuide collection, NHTSA API/CSV paths and supplied reference-PDF ingestion are retained.

**Actual local artifacts, not model scores:** `python ingest.py status` reported zero raw snapshots and zero processed documents for both new websites. The concurrent reference-PDF import has five processed reference documents and an active fixed index with 114 chunks; the vehicle table has 17,313 rows. `python ingest.py crawl-web --dry-run` validated five seed entries (three Wikipedia model articles, two general FuelEconomy guides) without network requests. No real EPA vehicle ID or numerical specification was added. These counts describe files/records, not mock model performance.

**Reproduction:** run `python -m pytest -q`, then `python ingest.py crawl-web --dry-run` and `python ingest.py status`. `tests/test_web_sources.py` uses synthetic HTML and fake HTTP responses to exercise URL allowlisting, robots denial, request delay, 403/429/503 stopping, redirect rejection, SHA-256 provenance, idempotence, headings/tables, exact vehicle/year filtering and the chat/API citation path. `tests/test_ui.py` verifies new source links and Wikipedia credits; `tests/test_contracts.py` locks the additive attribution fields and intent/source enum values. The tests do not measure factual answer quality.

**Remaining validation:** live collection was explicitly skipped by the owner. FuelEconomy.gov robots and live HTML availability were not verified here, so fixture selectors must be checked against a permitted real capture before declaring live ingestion successful. Add reviewed EPA vehicle pages with exact year/version metadata, collect when contact is configured, extract, rebuild the index and restart the API/UI. Run real-provider DEV evaluation afterward; no held-out evaluation questions were inspected or run for this feature. The existing structure-chunking expected failures remain WP-03 work.

**Ownership exception:** WP-11 explicitly owns the integration across ingestion, schemas, chunking, retrieval, NLU, prompts and adapters. No source field was renamed/removed, no SQL column was added, and the guards/quote/citation limits were retained. See the package file for the exact ownership list. Concurrent ADR-009 PDF changes were preserved.

**Publication regression check (October 10, 2026):** an offline test intermittently missed the only eligible vehicle document during approximate ID-filtered dense retrieval. Vehicle-scoped search now computes cosine similarity over eligible stored vectors, with stable ID ordering for ties. Unscoped search continues using Chroma's approximate index. Reproduce with `python -m pytest -q tests/test_web_sources.py` and the full offline suite. These are regression checks, not answer-quality measurements. At the owner's request, `.gitignore` also excludes filesystem `.fuse_hidden*` artifacts from the new public repository; this is the publication-specific ownership exception.
