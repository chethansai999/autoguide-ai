# WP-01: Collect the AutoGuide.com corpus

**Owner:** Vikas (Data Collection) · **Phase:** P0, critical path · **Report:** `reports/01_ingestion.md`

## Decision (ADR-008)
- **www.autoguide.com is the only website we scrape.**
- The NHTSA website is **not** crawled; it returned 403 anyway.
- NHTSA *data* is still used through the ratings CSV and the Recalls API, which are not scraping.

## Already built
- `ingestion/autoguide_crawler.py` finds article URLs through the sitemaps listed in robots.txt and keeps only safety-related ones (`config/autoguide_crawl.json`).
- It obeys robots.txt with wildcard support, fetches at most 1 page per second, uses an honest User-Agent with the team email, and stops on 403/429/503.
- It saves raw bytes plus URL, timestamp and SHA-256 for each page.
- `ingestion/autoguide_extractor.py` writes `Document` JSON with inline `## Heading` markers, tables and dates.
- Offline tests are in `tests/test_autoguide.py`.

## Tasks
1. Set `CRAWLER_CONTACT_EMAIL` in `.env`. Run `python ingest.py crawl --dry-run` and review the URL list. Tune `include_keywords` and `exclude_patterns` until it is safety-focused and gives 180 or more candidates.
2. Run `python ingest.py crawl --max-pages 180`, then `python ingest.py extract-autoguide`. Open 10 processed documents and check the extraction. Fix the selectors in `autoguide_extractor.py` (WP-02) if ads or navigation leak in.
3. Run `python ingest.py status` and record the **observed** counts in the report.
4. **Instructor sign-off (PM):**
   - the proposal names NHTSA as the single source;
   - AutoGuide has few or no PDFs, so ask whether the "5 PDFs" minimum is waived or can be met another way;
   - quote the reply in the report.
5. Record the AutoGuide robots.txt and terms-of-use review in `reports/07_ethics.md`:
   - AI crawlers are blocked by name, and our honest UA is allowed by the `*` rules;
   - the terms restrict republishing, so answers paraphrase, quote 25 words or fewer, and cite.

## Rules (non-negotiable)
Never bypass a block. That means no user-agent spoofing, no proxies, no headless browsers and no mirrors or caches. Don't crawl any other website.

## Files you own
`ingestion/autoguide_crawler.py`, `config/autoguide_crawl.json`, `reports/01_ingestion.md`

## Acceptance
- `python ingest.py status` shows at least 150 AutoGuide pages and at least 3 with tables. The PDF minimum is met, or the instructor's waiver is quoted.
- `tests/test_autoguide.py` passes.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-01-corpus-collection.md. Run `python ingest.py crawl --dry-run`, summarise the candidate URLs by section, and propose include_keywords/exclude_patterns changes in config/autoguide_crawl.json that keep the corpus safety-focused with 180+ candidates. Do not fetch pages beyond the dry run without asking. Run python -m pytest -q.
