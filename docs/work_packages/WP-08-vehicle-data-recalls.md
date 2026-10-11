# WP-08: Vehicle ratings data, resolver, recalls tool

**Owner:** open · **Phase:** P0 · **Report:** `reports/01_ingestion.md` (structured data section)

## Already working
- `structured/safercar.py` loads 17,313 rows and normalises the messy availability values ("S", "Std", "Standard" and so on → standard). It also derives model families (`F-150 SUPERCAB` → `F-150`) and renders vehicle evidence.
- `structured/resolver.py` returns resolved, ambiguous, missing or not found.
- `structured/recalls.py` uses the live API, then the DB cache, then the local API documents.

## Tasks
1. **Citation deep links.** `vehicle_citation_url()` currently returns `https://www.nhtsa.gov/ratings`. Verify NHTSA's per-vehicle page URL pattern by hand in a browser and implement it, with a test.
2. Data-quality report: the share of `unknown` values per feature by model year (pre-2011 rows have few ADAS fields), and the rows dropped or merged (for example the `ALFA` → `ALFA ROMEO` canonicalisation).
3. Recalls naming: the Recalls API sometimes names models differently from the ratings data (`SILVERADO` vs `SILVERADO 1500`). Measure the hit rate over 50 vehicles and improve `RecallsTool.lookup` name candidates.
4. Optional: VIN decode through vPIC (`vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json`) to fill make, model and year when the user gives a VIN.

## Files you own
`structured/*`, `tests/test_resolver.py`, `tests/test_safercar.py`

## Acceptance
- Tests cover the citation URL builder and recall name fallbacks (mocked HTTP).
- The report has the data-quality table.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-08-vehicle-data-recalls.md. Improve structured/recalls.py so that model-name candidates include the family, the full model names and the family with common suffixes removed. Add tests with an httpx.MockTransport (no network) covering hit, miss and API-down behaviour. Run python -m pytest -q.
