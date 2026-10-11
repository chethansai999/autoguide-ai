# AGENTS.md: operating manual for coding agents (and humans)

AutoGuide AI is a RAG chatbot for vehicle information and safety. It uses NHTSA ratings/recalls and approved document sources (ADR-009/010), with clarification for vehicle-specific questions. It is an MSAI 633 team project with 7 members. Read `docs/ARCHITECTURE.md` before changing anything structural.

## Start here
1. Read `docs/ARCHITECTURE.md` §2 (what exists) and §6 (who owns what).
2. Pick **one** work package in `docs/work_packages/`. Stay inside its "Files you own" list.
3. Run `python -m pytest -q` before you change anything. It must be green, and it runs fully offline.
4. Make the WP's acceptance tests pass, update the report it names, and open a small PR.

## Commands
```bash
pip install -r requirements.txt            # runtime
cp .env.example .env                       # never commit .env
python -m pytest -q                        # offline suite: mock LLM, hashing embedder, no network
python ingest.py status                    # pipeline state + corpus minimums
python ingest.py load-safercar             # vehicles table from data/structured/Safercar_data.csv
python ingest.py chunk --strategy fixed    # or structure / both
python ingest.py index --strategy fixed
uvicorn api.main:app --reload --port 8000  # API, docs at /docs
streamlit run app.py                       # UI
python -m finetune.synth_data && python -m finetune.eval_nlu --backend rules
```
If `make` is installed, the same commands exist as targets (see the `Makefile`).

## Hard rules
- **Secrets.** API keys live only in `.env`. Never write a key into code, tests, docs, notebooks, commit messages or issues. If you see a key anywhere in the repo, stop and tell a human.
- **Sources (ADR-010, owner-requested expansion).** AutoGuide articles use `ingestion/autoguide_crawler.py`. Reviewed English Wikipedia vehicle articles and FuelEconomy.gov guides/vehicle pages use `ingestion/web_sources.py` and `config/vehicle_web_sources.json`. Do **not** crawl NHTSA HTML, IIHS or other sites. Team-supplied reference PDFs are imported locally (ADR-009). NHTSA data is used only through the ratings CSV and the public Recalls/vPIC APIs. Retain Wikipedia attribution/license metadata through citations and UI. `crawl-web --dry-run` is an offline manifest preview, not robots permission verification.
- **Crawling.** Respect robots.txt, keep the delay at 1 request per second or slower, and use the honest User-Agent with the team email. On 403, 429 or 503, **stop**. Never rotate user agents, imitate browsers or other bots, or use proxies or headless browsers to get around a block.
- **Contracts.** `core/schemas.py` is the interface between teammates. Adding an optional field is fine. Renaming or removing a field, or changing a type, needs the tests in `tests/test_contracts.py` updated and a note in the PR.
- **Grounding.** Every factual answer cites `[n]` evidence. Never weaken `validate_citations`, `trim_long_quotes` (25-word limit), the prediction refusal or the injection guards to make a test pass.
- **Evaluation hygiene.** Tune only on `split == "dev"`. Never read, print or tune on `test` questions. The test split runs once (`--confirm-test`).
- **No network in unit tests.** Use the fixtures in `tests/conftest.py` (mock LLM, hashing embedder, fixture CSV and docs). A test that needs a model download or an API key must be marked and skipped by default.
- **Mock output is not a result.** Never put numbers produced with `LLM_PROVIDER=mock` or `EMBEDDING_BACKEND=hashing` into a report.

## Repo map
```
core/        config.py (all settings) · schemas.py (contracts) · llm.py (providers + mock) · observability.py
db/          models.py (SQL schema) · session.py
ingestion/   crawler · extractor · cleaner · api_collector · api_normalizer       (WP-01/02)
rag/         chunking · embeddings · vector_store · retrieval · reranker · evidence · generation · prompts · chatbot (facade)
structured/  safercar.py (CSV -> vehicles) · vocab.py · resolver.py · recalls.py   (WP-08)
agent/       nlu.py · lexicon.py · clarifier.py · guards.py · orchestrator.py      (WP-06/07)
finetune/    synth_data.py · train_nlu.py · openai_finetune.py · eval_nlu.py       (WP-07)
evaluation/  golden_set.py · metrics.py · run_evaluation.py · data/                (WP-09)
api/         main.py (FastAPI) · client.py (UI adapter)                            (WP-10)
app.py       Streamlit UI (design: Soujanya) · assets/
ingest.py    pipeline CLI
reports/     00-07 markdown reports (deliverables)
docs/        ARCHITECTURE.md · DECISIONS.md · work_packages/
```

## Conventions
- Python 3.11. Type hints at module boundaries. Pydantic models for anything that crosses a module.
- Use `logging` in new code, not `print`. Existing scripts may keep their prints.
- Deterministic by default: fixed seeds, sorted outputs, stable IDs.
- A new setting goes in `core/config.py` **and** `.env.example`, with a comment.
- A new SQL column must stay portable across SQLite and Postgres: use generic SQLAlchemy types and `JSON`.
- Branch names: `feature/wp-XX-short-name`. One WP per PR. Each PR needs one teammate's approval (proposal §9). End the PR description with which acceptance tests now pass.

## Definition of done (every PR)
- [ ] `python -m pytest -q` passes offline.
- [ ] The WP's acceptance criteria are met, and any `xfail` markers for them are removed.
- [ ] The relevant `reports/0X_*.md` is updated with **measured** numbers and how to reproduce them.
- [ ] No secrets, no new data sources, and no changes outside the WP's files without a note.
