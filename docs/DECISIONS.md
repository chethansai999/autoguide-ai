# Architecture Decision Records

Short records of decisions that are expensive to reverse. Add new ones at the bottom. Never edit an accepted decision; supersede it with a new one.

---

## ADR-001: One core service, thin adapters
**Status:** accepted · **Date:** 2026-10-10

**Context.** The UI, the HTTP API and the evaluation runner all need the same behaviour. The team also had two diverging pipelines in two repos.

**Decision.** All turn logic lives in `agent/orchestrator.ChatService.handle()`. Streamlit calls it in-process by default or over HTTP when `AUTOGUIDE_API_URL` is set. FastAPI and `evaluation/run_evaluation.py` call the same method. Baselines B0–B3 are `ChatOptions`, not separate code paths.

**Consequences.** What evaluation measures is exactly what users get. The adapters stay under about 150 lines each.

## ADR-002: Consolidate into the team repo; Soujanya's UI is the front end
**Status:** accepted · **Date:** 2026-10-10

**Context.** `Sai-Soujanya-Yendrathi/AutoGuide` held the polished Streamlit UI plus a second copy of the crawl, chunk and retrieval pipeline (`pipeline/`). `Bashanapellirahul/AutoGuide`, the repo named in the proposal, held the team's pipeline.

**Decision.** The team repo is the single source of truth. Her `app.py` and `assets/` move in unchanged, except for the backend call, clarification buttons and a session ID. Her `pipeline/` is not copied. Its good ideas are folded into work packages:
- the corpus-minimum gate is now `ingest.py status`;
- `## Heading` markers during extraction go to WP-02;
- golden-template columns go to `evaluation/golden_set.import_csv`.

**Consequences.** There is one interface, one test suite, and agents can't edit the wrong copy.

## ADR-003: Storage: SQLite/Postgres via SQLAlchemy, plus ChromaDB, plus in-memory BM25
**Status:** accepted · **Date:** 2026-10-10

**Decision.**
- Relational data goes through SQLAlchemy: SQLite by default, Postgres in Compose. It holds vehicles, sessions, messages, traces, feedback, the recall cache, the chunk and index registries, and evaluation runs.
- ChromaDB is kept for vectors, as the proposal says. There is one collection per (strategy, embedding model, version).
- BM25 is built in memory from `chunks.jsonl`.

**Alternatives considered.**
- pgvector for everything: fewer moving parts, but it changes the proposal's stated vector database.
- Elasticsearch/OpenSearch for BM25: overkill for fewer than 10k chunks.

## ADR-004: NHTSA-only sources; do not scrape autoguide.com
**Status:** superseded by ADR-008 · **Date:** 2026-10-10

**Context.** It was suggested to scrape `autoguide.com` to fill gaps in vehicle-model data.

**Decision.** No. There are three reasons.

1. Its robots.txt explicitly disallows AI crawlers by name (GPTBot, ClaudeBot, anthropic-ai, CCBot, PerplexityBot and others), which is a clear opt-out from this use. Crawling it with a generic user agent would be evasion.
2. The proposal commits to NHTSA as the single source, and the ethics report relies on official provenance.
3. The need is already met. `Safercar_data.csv` is NHTSA's 5-Star Safety Ratings dataset: 17,313 vehicle variants, 64 makes and model years 1990–2026. The NHTSA Recalls and vPIC APIs cover recalls and VIN decoding.

**Consequences.** Vehicle questions are answered from official structured data. Editorial content such as reviews is out of scope.

## ADR-005: Clarification = fine-tuned understanding + deterministic decision
**Status:** accepted · **Date:** 2026-10-10

**Context.** The requirement was an agent "fine-tuned to ask the questions for missing data".

**Decision.** The ML part is only the NLU (intent and slots). The decision about whether to ask, and what to ask for, is made by the resolver and clarifier against the vehicles table. The bot asks only when the answer depends on the missing information.

**Why not fine-tune an LLM end-to-end to ask questions?** Whether "2019 F-150" is ambiguous is a fact about the data (it has 10 variants), not about language. A generative model would learn to ask too often or too rarely, and nobody could audit the decision.

**Consequences.** The over-ask rate is measurable and fixable in code. Fine-tuning targets typos, aliases and context, where it can actually beat the rules.

## ADR-006: Provider-agnostic LLM with an offline mock
**Status:** accepted · **Date:** 2026-10-10

**Decision.** Every LLM call goes through `core/llm.py`. `LLM_PROVIDER=auto` picks OpenAI if a key exists, then Anthropic, then a deterministic **mock**. The mock produces extractive, cited answers so the whole system, including the UI and tests, works with no key.

**Consequences.** The team develops before the key arrives, and CI never needs secrets. Mock outputs must never be reported as results.

## ADR-007: Relevance labelled by source URL
**Status:** accepted · **Date:** 2026-10-10

**Decision.** Golden-set relevance uses `source_urls`, which is document-level. `relevant_chunk_ids` is optional and strategy-specific.

**Why.** Fixed and structure-aware chunking produce different chunk IDs, so chunk-level labels can't compare B1 and B2 against B3.

## ADR-008: AutoGuide.com is the only scraped website; the NHTSA website is not crawled
**Status:** accepted (supersedes ADR-004) · **Date:** 2026-10-10 · **Decided by:** Chethan (project owner)

**Decision.** The document corpus for general safety questions comes from **www.autoguide.com** articles. The NHTSA website is **not** crawled. NHTSA *data* is still used, because downloading a published dataset or calling a public API isn't scraping a website:
- the 5-Star Ratings CSV (`data/structured/Safercar_data.csv`);
- the Recalls API.

**Conditions** (enforced in `ingestion/autoguide_crawler.py` and its tests):
1. **robots.txt.** It is read first and obeyed for our user agent, including wildcard rules. If it can't be read, nothing is crawled. The site blocks named AI crawlers (GPTBot, ClaudeBot and others). Our crawler identifies itself honestly as an academic project with a contact email and never imitates a browser or another bot.
2. **Rate.** At most 1 request per second. The crawl stops on 403, 429 or 503. Query-string, search, account, comment and shop pages are never fetched.
3. **Scope.** Only safety-related articles, chosen by keywords in `config/autoguide_crawl.json`.
4. **Use of content.** The site's terms restrict reproducing or republishing its content. So answers paraphrase, quote no more than 25 consecutive words (`trim_long_quotes`), and cite and link the article. Raw pages stay in `data/raw/` and are not committed.

**Open item.** The approved proposal names NHTSA as the single source. The PM should confirm the change of source with the instructor and record it in `reports/00_proposal.md` and `reports/01_ingestion.md`.

## ADR-009: Team-supplied reference PDFs (IIHS crash-test protocols) join the corpus
**Status:** accepted · **Date:** 2026-10-10 · **Decided by:** Chethan (project owner)

**Context.** The project owner supplied five IIHS documents: the small overlap, moderate overlap 2.0 and side impact 2.0 crash test protocols, and the moderate overlap 2.0 and side impact 2.0 rating guidelines. They explain how frontal and side crash tests are run and rated, which the NHTSA ratings data and AutoGuide articles cover only briefly.

**Decision.** Add a "reference document" source for PDFs that the team was given. Nothing is scraped. `python ingest.py add-docs FILE.pdf ...` copies them to `data/raw/reference/` and extracts them to `data/processed/reference/` (`ingestion/reference_docs.py`). Chunking and indexing treat them like any other document. Citations link to the publisher's page that lists the documents (IIHS: test protocols and technical information).

**Conditions.**
1. Raw PDFs stay in `data/raw/` and are never committed (each is "© IIHS, all rights reserved").
2. Answers paraphrase them, quote at most 25 consecutive words, and cite them, as with AutoGuide (ADR-008).
3. Answers name the organisation behind each fact. IIHS rates tests good/acceptable/marginal/poor, while NHTSA uses 1–5 stars. The vehicle-specific ratings in our data are NHTSA's (system prompt rule 12).
4. These five PDFs meet the proposal's "5 PDFs" corpus minimum (`ingest.py status`, `reference_docs`).

**Open item.** The approved proposal names NHTSA as the source, and IIHS is a separate, non-government organisation. The PM should confirm with the instructor and record the change in `reports/00_proposal.md` and `reports/01_ingestion.md`.

## ADR-010: Curated Wikipedia and FuelEconomy.gov pages for vehicle information
**Status:** accepted (expands ADR-008; retains ADR-009) · **Date:** 2026-10-10 · **Decided by:** project owner through the attached website proposal and implementation request

**Decision.** Add English Wikipedia vehicle articles for history/overview and FuelEconomy.gov HTML for efficiency and specifications. Recalls continue through the NHTSA API; ratings continue through the CSV. No NHTSA or IIHS HTML crawler is enabled. Existing AutoGuide collection and supplied reference PDFs remain supported.

**Collection.** `config/vehicle_web_sources.json` is a bounded reviewed manifest, not a discovery crawler. Vehicle pages require make/model metadata; EPA vehicle pages also require year. Only HTTPS on the exact two new hosts is accepted. Wikipedia namespaces, arbitrary queries and unsafe URLs are rejected. FuelEconomy vehicle URLs are limited to `Find.do?action=sbs&id=<numeric id>`. Check each ID and variant before adding it; no example ID is claimed as a real car in the default manifest.

Read robots.txt before collection, fail closed if it is unreadable/disallows the page, honour crawl delay at one request per second or slower, and stop on 403/429/503 without retries or alternate identities. Never use proxies or follow unapproved redirects. The honest academic bot User-Agent requires a real team contact. Dry-run previews are fully offline and do not assert crawling permission. Live collection remains skipped at the owner's request; FuelEconomy live HTML/robots availability has not been verified in this environment.

**Grounding.** Raw bytes, URL, capture date, SHA-256 and reviewed vehicle scope survive extraction and indexing. Filter both retrieval paths before top-k. Missing year/variant metadata cannot match a requested year/variant; never fall back to unrelated vehicles. Numerical specifications ask for year; yearless model articles answer general history only. An EPA page is evidence for its listed version, not every version of that model.

**Attribution.** Preserve Wikipedia source, contributor history, CC BY-SA 4.0 link and modification notice in documents, chunks, evidence, API citations and UI. See [Wikimedia Terms §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) and [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy). No images are extracted. Answers retain citation validation, the 25-word quotation limit, prediction refusal and injection guards.

**Consequences.** This is additional retrieval knowledge, not a generation-model training run. The new `vehicle_info` intent is included in optional synthetic NLU generation; old classifiers need retraining before claiming support for it. Integration ownership is recorded in WP-11. Instructor alignment and real DEV evaluation remain project deliverables; no mock output is reported as model performance.
