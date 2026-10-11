# Work packages

Each work package (WP) is a self-contained brief that a teammate, or a coding agent working for one, can pick up. To hand one to an agent, copy the **Agent prompt** block at the bottom of the WP.

| WP | Title | Owner (proposal role) | Phase | Depends on | Status |
|---|---|---|---|---|---|
| [WP-01](WP-01-corpus-collection.md) | Collect the AutoGuide.com corpus | Vikas (Data Collection) | P0 | — | 🟡 crawler built; needs a real crawl run + instructor sign-off |
| [WP-02](WP-02-extraction.md) | Extraction, cleaning, tables, heading markers | Vishnu (Data Processing) | P0 | 01 | 🟡 partial |
| [WP-03](WP-03-structure-chunking.md) | Structure-aware chunking + comparison | Rahul (Chunking & DB) | P1 | 02 | 🟡 fixed done |
| [WP-04](WP-04-indexing.md) | Indexing, re-chunk swap, orphan checks | Rahul (Chunking & DB) | P0 | 03 | 🟢 scaffolded |
| [WP-05](WP-05-retrieval-tuning.md) | Retrieval, rerank, sufficiency, query rewriting | Chethan (Retrieval) | P0/P1 | 04 | 🟢 core done |
| [WP-06](WP-06-generation-guardrails.md) | Generation, citations, refusals, safeguards | Soujanya (Chatbot) | P0 | 05 | 🟢 core done |
| [WP-07](WP-07-clarifier-finetune.md) | Clarification agent + NLU fine-tuning | Chethan / open | P2 | 08 | 🟢 pipeline done; training not yet run |
| [WP-08](WP-08-vehicle-data-recalls.md) | Vehicle ratings data, resolver, recalls tool | open | P0 | — | 🟢 core done |
| [WP-09](WP-09-evaluation.md) | Golden set, B0–B3, statistics, failure analysis | Sai Krishna (Evaluation) | P1 | 05, 06 | 🟡 harness done; questions missing |
| [WP-10](WP-10-api-ui-deploy.md) | API, UI integration, Docker, README | Asad (PM) + Soujanya | P0/P3 | — | 🟢 scaffolded |
| [WP-11](WP-11-vehicle-web-sources.md) | Curated Wikipedia/FuelEconomy integration | Owner-requested cross-package feature | P0 | 02, 04, 05, 06, 07, 10 | 🟡 offline implementation verified; live collection skipped |

Status key: 🟢 working code exists and needs extending or tuning · 🟡 partly there · 🔴 blocked.
