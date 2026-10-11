# WP-05: Retrieval, rerank, sufficiency, query rewriting

**Owner:** Chethan (Retrieval Engineer) · **Phase:** P0/P1 · **Report:** `reports/04_retrieval_reranking.md`

## Already working
BM25 + dense + RRF (k = 60) → top 20 → cross-encoder → top 5 (`rag/retrieval.py`). `retrieve_with_trace()` returns the candidates before rerank. `rag/evidence.py` decides sufficiency using:
- the rerank logit, if reranking;
- else the dense cosine similarity;
- else BM25 overlap.

## Tasks
1. **Tune thresholds on dev only.** Sweep `MIN_RERANK_SCORE` and `MIN_DENSE_SCORE`. Plot the correct-refusal rate against the over-refusal rate on the dev unanswerable and answerable questions. Pick the threshold and justify it.
2. Measure Recall@20 before rerank and Recall@5, MRR and nDCG@10 after rerank, for B1, B2 and B3, plus the rerank latency cost (p50/p95).
3. **Query rewriting** for follow-ups. Replace the `condense_question()` heuristic in `agent/orchestrator.py` with an LLM rewrite behind a setting. Evaluate it on the multi-hop questions.
4. Optional ablation: `BAAI/bge-small-en-v1.5` (384 dimensions, 512 tokens) vs MiniLM. Change one thing at a time.
5. Make retrieval use the active collection from `index_registry` (with WP-04).

## Files you own
`rag/retrieval.py`, `rag/reranker.py`, `rag/evidence.py`, `condense_question()` in `agent/orchestrator.py`

## Acceptance
- The report contains a B1/B2/B3 retrieval table with bootstrap CIs and a threshold curve.
- Thresholds are committed in `.env.example`, with the dev-set justification in the report.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-05-retrieval-tuning.md. Add a script evaluation/sweep_thresholds.py that runs EvidenceChecker over the golden DEV split for a grid of MIN_RERANK_SCORE values and writes correct-refusal and over-refusal rates to data/eval_runs/. Never load test-split questions. Add a unit test with a fake retriever. Run python -m pytest -q.
