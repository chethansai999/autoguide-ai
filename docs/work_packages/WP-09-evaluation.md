# WP-09: Golden set, B0–B3, statistics, failure analysis

**Owner:** Sai Krishna (Evaluation Lead) · **Phase:** P1 · **Reports:** `reports/03_golden_set.md`, `reports/06_final_performance.md`

⚠️ `reports/03_golden_set.md` currently contains `.gitignore` text by mistake. Replace it.

## Already working
- `evaluation/golden_set.py`: schema (quotes of 25 words or fewer, notes required for unanswerable questions), mix validation, the stratified 60/40 split, and CSV import of Soujanya's template.
- `evaluation/metrics.py`: Recall@k, MRR, nDCG@10, bootstrap CI, exact McNemar, refusal rates.
- `evaluation/run_evaluation.py`: B0–B3 through the real `ChatService`, results stored in files and the DB, and a test-split lock.

## Tasks
1. Write 40 human-verified questions into `evaluation/data/golden_set.jsonl`: 15 single-hop, 6 multi-hop, 4 comparative, 4 temporal, 8 unanswerable, 3 ambiguous/adversarial. Label relevance with `source_urls`; see `golden_set.example.jsonl` for the format.
   - Include about 5 **vehicle-specific** questions, for example "Does the 2022 Honda Civic have blind spot detection?". Their expected source is the ratings record.
   - Include 2 **clarification** cases, for example "Is my F-150 safe?", with `expected_status: needs_clarification`.
2. Run `python -m evaluation.golden_set validate`, then `split`. Commit the split before any tuning.
3. Run `python -m evaluation.run_evaluation --baseline all --split dev` repeatedly while others tune.
4. Freeze the configuration. Then run the test split once: `--split test --confirm-test`.
5. Grade correctness by hand (or use `--judge` and spot-check 100% of disagreements). Run McNemar for B3 vs B1 and B3 vs B0. Categorise 15 failures.

## Files you own
`evaluation/*`, `reports/03_golden_set.md`, `reports/06_final_performance.md`

## Acceptance
- `validate` reports `ok: true`.
- The final report has every metric from proposal §7 with CIs and p-values, and the failure taxonomy.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-09-evaluation.md. Add evaluation/report.py that reads data/eval_runs/*/summary.json for a given split and writes a markdown comparison table (B0-B3 x metrics, with bootstrap CIs) to stdout. Add a unit test with two fake summaries. Never run the test split. Run python -m pytest -q.
