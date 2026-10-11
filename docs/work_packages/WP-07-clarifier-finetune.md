# WP-07: Clarification agent and NLU fine-tuning

**Owner:** Chethan / open · **Phase:** P2 (after the P0 demo path works) · **Report:** add a "Clarification agent" section to `reports/05_generation.md`

## Already working
- Rule-based NLU, resolver and clarifier, end to end with UI buttons.
- Synthetic data generator, DistilBERT trainer, OpenAI fine-tune script, and NLU evaluator.
- Baseline (rules) on the synthetic test set: intent 98.8%, make 83.8%, model 93.3%, decision accuracy 95.9%.
- Baseline on the realistic seeds (30 queries): intent 96.7%, make 93.3%, decision accuracy 96.7%.
- `train_nlu.py` → `HFNLU` → `eval_nlu.py` was verified end to end with a tiny local model. Real `distilbert-base-uncased` weights download on your machine at the first run.

## How fine-tuning works here
1. `python ingest.py load-safercar` loads the vehicles.
2. `python -m finetune.synth_data --n 8000` generates labelled examples from the CSV, split so the test set has unseen models.
3. Train on a laptop CPU or Colab:
   - `pip install -r requirements-finetune.txt`
   - `python -m finetune.train_nlu --smoke` to check the setup
   - `python -m finetune.train_nlu` for the full run
4. `python -m finetune.eval_nlu --backend hf --set realistic` and the same with `--backend rules`, to compare.
5. If it wins, set `NLU_BACKEND=hf` in `.env`.

Optional, and paid: `python -m finetune.openai_finetune` is a dry run that prints a token estimate. Add `--yes` to start the job, then set `NLU_BACKEND=llm` and `NLU_LLM_MODEL=ft:...`.

## Tasks
1. **Grow `evaluation/data/nlu_realistic.jsonl` from 30 to 100 or more** real phrasings, written by teammates who have not read the templates. This is what makes the comparison credible.
2. Train `hf`, then evaluate rules vs hf (and optionally llm) on the synthetic and realistic sets. Report all metrics in a table, including the over-ask and under-ask rates.
3. Error analysis: categorise 15 NLU errors (typo, alias, common-word model, intent confusion).
4. Optional: LLM rephrasing of clarification questions, behind a setting. The decision must stay in `agent/clarifier.py`.

## Files you own
`agent/nlu.py`, `agent/lexicon.py`, `agent/clarifier.py`, `finetune/*`, `evaluation/data/nlu_realistic.jsonl`

## Acceptance
- The trained model is in `models/clarifier/`, which is git-ignored. Share it via Drive or the Hugging Face Hub and document the download step.
- A comparison table with measured numbers is in the report. Fine-tuning is adopted only if it beats rules on the realistic set.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-07-clarifier-finetune.md. Add 20 more realistic, hand-labelled queries to evaluation/data/nlu_realistic.jsonl covering typos, aliases (chevy, vw), short years ('19), lowercase models and follow-ups. Use the existing JSON shape. Run python -m finetune.eval_nlu --backend rules --set realistic and report the metrics. Run python -m pytest -q.
