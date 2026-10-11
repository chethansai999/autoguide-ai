# WP-06: Generation, citations, refusals, safeguards

**Owner:** Sai Soujanya (Chatbot Developer) · **Phase:** P0 · **Reports:** `reports/05_generation.md`, `reports/07_ethics.md`

## Already working
- `rag/generation.py` builds numbered, delimited evidence blocks, calls the LLM via `core/llm.py`, trims quotes longer than 25 words and validates citations.
- `agent/guards.py` blocks prompt injection in user input, strips instructions from retrieved text, and adds the disclaimer.
- The refusal texts are in `rag/prompts.py`.

## Tasks
1. With a real key, run the dev questions and review 20 answers by hand. Iterate `SYSTEM_PROMPT` (version it in the report) on:
   - citation placement;
   - concision;
   - per-variant statements for vehicle answers.
2. Build an **injection test set** of at least 10 documents or chunks with embedded instructions (in `tests/fixtures/`). Show the guard flags firing and the answers staying grounded. This is the ethics-report evidence for safeguards 1 and 2.
3. Decide what happens when citation validation fails. Today the bot refuses. Consider one regeneration attempt with a stricter instruction, and measure how often it rescues an answer.
4. UI polish in `app.py`: show the `disclaimer`, and show a refusal reason badge.

## Files you own
`rag/generation.py`, `rag/prompts.py`, `agent/guards.py`, `app.py` (UI)

## Acceptance
- `tests/test_guards.py` is extended with the injection set; all tests pass.
- `reports/05_generation.md` shows prompt versions with measured citation validity and correctness on dev.
- `reports/07_ethics.md` shows at least 2 safeguards demonstrated with trace excerpts.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-06-generation-guardrails.md. Add tests/fixtures/injection/*.json documents containing embedded instructions and a test that runs them through rag.generation.build_context and asserts the instructions are removed and flagged. Do not weaken any guard. Run python -m pytest -q.
