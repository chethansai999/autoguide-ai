# WP-12: Local Apple Silicon generation

Owner-requested local inference, October 10, 2026. Hosted providers returned quota/credit errors; the owner requested keeping credentials locally and running the project on this machine. This additive provider uses an already downloaded model and preserves the existing grounded-generation contract.

## Files you own

- `core/local_llm.py`, `core/llm.py`, `core/config.py`, `.env.example`
- `api/readiness.py`, `app.py`
- `rag/generation.py` (preserve unknown structured feature availability)
- `requirements-local.txt`, `tests/test_local_llm.py`, `tests/test_ui.py`
- `README.md`, `docs/ARCHITECTURE.md`, `docs/work_packages/README.md`, this package, `reports/05_generation.md`

## Acceptance

1. `LLM_PROVIDER=local` selects an optional MLX provider. Other provider selection and hosted authentication behavior remain unchanged.
2. Load a local, complete model directory lazily. Missing artifacts fail without downloads; importing the application, readiness and offline tests must not load MLX or model weights.
3. Reuse one model per path within a process and serialize model access. The UI uses the running API over HTTP to avoid loading a second copy.
4. Preserve system instructions, history, sampler configuration, token/latency traces, citations, quote limits and refusal/injection guards. Reject prompts exceeding the configured context budget rather than dropping evidence.
5. Run the complete offline suite with simulated MLX modules. Record actual local model smoke checks separately from formal answer-quality evaluation. Do not read held-out evaluation questions.
6. Credentials stay in ignored `.env`; converted weights stay in ignored `data/models/`. Publish code only to the owner-requested `chethansai999/autoguide-ai` repository.

MLX on Apple Silicon is optional. Quantization changes weight precision; it is not fine-tuning or training on the vehicle corpus. `/ready` checks files and runtime availability, not successful generation or model quality.

Real local smoke testing exposed a generated claim that an unreported backup camera was absent. The structured-data path now renders requested feature fields directly when any requested availability is unknown and explains the limitation, with the same citation/quotation validators. This applies to every provider and is traced as `structured-data`, not as a model call or an outage fallback. Explicit `not available` values remain distinct from unknown values. This is a WP-12 generation ownership exception justified by the observed local integration failure.
