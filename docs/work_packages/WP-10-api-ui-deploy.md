# WP-10: API, UI integration, Docker, README

**Owners:** Asad (PM) + Soujanya (UI) · **Phase:** P0 (run the demo), P3 (polish)

## Already working
- FastAPI (`api/main.py`) with `/ready`, `/v1/chat`, sessions, feedback, the vehicle resolver and retrieval endpoints.
- Soujanya's UI wired through `api/client.py`: clarification buttons, session IDs, a demo banner while the provider is mock, and an intent/latency line in the evidence inspector.
- Docker Compose with Postgres, the API and the UI.
- A headless UI test in `tests/test_ui.py`.

## Tasks
1. **Demo checklist.** Before the demo, `GET /ready` must show vehicles loaded, an active index, `llm_provider: openai` and the intended NLU backend. Rehearse three scripted conversations:
   - a concept question;
   - a vehicle question → clarification → answer;
   - a refusal.
2. Add a thumbs up/down control in the UI that calls `/v1/feedback`, or writes directly in-process.
3. Add Alembic only if the team runs Postgres with real data. Otherwise `init_db()` is enough.
4. Write the README: reproduction commands, known limitations, and the AI tools used (proposal §9).
5. Branch protection on `main` with one required review.

## Files you own
`api/*`, `app.py` (with Soujanya), `Dockerfile`, `docker-compose.yml`, `Makefile`, `README.md`

## Acceptance
- `docker compose up --build` followed by `docker compose run --rm api python ingest.py all` gives a working UI on :8501.
- `tests/test_api.py` and `tests/test_ui.py` pass.

## Agent prompt
> Read AGENTS.md and docs/work_packages/WP-10-api-ui-deploy.md. Add thumbs up/down buttons under each assistant message in app.py that record feedback (POST /v1/feedback when AUTOGUIDE_API_URL is set, otherwise write a Feedback row in-process via db.session). Keep the existing styling. Extend tests/test_ui.py to click a thumbs-up. Run python -m pytest -q.
