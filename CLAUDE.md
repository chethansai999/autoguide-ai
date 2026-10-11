@AGENTS.md

Claude Code specifics:
- Run `python -m pytest -q` after each change; the suite is offline and takes seconds.
- When asked to "do WP-XX", open `docs/work_packages/WP-XX-*.md` first and work only in its listed files.
- Keep `.env` out of every tool output, and never echo `OPENAI_API_KEY`.
