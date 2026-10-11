"""Backend client for the Streamlit UI.

Keeps the UI's original contract (Soujanya's ``answer_question``) and adds
the fields the agent needs. In-process by default; set AUTOGUIDE_API_URL to
call the FastAPI service over HTTP instead (docker-compose does this).

    answer_question(question, history=None, session_id=None) -> {
        "answer": str,
        "sources": [{"ref": "[1]", "title": str, "url": str}],
        "trace":   [{"title", "text", "score", "rerank_score", ...}],
        "refused": bool,
        "demo":    bool,                 # True while LLM provider is "mock"
        # additions
        "status": "answered" | "needs_clarification" | "refused" | "error",
        "clarification": {"question": str, "options": [str]} | None,
        "disclaimer": str | None,
        "session_id": str,
        "message_id": str,
        "debug": {...full Trace...},
    }
"""

from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from core.schemas import ChatOptions, ChatRequest, ChatResponse, FeedbackRequest


@lru_cache(maxsize=1)
def _service():
    from agent.orchestrator import ChatService
    from core.observability import configure_logging

    configure_logging()  # so backend errors reach the terminal running Streamlit
    return ChatService()


def chat(message: str, session_id: str | None = None, options: ChatOptions | None = None) -> ChatResponse:
    request = ChatRequest(session_id=session_id, message=message, options=options or ChatOptions())
    url = get_settings().autoguide_api_url
    if url:
        import httpx

        response = httpx.post(f"{url.rstrip('/')}/v1/chat", json=request.model_dump(mode="json"), timeout=120)
        response.raise_for_status()
        return ChatResponse.model_validate(response.json())
    return _service().handle(request)


def readiness() -> dict:
    """Read the backend's configuration, including when it is degraded."""
    url = get_settings().autoguide_api_url
    if url:
        import httpx

        response = httpx.get(f"{url.rstrip('/')}/ready", timeout=10)
        if response.status_code != 503:
            response.raise_for_status()
        return response.json()
    from api.readiness import inspect_readiness

    return inspect_readiness(_service())


def submit_feedback(message_id: str, rating: int, comment: str | None = None) -> None:
    request = FeedbackRequest(message_id=message_id, rating=rating, comment=comment)
    url = get_settings().autoguide_api_url
    if url:
        import httpx

        response = httpx.post(f"{url.rstrip('/')}/v1/feedback", json=request.model_dump(mode="json"), timeout=10)
        response.raise_for_status()
        return
    from api.feedback import record_feedback

    record_feedback(_service().session_factory, request)


def to_ui_dict(response: ChatResponse) -> dict:
    trace = response.trace
    rows = []
    if trace:
        source_rows = trace.evidence or trace.candidates
        for row in source_rows:
            rows.append({
                **row,
                "score": row.get("retrieval_score"),
                "rerank_score": row.get("rerank_score"),
            })
    return {
        "answer": response.answer,
        "sources": [{"ref": f"[{c.ref}]", "title": c.title, "url": c.url,
                     "attribution": c.attribution} for c in response.citations],
        "trace": rows,
        "refused": response.status.value in ("refused", "error"),
        "demo": get_settings().resolved_llm_provider() == "mock",
        "status": response.status.value,
        "clarification": (
            {"question": response.clarification.question, "options": response.clarification.options}
            if response.clarification and response.clarification.should_ask else None
        ),
        "disclaimer": response.disclaimer,
        "session_id": response.session_id,
        "message_id": response.message_id,
        "debug": trace.model_dump(mode="json") if trace else None,
    }


def answer_question(question: str, history: list[dict] | None = None, session_id: str | None = None) -> dict:
    """UI entry point. ``history`` is accepted for compatibility; the backend
    keeps its own conversation state keyed by ``session_id``."""
    return to_ui_dict(chat(question, session_id=session_id))
