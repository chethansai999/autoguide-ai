"""FastAPI app: a thin HTTP adapter over agent.orchestrator.ChatService.

    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000/docs      (interactive OpenAPI)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from agent.orchestrator import ChatService
from api.feedback import record_feedback
from api.readiness import inspect_readiness
from core.config import get_settings
from core.observability import configure_logging
from core.schemas import ChatRequest, ChatResponse, FeedbackRequest, ResolutionResult, VehicleRecord, VehicleSlots
from db.models import Message

_service: ChatService | None = None


def get_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    get_service()
    yield


app = FastAPI(title="AutoGuide AI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> dict:
    """503 for missing deployment artifacts; /health remains the liveness probe."""
    report = inspect_readiness(get_service())
    if not report["ready"]:
        response.status_code = 503
    return report


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_service().handle(request)


@app.get("/v1/sessions/{session_id}/messages")
def session_messages(session_id: str) -> list[dict]:
    with get_service().session_factory() as db:
        rows = db.scalars(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
        ).all()
        return [
            {"id": m.id, "role": m.role, "content": m.content, "status": m.status,
             "citations": m.citations, "created_at": m.created_at.isoformat()}
            for m in rows
        ]


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest) -> dict:
    try:
        record_feedback(get_service().session_factory, request)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return {"ok": True}


@app.get("/v1/vehicles/resolve", response_model=ResolutionResult)
def resolve_vehicle(
    make: str | None = None,
    model: str | None = None,
    year: int | None = Query(default=None, ge=1980, le=2100),
    body_style: str | None = None,
    drive_train: str | None = None,
) -> ResolutionResult:
    slots = VehicleSlots(make=make, model=model, model_year=year, body_style=body_style, drive_train=drive_train)
    return get_service().resolver.resolve(slots)


@app.get("/v1/vehicles/{vehicle_id}", response_model=VehicleRecord)
def get_vehicle(vehicle_id: int) -> VehicleRecord:
    record = get_service().resolver.get(vehicle_id)
    if record is None:
        raise HTTPException(404, "vehicle not found")
    return record


@app.post("/v1/retrieve")
def retrieve(query: str, mode: str = "hybrid", rerank: bool = True, strategy: str | None = None) -> dict:
    """Retrieval only (debugging and retrieval evaluation)."""
    result = get_service().evidence_checker.check_evidence(query, mode=mode, rerank=rerank, strategy=strategy)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "chunk"} for r in rows]  # noqa: E731
    return {**result, "evidence": strip(result["evidence"]), "candidates": strip(result["candidates"])}
