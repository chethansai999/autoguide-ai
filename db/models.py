"""SQLAlchemy ORM models.

Portable across SQLite (default, zero setup) and Postgres (docker-compose).
Schema changes: for this project ``init_db()`` (create_all) is sufficient;
if you need to alter an existing Postgres table, add Alembic (WP-10).

Tables
    vehicles        NHTSA 5-Star ratings + safety features (Safercar CSV)
    documents       registry of extracted source documents
    chunks          registry of indexed chunks (orphan checks, re-chunking)
    index_registry  Chroma collections built + which one is active
    chat_sessions   conversation state incl. carried-over vehicle slots
    messages        user / assistant turns
    turn_traces     retrieval + LLM trace per assistant turn
    feedback        thumbs up / down
    recall_cache    cached NHTSA recalls API responses
    eval_runs       one row per evaluation run (baseline + config)
    eval_results    one row per (run, question)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# ----------------------------------------------------------------- vehicles

class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # CSV row number
    make: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(160))
    model_family: Mapped[str] = mapped_column(String(80), index=True)
    model_year: Mapped[int] = mapped_column(Integer, index=True)
    body_style: Mapped[str | None] = mapped_column(String(80))
    drive_train: Mapped[str | None] = mapped_column(String(32))
    vehicle_type: Mapped[str | None] = mapped_column(String(64))
    vehicle_class: Mapped[str | None] = mapped_column(String(64))
    production_release: Mapped[int | None] = mapped_column(Integer)
    variant_label: Mapped[str] = mapped_column(String(240))

    overall_stars: Mapped[float | None] = mapped_column(Float)
    frontal_stars: Mapped[float | None] = mapped_column(Float)
    side_stars: Mapped[float | None] = mapped_column(Float)
    rollover_stars: Mapped[float | None] = mapped_column(Float)
    side_pole_stars: Mapped[float | None] = mapped_column(Float)
    rollover_risk_pct: Mapped[float | None] = mapped_column(Float)

    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evaluations: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    safety_concerns: Mapped[list[Any]] = mapped_column(JSON, default=list)
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_vehicles_lookup", "make", "model_family", "model_year"),)


# ------------------------------------------------------------------- corpus

class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_url: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40))
    page_type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str | None] = mapped_column(String(64))
    crawl_timestamp: Mapped[str | None] = mapped_column(String(40))
    effective_date: Mapped[str | None] = mapped_column(String(40))
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(16))
    strategy: Mapped[str] = mapped_column(String(16), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    text_hash: Mapped[str] = mapped_column(String(64))
    collection: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class IndexRegistry(Base):
    __tablename__ = "index_registry"

    collection: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(16))
    version: Mapped[str] = mapped_column(String(16))
    embedding_model: Mapped[str] = mapped_column(String(160))
    dimension: Mapped[int] = mapped_column(Integer)
    metric: Mapped[str] = mapped_column(String(16), default="cosine")
    normalized: Mapped[bool] = mapped_column(Boolean, default=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# --------------------------------------------------------------------- chat

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    slots: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pending_intent: Mapped[str | None] = mapped_column(String(32))
    pending_features: Mapped[list[Any]] = mapped_column(JSON, default=list)
    pending_question: Mapped[str | None] = mapped_column(Text)
    # clarification button label (lower-cased) -> slot values / vehicle_id
    pending_options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # last answered intent/features, for follow-ups like "what about the 2020 one?"
    last_intent: Mapped[str | None] = mapped_column(String(32))
    last_features: Mapped[list[Any]] = mapped_column(JSON, default=list)
    clarification_turns: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))          # user | assistant
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TurnTrace(Base):
    __tablename__ = "turn_traces"

    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), primary_key=True)
    intent: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    trace: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RecallCache(Base):
    __tablename__ = "recall_cache"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)  # MAKE|MODEL|YEAR
    source_url: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# --------------------------------------------------------------- evaluation

class EvalRun(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    baseline: Mapped[str] = mapped_column(String(16))
    split: Mapped[str] = mapped_column(String(8))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.run_id"), index=True)
    question_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    answer: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
