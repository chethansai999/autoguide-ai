"""Data contracts shared by every component.

These models ARE the interfaces between work packages. Changing a field here
is a cross-team change: update the tests in ``tests/test_contracts.py`` and
note it in the PR description. Adding an optional field is safe.

Pipeline stage contracts
    ingestion  -> Document      (data/processed/**/*.json)
    chunking   -> Chunk         (data/chunks/<strategy>/chunks.jsonl)
    retrieval  -> RetrievedChunk
    agent      -> Evidence -> ChatResponse
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

class SourceType(str, Enum):
    WEB_PAGE = "nhtsa_web"
    PDF = "nhtsa_pdf"
    RECALLS_API = "nhtsa_recalls_api"
    RATINGS_DATASET = "nhtsa_ratings_dataset"
    AUTOGUIDE = "autoguide_article"
    REFERENCE = "reference_document"  # team-supplied PDFs, e.g. IIHS protocols (ADR-009)
    WIKIPEDIA = "wikipedia_article"  # curated vehicle articles (ADR-010)
    FUEL_ECONOMY = "fueleconomy_page"


class Document(BaseModel):
    """One extracted source document.

    Accepts both existing on-disk shapes: the crawler/extractor output
    (``doc_id``/``source_url``) and the API normalizer output (``id``/``url``).
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    doc_id: str = Field(validation_alias=AliasChoices("doc_id", "id", "document_id"))
    source_url: str = Field(validation_alias=AliasChoices("source_url", "url"))
    title: str = ""
    text: str
    page_type: str = "html"                      # html | pdf | api
    source_type: str = SourceType.WEB_PAGE.value
    content_hash: str | None = None
    crawl_timestamp: str | None = None
    effective_date: str | None = None
    version: str = "v1"
    headings: list[Any] = Field(default_factory=list)
    tables: list[Any] = Field(default_factory=list)
    pages: list[dict[str, Any]] = Field(default_factory=list)
    vehicle: dict[str, Any] | None = None
    attribution: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One retrievable unit. ``chunk_id`` = doc_id:version:strategy:NNN."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    doc_id: str
    version: str
    strategy: Literal["fixed", "structure"]
    chunk_index: int
    text: str
    title: str
    source_url: str
    source_type: str = SourceType.WEB_PAGE.value
    page_type: str = "html"
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    page_number: int | None = None
    vehicle: dict[str, Any] | None = None
    attribution: dict[str, str] = Field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Dict shape consumed by rag.retrieval (keys id/text/title/url)."""
        record = self.model_dump()
        record["id"] = self.chunk_id
        record["url"] = self.source_url
        record["heading"] = " > ".join(self.heading_path)
        return record


class RetrievedChunk(BaseModel):
    """A chunk as returned by retrieval, with scores before/after rerank."""

    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    url: str
    text: str
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


# --------------------------------------------------------------------------
# Vehicles (structured NHTSA ratings data)
# --------------------------------------------------------------------------

class Availability(str, Enum):
    STANDARD = "standard"
    OPTIONAL = "optional"
    VARIES = "varies"              # standard on some trims, optional on others
    NOT_AVAILABLE = "not_available"
    UNKNOWN = "unknown"


class VehicleRecord(BaseModel):
    id: int
    make: str
    model: str
    model_family: str
    model_year: int
    body_style: str | None = None
    drive_train: str | None = None
    vehicle_type: str | None = None
    variant_label: str
    overall_stars: float | None = None
    frontal_stars: float | None = None
    side_stars: float | None = None
    rollover_stars: float | None = None
    side_pole_stars: float | None = None
    rollover_risk_pct: float | None = None
    features: dict[str, str] = Field(default_factory=dict)       # name -> Availability
    evaluations: dict[str, str] = Field(default_factory=dict)    # NHTSA test results
    safety_concerns: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.model_year} {self.make} {self.model}"


class VehicleSlots(BaseModel):
    """What we know about the vehicle the user means. All optional."""

    make: str | None = None
    model: str | None = None          # a model family or a full model name
    model_year: int | None = None
    body_style: str | None = None
    drive_train: str | None = None
    vin: str | None = None

    def is_empty(self) -> bool:
        return not any(self.model_dump().values())

    def merged_with(self, newer: "VehicleSlots") -> "VehicleSlots":
        """Carry slots across turns; newer non-empty values win.

        Changing the make or model invalidates the more specific slots,
        so "and the Civic?" after "2020 Accord" keeps the year but drops
        the Accord's body style / drive train.
        """
        old = self.model_dump()
        new = {k: v for k, v in newer.model_dump().items() if v not in (None, "")}
        if "make" in new and old.get("make") and new["make"] != old["make"]:
            old.update(model=None, body_style=None, drive_train=None)
        if "model" in new and old.get("model") and new["model"] != old["model"]:
            old.update(body_style=None, drive_train=None)
        old.update(new)
        return VehicleSlots(**old)


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"          # exactly one vehicle variant
    AMBIGUOUS = "ambiguous"        # several variants match
    MISSING = "missing"            # a required slot is absent
    NOT_FOUND = "not_found"        # slots given but no such vehicle


class ResolutionResult(BaseModel):
    status: ResolutionStatus
    vehicles: list[VehicleRecord] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    message: str | None = None


# --------------------------------------------------------------------------
# Agent / NLU
# --------------------------------------------------------------------------

class Intent(str, Enum):
    CONCEPT = "concept_question"       # how/what/why -> RAG over NHTSA corpus
    VEHICLE_LOOKUP = "vehicle_lookup"  # ratings/features of a specific vehicle
    RECALL_LOOKUP = "recall_lookup"    # recalls for a specific vehicle
    VEHICLE_INFO = "vehicle_info"      # specifications/history/efficiency via RAG
    OUT_OF_SCOPE = "out_of_scope"      # prices, repair costs, non-safety topics
    SMALLTALK = "smalltalk"            # greetings, thanks


class NLUResult(BaseModel):
    intent: Intent
    intent_confidence: float = 1.0
    slots: VehicleSlots = Field(default_factory=VehicleSlots)
    features: list[str] = Field(default_factory=list)   # canonical feature keys
    backend: str = "rules"


class ClarificationDecision(BaseModel):
    should_ask: bool
    missing_slot: str | None = None
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    reason: str = ""


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

class Evidence(BaseModel):
    """One numbered item shown to the LLM as ``Evidence [ref]``."""

    ref: int
    evidence_id: str
    kind: Literal["chunk", "vehicle_record", "recall_record"]
    title: str
    url: str
    text: str
    scores: dict[str, float | None] = Field(default_factory=dict)
    attribution: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    ref: int
    title: str
    url: str
    kind: str = "chunk"
    attribution: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Chat API
# --------------------------------------------------------------------------

class ChatOptions(BaseModel):
    """Per-request overrides. Evaluation baselines are expressed with these:

    B0 use_retrieval=False | B1 retrieval_mode=bm25, rerank=False, fixed
    B2 retrieval_mode=dense, rerank=False, fixed | B3 hybrid, rerank, structure
    """

    use_retrieval: bool = True
    retrieval_mode: Literal["hybrid", "bm25", "dense"] | None = None
    rerank: bool | None = None
    chunk_strategy: Literal["fixed", "structure"] | None = None
    use_clarifier: bool = True
    debug: bool = True


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    options: ChatOptions = Field(default_factory=ChatOptions)


class TurnStatus(str, Enum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    REFUSED = "refused"
    ERROR = "error"


class Trace(BaseModel):
    """Everything needed for the evidence inspector and for evaluation."""

    intent: str | None = None
    nlu_backend: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    resolution: str | None = None
    standalone_question: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)  # before rerank
    evidence: list[dict[str, Any]] = Field(default_factory=list)    # sent to LLM
    evidence_decision: dict[str, Any] = Field(default_factory=dict)
    guard_flags: list[str] = Field(default_factory=list)
    llm: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    status: TurnStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    clarification: ClarificationDecision | None = None
    refusal_reason: str | None = None
    disclaimer: str | None = None
    trace: Trace | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackRequest(BaseModel):
    message_id: str
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)
