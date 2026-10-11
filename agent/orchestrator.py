"""The conversation turn pipeline. One public entry point:

    ChatService().handle(ChatRequest) -> ChatResponse

    user message
      -> input guard (prompt injection)                       agent/guards.py
      -> NLU: intent + vehicle slots + features              agent/nlu.py
      -> merge with session state (slot carry-over, pending clarification)
      -> route by intent
           smalltalk      -> canned reply
           out_of_scope   -> refusal
           vehicle/recall -> resolver -> clarifier (ask?)    structured/, agent/clarifier.py
                             -> vehicle evidence / recalls tool
           concept        -> hybrid retrieval + rerank + sufficiency   rag/evidence.py
      -> grounded generation + citation/quote validation     rag/generation.py
      -> persist turn + trace                                db/models.py

Both adapters use this class: FastAPI (api/main.py) and Streamlit (in-process
via api/client.py). Evaluation drives it too, using ChatOptions for B0-B3.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from agent import clarifier
from agent.guards import check_user_input, disclaimer_for
from agent.lexicon import CONCEPT_CUE, FOLLOW_UP_CUE, OUT_OF_SCOPE, PREDICTION, RANKING, RATING_WORDS, RECALL_WORDS, VEHICLE_INFO_WORDS, FUEL_ECONOMY_WORDS
from agent.nlu import NLUModel, RuleNLU, build_nlu, extract_vehicle_pair, extract_year
from core.config import Settings, get_settings
from core.observability import StageTimer
from core.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ClarificationDecision,
    Evidence,
    Intent,
    ResolutionResult,
    ResolutionStatus,
    SourceType,
    Trace,
    TurnStatus,
    VehicleSlots,
)
from db.models import ChatSession, Message, TurnTrace
from db.session import get_sessionmaker, init_db
from rag.evidence import EvidenceChecker
from rag.generation import GroundedGenerator
from rag.prompts import GENERAL_KNOWLEDGE_DISCLAIMER, REFUSAL_MESSAGES, smalltalk_reply
from structured.recalls import RecallsTool
from structured.resolver import VehicleResolver
from structured.safercar import vehicle_evidence

log = logging.getLogger(__name__)

PRONOUN_FOLLOW_UP = re.compile(r"\b(it|its|they|them|that|this|those|these)\b", re.I)
MAX_VEHICLE_EVIDENCE = 4


def condense_question(message: str, history: list[dict]) -> str:
    """Make a follow-up retrievable on its own.

    Heuristic for now: short follow-ups with a pronoun are prefixed with the
    previous user question. TODO(WP-05): LLM query rewriting, evaluated on
    the multi-hop golden questions.
    """
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if previous and len(message.split()) <= 10 and PRONOUN_FOLLOW_UP.search(message):
        return f"{previous} {message}"
    return message


class ChatService:
    def __init__(
        self,
        settings: Settings | None = None,
        session_factory: sessionmaker[Session] | None = None,
        nlu: NLUModel | None = None,
        resolver: VehicleResolver | None = None,
        evidence_checker: EvidenceChecker | None = None,
        generator: GroundedGenerator | None = None,
        recalls: RecallsTool | None = None,
    ):
        self.settings = settings or get_settings()
        if session_factory is None:
            init_db()
            session_factory = get_sessionmaker()
        self.session_factory = session_factory
        self.resolver = resolver or VehicleResolver(session_factory)
        self._nlu = nlu
        self.evidence_checker = evidence_checker or EvidenceChecker(settings=self.settings)
        self.generator = generator or GroundedGenerator()
        self.recalls = recalls or RecallsTool(session_factory, self.settings)

    @property
    def nlu(self) -> NLUModel:
        if self._nlu is None:
            self._nlu = build_nlu(self.resolver.vocab, self.settings)
        return self._nlu

    # ------------------------------------------------------------------ api
    def handle(self, request: ChatRequest) -> ChatResponse:
        timer = StageTimer()
        trace = Trace()
        with self.session_factory() as db:
            chat = db.get(ChatSession, request.session_id) if request.session_id else None
            if chat is None:
                chat = ChatSession(id=request.session_id or uuid.uuid4().hex, slots={}, pending_options={}, pending_features=[])
            else:
                db.expunge(chat)
            history = self._history(db, chat.id)

        # Retrieval, model loading and provider calls can take seconds. Keep
        # chat state detached until persistence so a new conversation does not
        # hold SQLite's writer lock while another UI/API request is answering.
        try:
            response = self._turn(chat, request, history, trace, timer)
        except Exception as error:  # never leak a stack trace to the UI
            log.exception("Turn failed")
            # Keep a short, key-free description for the evidence inspector / DB trace.
            detail = " ".join(str(error).split())[:300]
            trace.guard_flags.append(f"error:{type(error).__name__}")
            trace.llm = {**trace.llm, "error": f"{type(error).__name__}: {detail}"}
            response = self._reply(chat, TurnStatus.ERROR, REFUSAL_MESSAGES["error"], trace, refusal_reason="error")

        trace.timings_ms = {**timer.timings, "total": timer.total()}
        if request.options.debug:
            response.trace = trace
        with self.session_factory() as db:
            db.add(chat)
            db.add(Message(id=uuid.uuid4().hex, session_id=chat.id, role="user", content=request.message))
            db.add(Message(
                id=response.message_id,
                session_id=chat.id,
                role="assistant",
                content=response.answer,
                status=response.status.value,
                citations=[c.model_dump() for c in response.citations],
            ))
            db.add(TurnTrace(
                message_id=response.message_id,
                intent=trace.intent,
                status=response.status.value,
                latency_ms=trace.timings_ms.get("total"),
                cost_usd=trace.llm.get("cost_usd"),
                trace=trace.model_dump(mode="json"),
            ))
            chat.updated_at = datetime.now(timezone.utc)
            db.commit()
        return response

    # ------------------------------------------------------------- the turn
    def _turn(self, chat: ChatSession, request: ChatRequest, history, trace: Trace, timer: StageTimer) -> ChatResponse:
        message = request.message.strip()
        options = request.options

        guard = check_user_input(message)
        trace.guard_flags += guard.flags
        if guard.blocked:
            return self._refuse(chat, guard.reason, trace)
        if not re.search(r"[A-Za-z0-9]", message):  # "?", "..." -> say what the bot can do
            return self._reply(chat, TurnStatus.ANSWERED, smalltalk_reply(""), trace)
        if PREDICTION.search(message):  # forecasts are refused before any vehicle lookup
            self._clear_pending(chat)
            return self._refuse(chat, "unsupported_prediction", trace)

        with timer.stage("nlu"):
            nlu = self.nlu.parse(message, history)
        intent, features = nlu.intent, list(nlu.features)
        # Older trained artifacts predate ADR-010. Keep explicit new-topic
        # routing deterministic across backends without weakening safety guards.
        if VEHICLE_INFO_WORDS.search(message) and not (OUT_OF_SCOPE.search(message) or RECALL_WORDS.search(message)
                                                       or RATING_WORDS.search(message) or features):
            intent, _ = RuleNLU(self.resolver.vocab).classify_intent(message, nlu.slots, features)
            nlu.intent = intent
        trace.nlu_backend = nlu.backend
        if RANKING.search(message) and not (nlu.slots.make or nlu.slots.model):
            self._clear_pending(chat)  # "which car is safest?" ranks every vehicle: say so, offer a comparison
            return self._refuse(chat, "ranking_unsupported", trace)

        # ---- clarification follow-up? (button click or short slot answer)
        picked = (chat.pending_options or {}).get(message.lower())
        question = message
        is_follow_up = bool(chat.pending_intent) and (
            picked is not None
            or (len(message.split()) <= 6 and not nlu.slots.is_empty() and intent != Intent.RECALL_LOOKUP)
        )
        if is_follow_up:
            intent = Intent(chat.pending_intent)
            features = list(dict.fromkeys((chat.pending_features or []) + features))
            question = f"{chat.pending_question} ({message})" if chat.pending_question else message
        else:
            open_features = list(chat.pending_features or [])  # an unanswered "which version?" keeps its topic
            self._clear_pending(chat)
            intent, features = self._contextual_intent(chat, message, nlu, open_features)

        new_slots = nlu.slots
        if picked:
            new_slots = VehicleSlots(**{**new_slots.model_dump(exclude_none=True),
                                        **{k: v for k, v in picked.items() if k != "vehicle_id"}})
        slots = VehicleSlots(**(chat.slots or {}))
        if intent in (Intent.VEHICLE_LOOKUP, Intent.RECALL_LOOKUP, Intent.VEHICLE_INFO):
            slots = slots.merged_with(new_slots)
        trace.intent = intent.value
        trace.slots = slots.model_dump(exclude_none=True)

        if intent == Intent.SMALLTALK:
            return self._reply(chat, TurnStatus.ANSWERED, smalltalk_reply(message), trace)
        if intent == Intent.OUT_OF_SCOPE:
            return self._refuse(chat, "out_of_scope", trace)
        if intent == Intent.VEHICLE_INFO:
            return self._answer_vehicle_info(chat, message, question, slots, is_follow_up, history, options, trace, timer)
        if intent == Intent.VEHICLE_LOOKUP and not is_follow_up:
            pair = extract_vehicle_pair(message, self.resolver.vocab)
            if pair:
                return self._compare(chat, message, pair, features, history, trace, timer)

        evidence: list[Evidence] = []
        if intent in (Intent.VEHICLE_LOOKUP, Intent.RECALL_LOOKUP):
            with timer.stage("resolve"):
                if picked and picked.get("vehicle_id"):
                    record = self.resolver.get(int(picked["vehicle_id"]))
                    resolution = ResolutionResult(status=ResolutionStatus.RESOLVED, vehicles=[record]) if record else self.resolver.resolve(slots)
                else:
                    resolution = self.resolver.resolve(slots)
            trace.resolution = resolution.status.value
            if not self.resolver.is_ready():
                return self._refuse(chat, "vehicle_data_unavailable", trace)

            decision = clarifier.decide(
                intent, resolution, slots, features,
                chat.clarification_turns or 0, self.settings.max_clarification_turns,
            ) if options.use_clarifier else ClarificationDecision(should_ask=False, reason="clarifier_disabled")
            trace.evidence_decision = {"clarification": decision.reason}

            if decision.should_ask:
                chat.pending_intent = intent.value
                chat.pending_features = features
                chat.pending_question = question if not is_follow_up else chat.pending_question
                chat.pending_options = self._option_map(decision, resolution)
                chat.clarification_turns = (chat.clarification_turns or 0) + 1
                chat.slots = slots.model_dump(exclude_none=True)
                return self._reply(chat, TurnStatus.NEEDS_CLARIFICATION, decision.question or "", trace, clarification=decision)

            self._clear_pending(chat)
            vehicles = resolution.vehicles
            if not vehicles:
                chat.slots = slots.model_dump(exclude_none=True)
                return self._refuse(chat, "no_evidence", trace)
            chosen = vehicles[0]
            chat.slots = VehicleSlots(
                make=chosen.make, model=chosen.model if len(vehicles) == 1 else slots.model,
                model_year=chosen.model_year,
                body_style=chosen.body_style if len(vehicles) == 1 else slots.body_style,
                drive_train=chosen.drive_train if len(vehicles) == 1 else slots.drive_train,
            ).model_dump(exclude_none=True)

            if intent == Intent.VEHICLE_LOOKUP:
                same = decision.reason == "all_variants_agree"
                for ref, vehicle in enumerate(vehicles[: 1 if same else MAX_VEHICLE_EVIDENCE], start=1):
                    item = vehicle_evidence(vehicle, ref, features)
                    if same and len(vehicles) > 1:
                        item.text += (
                            f"\nNote: NHTSA lists {len(vehicles)} versions of this vehicle "
                            f"and these values are the same for all of them."
                        )
                    evidence.append(item)
            else:
                with timer.stage("recalls"):
                    recall_evidence, status = self.recalls.lookup(vehicles, start_ref=1)
                trace.evidence_decision["recalls"] = status
                if status == "unavailable":
                    return self._reply(
                        chat, TurnStatus.REFUSED,
                        "I couldn't reach NHTSA's recall service right now. Please check "
                        "https://www.nhtsa.gov/recalls with your VIN.",
                        trace, refusal_reason="recalls_unavailable",
                    )
                if status == "none":
                    v = vehicles[0]
                    recall_evidence = [Evidence(
                        ref=1, evidence_id=f"recall:none:{v.id}", kind="recall_record",
                        title=f"NHTSA Recalls API: {v.model_year} {v.make} {v.model_family}",
                        url=self.recalls._url(v.make, v.model_family, v.model_year),
                        text=(f"The NHTSA Recalls API returned no recalls for the {v.model_year} {v.make} "
                              f"{v.model_family} when checked on {datetime.now(timezone.utc).date()}."),
                    )]
                evidence.extend(recall_evidence)
            question = f"{question}\n(Vehicle: {chosen.model_year} {chosen.make} {chosen.model})"

        elif intent == Intent.CONCEPT:
            chat.last_intent, chat.last_features = intent.value, features  # follow-ups keep the topic
            if not options.use_retrieval:  # baseline B0
                with timer.stage("generation"):
                    answer = self.generator.generate_closed_book(message)
                trace.llm = answer.llm.as_trace()
                return self._reply(chat, TurnStatus.ANSWERED, answer.text, trace)

            standalone = condense_question(message, history)
            trace.standalone_question = standalone
            with timer.stage("retrieval"):
                if FUEL_ECONOMY_WORDS.search(standalone):
                    scope = [SourceType.FUEL_ECONOMY.value]
                elif VEHICLE_INFO_WORDS.search(standalone):
                    scope = [SourceType.WIKIPEDIA.value, SourceType.FUEL_ECONOMY.value]
                else:
                    scope = [SourceType.AUTOGUIDE.value, SourceType.WEB_PAGE.value, SourceType.PDF.value,
                             SourceType.RECALLS_API.value, SourceType.RATINGS_DATASET.value, SourceType.REFERENCE.value]
                check = self.evidence_checker.check_evidence(
                    standalone,
                    mode=options.retrieval_mode,
                    rerank=options.rerank,
                    strategy=options.chunk_strategy,
                    source_types=scope,
                )
            trace.candidates = [self._trace_row(c) for c in check.get("candidates", [])]
            trace.evidence_decision = {
                "sufficient": check["sufficient"],
                "reason": check["reason"],
                "top_score": check.get("top_score"),
                "threshold": check.get("threshold"),
            }
            if not check["sufficient"]:
                if check["reason"] == "index_unavailable" and self._real_llm_available():
                    return self._answer_general_knowledge(chat, question, history, trace, timer)
                return self._refuse(chat, check["reason"], trace)
            evidence = self._chunk_evidence(check["evidence"])

        chat.last_intent, chat.last_features = intent.value, features
        return self._generate_answer(chat, message, question, intent, evidence, history, trace, timer)

    def _real_llm_available(self) -> bool:
        """True when a real generation provider is configured.

        The general-knowledge fallback needs an actual LLM; with the mock
        provider the honest refusal stays, since mock text is a placeholder,
        not an answer.
        """
        return self.settings.resolved_llm_provider() != "mock"

    def _answer_general_knowledge(self, chat, question, history, trace, timer) -> ChatResponse:
        """Answer an automotive question from the LLM's own knowledge.

        Reached only when the article index is unavailable AND a real LLM
        provider is configured. Predictions, off-topic questions and prompt
        injections are refused earlier in handle(), and the system prompt
        forbids inventing vehicle-specific ratings/recalls/specs, so this
        path cannot fabricate NHTSA data. The disclaimer marks the answer
        as general knowledge, never NHTSA-sourced.
        """
        with timer.stage("generation"):
            generated = self.generator.generate_general_knowledge(question, history)
        trace.llm = generated.llm.as_trace()
        trace.guard_flags += generated.flags
        return self._reply(
            chat, TurnStatus.ANSWERED, generated.text, trace,
            disclaimer=GENERAL_KNOWLEDGE_DISCLAIMER,
        )

    def _generate_answer(self, chat, message, question, intent, evidence, history, trace, timer) -> ChatResponse:
        trace.evidence = [{"ref": e.ref, "id": e.evidence_id, "kind": e.kind, "title": e.title,
                           "url": e.url, "text": e.text[:600], **e.scores} for e in evidence]
        with timer.stage("generation"):
            generated = self.generator.generate(question, evidence, history)
        trace.llm = generated.llm.as_trace()
        trace.guard_flags += generated.flags
        if not generated.valid_citations:
            trace.guard_flags.append(f"citation_validation_failed:{sorted(generated.invalid_refs)}")
            return self._refuse(chat, "citation_validation_failed", trace)

        citations = [Citation(ref=e.ref, title=e.title, url=e.url, kind=e.kind, attribution=e.attribution)
                     for e in evidence if e.ref in generated.cited_refs]
        return self._reply(
            chat, TurnStatus.ANSWERED, generated.text, trace,
            citations=citations, disclaimer=disclaimer_for(message, intent.value),
        )

    def _compare(self, chat, message, pair, features, history, trace, timer) -> ChatResponse:
        """Two vehicles in one question: evidence for both, one answer."""
        trace.intent = Intent.VEHICLE_LOOKUP.value
        trace.slots = {"compare": [s.model_dump(exclude_none=True) for s in pair]}
        if not self.resolver.is_ready():
            return self._refuse(chat, "vehicle_data_unavailable", trace)
        with timer.stage("resolve"):
            resolutions = [self.resolver.resolve(s) for s in pair]
        trace.resolution = "compare:" + ",".join(r.status.value for r in resolutions)
        self._clear_pending(chat)
        missing = [s for s, r in zip(pair, resolutions) if not r.vehicles]
        if missing:
            named = " and ".join(
                " ".join(str(x) for x in (s.model_year, (s.make or "").title(), (s.model or "").title()) if x) or "one vehicle"
                for s in missing
            )
            text = (f"I couldn't pin down {named} in NHTSA's ratings data. To compare, give the model year, "
                    "make and model of both vehicles, for example \"2020 Toyota Camry vs 2020 Honda Accord\".")
            return self._reply(chat, TurnStatus.NEEDS_CLARIFICATION, text, trace)

        evidence: list[Evidence] = []
        labels = []
        for resolution in resolutions:
            vehicles = resolution.vehicles
            same = not clarifier.variants_disagree(vehicles, features)
            first = vehicles[0]
            labels.append(f"{first.model_year} {first.make.title()} {first.model_family.title()}")
            for vehicle in vehicles[: 1 if same else 2]:
                item = vehicle_evidence(vehicle, len(evidence) + 1, features)
                if len(vehicles) > 1:
                    item.text += (
                        f"\nNote: NHTSA lists {len(vehicles)} versions of this vehicle and these values are the same for all of them."
                        if same else
                        f"\nNote: NHTSA lists {len(vehicles)} versions of this vehicle with different values; this is one of them."
                    )
                evidence.append(item)
        trace.evidence_decision = {"clarification": "comparison", "vehicles": labels}
        chat.slots = {}
        chat.last_intent, chat.last_features = Intent.VEHICLE_LOOKUP.value, features
        question = f"{message}\n(Compare: {' vs '.join(labels)})"
        return self._generate_answer(chat, message, question, Intent.VEHICLE_LOOKUP, evidence, history, trace, timer)

    # ------------------------------------------------------------- helpers
    def _answer_vehicle_info(self, chat, message, question, slots, is_follow_up,
                             history, options, trace, timer) -> ChatResponse:
        if not is_follow_up and chat.last_intent == Intent.VEHICLE_INFO.value and len(message.split()) <= 8 and not VEHICLE_INFO_WORDS.search(message):
            previous = next((m["content"] for m in reversed(history)
                if m["role"] == "user" and VEHICLE_INFO_WORDS.search(m["content"])), message)
            question = f"{previous}\nFollow-up: {message}"
        history_only = bool(re.search(r"\b(history|overview|tell me about|information about|info on)\b", question, re.I)) and not (
            FUEL_ECONOMY_WORDS.search(question) or re.search(r"\b(engine|horsepower|hp|torque|transmission|specs|specifications|drivetrain|fuel type)\b", question, re.I))
        if history_only and extract_year(question) is None:
            slots.model_year = None
        required = [key for key in ("make", "model", *([] if history_only else ["model_year"])) if not getattr(slots, key)]
        chat.slots = slots.model_dump(exclude_none=True)
        trace.slots = chat.slots
        if required:
            if (chat.clarification_turns or 0) >= self.settings.max_clarification_turns:
                self._clear_pending(chat)
                return self._refuse(chat, "no_evidence", trace)
            label = {"make": "make", "model": "model", "model_year": "model year"}[required[0]]
            decision = ClarificationDecision(should_ask=True, missing_slot=required[0],
                question=f"What is the vehicle's {label}? Specifications and fuel economy can vary by year and version.",
                reason="web_vehicle_details_missing")
            chat.pending_intent = Intent.VEHICLE_INFO.value
            chat.pending_question = question if not is_follow_up else chat.pending_question
            chat.pending_options, chat.pending_features = {}, []
            chat.clarification_turns = (chat.clarification_turns or 0) + 1
            return self._reply(chat, TurnStatus.NEEDS_CLARIFICATION, decision.question, trace, clarification=decision)
        self._clear_pending(chat)
        # Missing metadata never acts as a wildcard, including yearless model
        # articles used for a numerical model-year specification question.
        vehicle_filter = {"make": slots.make, "model": slots.model}
        for key, value in (("year", slots.model_year), ("drive_train", slots.drive_train), ("body_style", slots.body_style)):
            if value:
                vehicle_filter[key] = value
        scope = ([SourceType.WIKIPEDIA.value] if history_only else
                 [SourceType.FUEL_ECONOMY.value] if FUEL_ECONOMY_WORDS.search(question) else
                 [SourceType.WIKIPEDIA.value, SourceType.FUEL_ECONOMY.value])
        standalone = f"{question}\nCurrent vehicle: {' '.join(str(v) for v in (slots.model_year, slots.make, slots.model, slots.drive_train) if v)}"
        trace.standalone_question = standalone
        with timer.stage("retrieval"):
            check = self.evidence_checker.check_evidence(standalone, vehicle_filter=vehicle_filter,
                source_types=scope, mode=options.retrieval_mode, rerank=options.rerank, strategy=options.chunk_strategy)
        trace.candidates = [self._trace_row(c) for c in check.get("candidates", [])]
        trace.evidence_decision = {key: check.get(key) for key in ("sufficient", "reason", "top_score", "threshold")}
        trace.evidence_decision["vehicle_filter"] = vehicle_filter
        chat.last_intent, chat.last_features = Intent.VEHICLE_INFO.value, []
        if not check["sufficient"]:
            return self._refuse(chat, check["reason"], trace)
        return self._generate_answer(chat, message, standalone, Intent.VEHICLE_INFO,
            self._chunk_evidence(check["evidence"]), history, trace, timer)

    @staticmethod
    def _chunk_evidence(results: list[dict]) -> list[Evidence]:
        return [Evidence(ref=ref, evidence_id=str(row["id"]), kind="chunk",
            title=row.get("title") or "Vehicle document", url=row.get("url", ""), text=row.get("text", ""),
            attribution=row.get("attribution") or row.get("chunk", {}).get("attribution", {}),
            scores={key: row.get(key) for key in ("retrieval_score", "bm25_score", "dense_score", "rerank_score")})
            for ref, row in enumerate(results, start=1)]

    @staticmethod
    def _contextual_intent(chat: ChatSession, message: str, nlu,
                           open_features: list[str] | None = None) -> tuple[Intent, list[str]]:
        """Resolve follow-ups that only make sense in context:
        "what about the 2020 one?" / "does it have blind spot detection?" after a
        vehicle answer, or "does it work at night?" after a concept answer or
        while a question about a feature is still open.
        """
        intent, features = nlu.intent, list(nlu.features)
        if intent == Intent.SMALLTALK or OUT_OF_SCOPE.search(message):
            return intent, features
        last = Intent(chat.last_intent) if chat.last_intent else None
        if chat.slots and not (RECALL_WORDS.search(message) or RATING_WORDS.search(message) or features):
            if VEHICLE_INFO_WORDS.search(message) and not CONCEPT_CUE.search(message):
                return Intent.VEHICLE_INFO, features
            if last == Intent.VEHICLE_INFO and len(message.split()) <= 8 and (
                    PRONOUN_FOLLOW_UP.search(message) or not nlu.slots.is_empty()):
                return Intent.VEHICLE_INFO, features
        weak = nlu.intent_confidence <= 0.6 or intent == Intent.CONCEPT
        pronoun = bool(PRONOUN_FOLLOW_UP.search(message))
        short_slots = len(message.split()) <= 8 and not nlu.slots.is_empty()
        if last in (Intent.VEHICLE_LOOKUP, Intent.RECALL_LOOKUP) and chat.slots and weak and (pronoun or short_slots):
            if RECALL_WORDS.search(message):
                return Intent.RECALL_LOOKUP, features
            if features or RATING_WORDS.search(message) or short_slots:
                return Intent.VEHICLE_LOOKUP, features or list(chat.last_features or [])
        # "what about the side crash rating?" / "and blind spot detection?" after a vehicle answer
        if (last in (Intent.VEHICLE_LOOKUP, Intent.RECALL_LOOKUP) and chat.slots and nlu.slots.is_empty()
                and len(message.split()) <= 12 and not CONCEPT_CUE.search(message)
                and (FOLLOW_UP_CUE.search(message) or RATING_WORDS.search(message) or RECALL_WORDS.search(message))):
            if RECALL_WORDS.search(message):
                return Intent.RECALL_LOOKUP, features
            return Intent.VEHICLE_LOOKUP, features
        if last == Intent.CONCEPT and intent == Intent.OUT_OF_SCOPE and nlu.intent_confidence <= 0.6 and pronoun:
            return Intent.CONCEPT, features
        # "Does it work in bad weather?" about a feature under discussion
        known = list(dict.fromkeys([*(open_features or []), *(chat.last_features or [])]))
        unsure = intent == Intent.CONCEPT or (intent == Intent.OUT_OF_SCOPE and nlu.intent_confidence <= 0.6)
        if known and pronoun and unsure and nlu.slots.is_empty():
            return Intent.CONCEPT, features or known
        return intent, features

    def _history(self, db, session_id: str) -> list[dict]:
        rows = db.scalars(
            select(Message).where(Message.session_id == session_id)
            .order_by(Message.created_at.desc()).limit(self.settings.history_turns * 2)
        ).all()
        return [{"role": m.role, "content": m.content} for m in reversed(rows)]

    @staticmethod
    def _clear_pending(chat: ChatSession) -> None:
        chat.pending_intent = None
        chat.pending_features = []
        chat.pending_question = None
        chat.pending_options = {}
        chat.clarification_turns = 0

    @staticmethod
    def _option_map(decision: ClarificationDecision, resolution: ResolutionResult) -> dict:
        mapping: dict[str, dict] = {}
        if decision.missing_slot == "variant":
            by_label = {}
            for vehicle in resolution.vehicles:
                by_label.setdefault(vehicle.variant_label, vehicle)
            for label in decision.options:
                v = by_label.get(label)
                if v:
                    mapping[label.lower()] = {"vehicle_id": v.id, "model": v.model,
                                              "body_style": v.body_style, "drive_train": v.drive_train}
        elif decision.missing_slot == "model_year":
            mapping = {o.lower(): {"model_year": int(o)} for o in decision.options if o.isdigit()}
        elif decision.missing_slot in ("make", "model"):
            mapping = {o.lower(): {decision.missing_slot: o} for o in decision.options}
        return mapping

    @staticmethod
    def _trace_row(candidate: dict) -> dict:
        keys = ("id", "title", "url", "retrieval_rank", "retrieval_score", "bm25_rank", "bm25_score",
                "dense_rank", "dense_score", "rerank_score", "rerank_rank")
        row = {k: candidate.get(k) for k in keys}
        row["text"] = (candidate.get("text") or "")[:300]
        return row

    def _refuse(self, chat: ChatSession, reason: str | None, trace: Trace) -> ChatResponse:
        reason = reason or "no_evidence"
        text = REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["no_evidence"])
        return self._reply(chat, TurnStatus.REFUSED, text, trace, refusal_reason=reason)

    @staticmethod
    def _reply(chat, status, answer, trace, citations=None, clarification=None,
               refusal_reason=None, disclaimer=None) -> ChatResponse:
        return ChatResponse(
            session_id=chat.id,
            message_id=uuid.uuid4().hex,
            status=status,
            answer=answer,
            citations=citations or [],
            clarification=clarification,
            refusal_reason=refusal_reason,
            disclaimer=disclaimer,
        )
