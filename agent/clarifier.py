"""Decide whether to ask a follow-up question, and phrase it.

Policy (in order):
1. A required slot (make -> model -> model_year) is missing  -> ask for it.
2. The vehicle doesn't exist in the data                     -> ask, offering suggestions.
3. Several variants match AND the answer differs between them -> ask which variant.
   If every variant gives the same answer, do NOT ask (avoid over-asking).
4. Recall questions never ask for a variant: recalls are keyed by make/model/year.
5. After ``max_turns`` clarifications in a row, stop asking and answer for
   all candidate variants instead.

Question wording is templated (deterministic, testable). An LLM may rephrase
it later (WP-07), but the *decision* stays here.
"""

from __future__ import annotations

from core.schemas import (
    ClarificationDecision,
    Intent,
    ResolutionResult,
    ResolutionStatus,
    VehicleRecord,
    VehicleSlots,
)
from structured.safercar import answer_fields, field_value

MAX_OPTIONS = 10


def _title(text: str | None) -> str:
    return (text or "").title()


def variants_disagree(vehicles: list[VehicleRecord], features: list[str]) -> bool:
    fields = answer_fields(features)
    for field in fields:
        values = {repr(field_value(v, field)) for v in vehicles}
        if len(values) > 1:
            return True
    return False


def decide(
    intent: Intent,
    resolution: ResolutionResult,
    slots: VehicleSlots,
    features: list[str],
    clarification_turns: int = 0,
    max_turns: int = 3,
) -> ClarificationDecision:
    if resolution.status == ResolutionStatus.RESOLVED:
        return ClarificationDecision(should_ask=False, reason="resolved")

    if clarification_turns >= max_turns:
        return ClarificationDecision(should_ask=False, reason="max_clarification_turns")

    if resolution.status == ResolutionStatus.AMBIGUOUS:
        if intent == Intent.RECALL_LOOKUP:
            return ClarificationDecision(should_ask=False, reason="recalls_keyed_by_make_model_year")
        if not variants_disagree(resolution.vehicles, features):
            return ClarificationDecision(should_ask=False, reason="all_variants_agree")
        vehicle = resolution.vehicles[0]
        name = f"{vehicle.model_year} {_title(vehicle.make)} {_title(vehicle.model_family)}"
        return ClarificationDecision(
            should_ask=True,
            missing_slot="variant",
            question=(
                f"NHTSA rates {len(resolution.options)} versions of the {name}, and the answer "
                f"differs between them. Which one do you mean?"
            ),
            options=resolution.options[:MAX_OPTIONS],
            reason="variants_disagree",
        )

    if resolution.status == ResolutionStatus.NOT_FOUND:
        prompt = resolution.message or "I couldn't find that vehicle in NHTSA's ratings data."
        suffix = " Did you mean one of these?" if resolution.options else " Could you check the make, model and year?"
        return ClarificationDecision(
            should_ask=True,
            missing_slot=(resolution.missing_slots or [None])[0],
            question=prompt + suffix,
            options=resolution.options[:MAX_OPTIONS],
            reason="not_found",
        )

    # MISSING
    slot = (resolution.missing_slots or ["make"])[0]
    if slot == "make" and not slots.model:
        question = "Which vehicle do you mean? Please tell me the model year, make and model (for example, 2020 Honda Accord)."
        options: list[str] = []
    elif slot == "make":
        question = f"Several makes have a model called {_title(slots.model)}. Which make is it?"
        options = resolution.options
    elif slot == "model":
        question = f"Which {_title(slots.make)} model do you mean?"
        if not slots.model_year:
            question += " Including the model year helps too."
        options = resolution.options
    else:
        question = f"Which model year is it? {resolution.message or ''}".strip()
        options = resolution.options
    return ClarificationDecision(
        should_ask=True,
        missing_slot=slot,
        question=question,
        options=options[:MAX_OPTIONS],
        reason=f"missing_{slot}",
    )
