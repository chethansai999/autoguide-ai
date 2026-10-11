"""Resolve user-provided vehicle slots to NHTSA ratings rows.

The resolver - not a language model - decides what information is missing,
because only the data knows whether a "2019 F-150" is one vehicle or ten.

    resolve(VehicleSlots) -> ResolutionResult
        RESOLVED   exactly one variant
        AMBIGUOUS  several variants (options = variant labels)
        MISSING    make / model / model_year absent (options = valid values)
        NOT_FOUND  slots given but no such vehicle (options = suggestions)
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from core.schemas import ResolutionResult, ResolutionStatus, VehicleSlots
from db.models import Vehicle
from structured.safercar import compact, to_record
from structured.vocab import VehicleVocab

DRIVE_CLASSES = {
    "AWD": {"AWD"}, "FWD": {"FWD"}, "RWD": {"RWD"},
    "4WD": {"4WD", "4X4"}, "4X4": {"4WD", "4X4"},
    "2WD": {"2WD", "4X2"}, "4X2": {"2WD", "4X2"},
}

BODY_SYNONYMS = {
    "SEDAN": ["4DR"], "COUPE": ["2DR", "C"], "CONVERTIBLE": ["CV", "CONVERTIBLE"],
    "HATCHBACK": ["HB", "HATCH"], "SUV": ["SUV"], "WAGON": ["SW", "WAGON"],
    "VAN": ["VAN", "PV", "MV"], "MINIVAN": ["VAN", "MV"],
    "CREWCAB": ["PUCC", "CREW"], "EXTENDEDCAB": ["PUEC", "SUPERCAB", "EXTENDED"],
    "REGULARCAB": ["PURC", "REGULAR"], "PICKUP": ["PU"],
}


def model_matches(query: str, model: str) -> bool:
    """Token-boundary prefix match: 'f150' ~ 'F-150 SUPERCAB', 'crv' ~ 'CR-V HYBRID'."""
    target = compact(query)
    if not target:
        return False
    acc = ""
    for token in model.upper().replace("(", " (").split():
        acc += compact(token)
        if acc == target:
            return True
        if len(acc) >= len(target):
            return False
    return False


def drive_matches(requested: str, actual: str | None) -> bool:
    wanted = DRIVE_CLASSES.get(compact(requested), {compact(requested)})
    have = {compact(p) for p in (actual or "").upper().split("/")}
    return bool(wanted & have)


def body_matches(requested: str, vehicle: Vehicle) -> bool:
    key = compact(requested)
    needles = BODY_SYNONYMS.get(key, [key])
    hay = compact(f"{vehicle.body_style or ''} {vehicle.model}")
    return any(n and n in hay for n in needles)


class VehicleResolver:
    def __init__(self, session_factory: sessionmaker[Session], vocab: VehicleVocab | None = None):
        self.session_factory = session_factory
        self._vocab = vocab

    @property
    def vocab(self) -> VehicleVocab:
        if self._vocab is None:
            with self.session_factory() as session:
                self._vocab = VehicleVocab.from_session(session)
        return self._vocab

    def is_ready(self) -> bool:
        return bool(self.vocab.makes)

    # --------------------------------------------------------------- main
    def resolve(self, slots: VehicleSlots) -> ResolutionResult:
        vocab = self.vocab
        if not vocab.makes:
            return ResolutionResult(
                status=ResolutionStatus.NOT_FOUND,
                message="Vehicle data is not loaded. Run `python ingest.py load-safercar`.",
            )

        make = vocab.canonical_make(slots.make) if slots.make else None
        if slots.make and not make:
            return ResolutionResult(
                status=ResolutionStatus.NOT_FOUND,
                missing_slots=["make"],
                options=vocab.suggest_makes(slots.make),
                message=f"I couldn't find the make \"{slots.make}\" in NHTSA's ratings data.",
            )

        if not make and slots.model:
            makes = vocab.makes_for_model(slots.model)
            if len(makes) == 1:
                make = makes[0]
            elif len(makes) > 1:
                return ResolutionResult(
                    status=ResolutionStatus.MISSING, missing_slots=["make"], options=makes[:8]
                )

        if not make:
            return ResolutionResult(
                status=ResolutionStatus.MISSING,
                missing_slots=["make"],
                options=vocab.popular_makes(),
            )

        if not slots.model:
            return ResolutionResult(
                status=ResolutionStatus.MISSING,
                missing_slots=["model"],
                options=vocab.families.get(make, [])[:10],
            )

        with self.session_factory() as session:
            rows = session.scalars(select(Vehicle).where(Vehicle.make == make)).all()
            rows = [
                v for v in rows
                if model_matches(slots.model, v.model) or model_matches(slots.model, v.model_family)
            ]
            if not rows:
                return ResolutionResult(
                    status=ResolutionStatus.NOT_FOUND,
                    missing_slots=["model"],
                    options=vocab.suggest_families(make, slots.model),
                    message=f"I couldn't find a {make.title()} \"{slots.model}\" in NHTSA's ratings data.",
                )

            years = sorted({v.model_year for v in rows}, reverse=True)
            family = rows[0].model_family
            if not slots.model_year:
                return ResolutionResult(
                    status=ResolutionStatus.MISSING,
                    missing_slots=["model_year"],
                    options=[str(y) for y in years[:10]],
                    message=f"NHTSA has data for the {make.title()} {family.title()} from {min(years)} to {max(years)}.",
                )

            rows = [v for v in rows if v.model_year == slots.model_year]
            if not rows:
                nearest = sorted(years, key=lambda y: abs(y - slots.model_year))[:5]
                return ResolutionResult(
                    status=ResolutionStatus.NOT_FOUND,
                    missing_slots=["model_year"],
                    options=[str(y) for y in sorted(nearest, reverse=True)],
                    message=(
                        f"NHTSA's ratings data has no {slots.model_year} {make.title()} "
                        f"{family.title()}. Available nearby years are listed below."
                    ),
                )

            if slots.drive_train:
                filtered = [v for v in rows if drive_matches(slots.drive_train, v.drive_train)]
                rows = filtered or rows
            if slots.body_style:
                filtered = [v for v in rows if body_matches(slots.body_style, v)]
                rows = filtered or rows

            # exact variant pick from a clarification button
            exact = [v for v in rows if compact(v.model) == compact(slots.model)]
            if exact and (slots.body_style or slots.drive_train):
                rows = exact

            records = [to_record(v) for v in rows]

        labels = list(dict.fromkeys(r.variant_label for r in records))
        if len(labels) == 1:
            return ResolutionResult(status=ResolutionStatus.RESOLVED, vehicles=records[:1])
        return ResolutionResult(
            status=ResolutionStatus.AMBIGUOUS, vehicles=records, options=labels
        )

    def get(self, vehicle_id: int):
        with self.session_factory() as session:
            vehicle = session.get(Vehicle, vehicle_id)
            return to_record(vehicle) if vehicle else None
