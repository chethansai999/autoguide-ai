"""Load and normalize NHTSA 5-Star Safety Ratings data (Safercar_data.csv).

The raw file has 128 columns and inconsistent values ("S", "Std", "Standard",
"STANDARD" all mean standard). This module turns each row into a clean
``vehicles`` row and renders a vehicle as citable evidence text.

    python ingest.py load-safercar            # CSV -> vehicles table
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from core.config import get_settings
from core.schemas import Availability, Evidence, VehicleRecord
from db.models import Vehicle

log = logging.getLogger(__name__)

# canonical feature key -> (label, CSV columns in priority order)
FEATURES: dict[str, tuple[str, tuple[str, ...]]] = {
    "abs": ("Anti-lock brakes (ABS)", ("ABS",)),
    "electronic_stability_control": ("Electronic stability control (ESC)", ("NHTSA_ESC",)),
    "forward_collision_warning": ("Forward collision warning", ("FRNT_COLLISION_WARNING", "NHTSA_FRNT_COLLISION_WARNING")),
    "automatic_emergency_braking": ("Automatic emergency braking (crash imminent braking)", ("CRASH_IMMINENT_BRAKE", "NHTSA_CRASH_IMMINENT_BRAKE")),
    "dynamic_brake_support": ("Dynamic brake support", ("DYNAMIC_BRAKE_SUPPORT", "NHTSA_DYNAMIC_BRAKE_SUPPORT")),
    "lane_departure_warning": ("Lane departure warning", ("LANE_DEPARTURE_WARNING", "NHTSA_LANE_DEPARTURE_WARNING")),
    "blind_spot_detection": ("Blind spot detection", ("BLIND_SPOT_DETECTION",)),
    "adaptive_cruise_control": ("Adaptive cruise control", ("ADAPTIVE_CRUISE_CONTROL",)),
    "backup_camera": ("Rearview (backup) camera", ("BACKUP_CAMERA", "NHTSA_BACKUP_CAMERA")),
    "daytime_running_lights": ("Daytime running lights", ("DAY_RUN_LIGHTS",)),
    "head_side_airbags": ("Head side airbags (curtain)", ("HEAD_SAB",)),
    "torso_side_airbags": ("Torso side airbags", ("TORSO_SAB",)),
    "pelvis_side_airbags": ("Pelvis side airbags", ("PELVIS_SAB",)),
    "knee_airbags": ("Knee airbags", ("KNEE_BOLSTERS",)),
    "seat_belt_pretensioners": ("Seat belt pretensioners", ("SEAT_BELT_PRETENSIONER",)),
    "seat_belt_load_limiters": ("Seat belt load limiters", ("LOAD_LIMITERS",)),
    "automatic_crash_notification": ("Automatic crash notification", ("AUTO_CRASH_NOTIFICATION",)),
    "automatic_door_locks": ("Automatic door locks", ("ADL",)),
}

# NHTSA performance evaluations of crash-avoidance tech
EVALUATIONS: dict[str, tuple[str, str]] = {
    "forward_collision_warning": ("Forward collision warning test", "NHTSA_FCW_EVALUATION"),
    "lane_departure_warning": ("Lane departure warning test", "NHTSA_LDW_EVALUATION"),
    "automatic_emergency_braking": ("Crash imminent braking test", "NHTSA_CIB_EVALUATION"),
    "dynamic_brake_support": ("Dynamic brake support test", "NHTSA_DBS_EVALUATION"),
}

RATINGS: dict[str, tuple[str, str]] = {
    "overall_stars": ("Overall rating", "OVERALL_STARS"),
    "frontal_stars": ("Frontal crash rating", "OVERALL_FRNT_STARS"),
    "side_stars": ("Side crash rating", "OVERALL_SIDE_STARS"),
    "rollover_stars": ("Rollover rating", "ROLLOVER_STARS"),
    "side_pole_stars": ("Side pole crash rating", "SIDE_POLE_STARS"),
}

CONCERN_COLUMNS = (
    "FRNT_SAFETY_CONCERN_DRIV", "FRNT_SAFETY_CONCERN_PASS", "SIDE_SAFETY_CONCERN_DRIV",
    "SIDE_SAFETY_CONCERN_PASS", "POLE_SAFETY_CONCERN", "ROLL_SAFETY_CONCERN",
    "FRNT_FOOT_NOTES", "SIDE_FOOT_NOTES", "POLE_FOOT_NOTES", "ROLL_FOOT_NOTES",
)

MAKE_CANONICAL = {"ALFA": "ALFA ROMEO"}

_GENERIC_PREFIXES = {"ALL-NEW", "NEW", "THE"}

# First tokens that are too generic to be a model family on their own.
_TWO_TOKEN_FAMILY_PREFIXES = {
    "ACTIVE",
    "MODEL", "GRAND", "SANTA", "RANGE", "TOWN", "CROWN", "MONTE", "LAND", "NEW",
    "PT", "GRAN", "SPACE", "SUPER", "CLUB", "LE", "EL", "DE", "MARK", "TRANSIT",
}

BODY_STYLE_LABELS = {
    "4 DR": "4-door", "4-DR": "4-door", "2 DR": "2-door", "2-DR": "2-door", "C": "coupe",
    "CV": "convertible", "SUV": "SUV", "PU/CC": "crew cab pickup", "PU/EC": "extended cab pickup",
    "PU/RC": "regular cab pickup", "5 HB": "5-door hatchback", "3 HB": "3-door hatchback",
    "SW": "station wagon", "VAN": "van", "PV": "passenger van", "MV": "minivan",
    "5HB": "5-door hatchback", "3HB": "3-door hatchback", "4 DR SUV": "SUV", "2 DR SUV": "2-door SUV",
    "4 DR WAGON": "wagon", "WAGON": "wagon", "CONVERTIBLE": "convertible", "3 DR": "3-door",
}


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = " ".join(str(value).split())
    return text or None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def normalize_availability(raw: Any) -> str:
    """Map the many raw spellings to an ``Availability`` value."""
    text = (_clean(raw) or "").lower()
    if not text or text in {"n/a", "tbd", "-"}:
        return Availability.UNKNOWN.value
    if text in {"no", "n", "none", "not available", "na"} or text.startswith("not avail"):
        return Availability.NOT_AVAILABLE.value
    if text in {"s", "std", "standard", "yes", "y", "sf", "sf/sr", "dual"} or text.startswith("standard"):
        if "optional" in text:
            return Availability.VARIES.value
        return Availability.STANDARD.value
    if text in {"o", "opt", "optional", "a", "avl", "available"} or "avail" in text or text.startswith("optional"):
        return Availability.OPTIONAL.value
    if text == "both" or ("standard" in text and ("none" in text or "optional" in text)):
        return Availability.VARIES.value
    if "protection" in text:  # e.g. "Head Protection" = present
        return Availability.STANDARD.value
    return Availability.UNKNOWN.value


def model_family(make: str, model: str) -> str:
    """'F-150 SUPERCAB' -> 'F-150'; 'GRAND CHEROKEE L' -> 'GRAND CHEROKEE';
    'ALL-NEW Q5' -> 'Q5'; 'A4/S4 AVANT' -> 'A4'; '3 SERIES GT' -> '3 SERIES'."""
    full = " ".join(model.upper().replace("(", " (").split())
    text = full
    for prefix in {make, make.replace("-", " ")}:
        if text.startswith(prefix + " "):
            rest = text[len(prefix) + 1:]
            # 'MAZDA 2' / 'POLESTAR 2': the make is part of the model name
            if rest.split()[0].isdigit() and len(rest.split()[0]) <= 2:
                return f"{prefix} {rest.split()[0]}"
            text = rest
    tokens = [t.strip(",;") for t in text.split() if not t.startswith(("(", '"'))]
    while len(tokens) > 1 and tokens[0] in _GENERIC_PREFIXES:
        tokens = tokens[1:]
    # Porsche-style internal codes: '982 718 BOXSTER' -> '718 BOXSTER'
    if len(tokens) > 1 and re.fullmatch(r"\d{3}(\.\d)?", tokens[0]) and tokens[1][:1].isdigit():
        tokens = tokens[1:]
    tokens = [t.rstrip("/") for t in tokens if t.rstrip("/")]
    if not tokens:
        return full
    first = tokens[0]
    if "/" in first and len(first.split("/")[0]) >= 2:
        first = first.split("/")[0]
    if len(tokens) > 1 and (first in _TWO_TOKEN_FAMILY_PREFIXES or len(first) <= 1):
        return f"{first} {tokens[1]}"
    return first


def body_label(body_style: str | None) -> str:
    raw = " ".join((body_style or "").upper().split())
    release = ""
    for marker in ("EARLY RELEASE", "LATER RELEASE"):
        if marker in raw:
            release = f" ({marker.lower()})"
            raw = raw.replace(f"({marker})", "").replace(marker, "").strip()
    raw = raw.replace("W/ SAB", "W/SAB").replace("W/SAB", "").strip(" -")
    key = raw.replace("-", " ").replace("4DR", "4 DR").replace("2DR", "2 DR").strip()
    label = BODY_STYLE_LABELS.get(key) or BODY_STYLE_LABELS.get(raw) or raw.lower()
    return f"{label}{release}".strip()


def variant_label(model: str, body_style: str | None, drive_train: str | None) -> str:
    parts = [model] + [p for p in (body_label(body_style), drive_train) if p]
    return " · ".join(parts)


def row_to_vehicle(row_id: int, row: dict[str, Any]) -> Vehicle:
    make = (_clean(row.get("MAKE")) or "").upper()
    make = MAKE_CANONICAL.get(make, make)
    model = (_clean(row.get("MODEL")) or "").upper()
    body = _clean(row.get("BODY_STYLE"))
    drive = (_clean(row.get("DRIVE_TRAIN")) or "").upper() or None

    features: dict[str, str] = {}
    for key, (_label, columns) in FEATURES.items():
        values = [normalize_availability(row.get(c)) for c in columns]
        known = [v for v in values if v != Availability.UNKNOWN.value]
        features[key] = known[0] if known else Availability.UNKNOWN.value

    evaluations = {
        key: value
        for key, (_label, column) in EVALUATIONS.items()
        if (value := _clean(row.get(column)))
    }
    concerns: list[str] = []
    for column in CONCERN_COLUMNS:
        text = _clean(row.get(column))
        if text and text not in concerns:
            concerns.append(text)

    rollover_risk = _num(row.get("ROLLOVER_POSSIBILITY"))
    if rollover_risk is not None and rollover_risk <= 1:
        rollover_risk *= 100

    raw = {k: v for k, v in ((k, _clean(v)) for k, v in row.items()) if v is not None}

    return Vehicle(
        id=row_id,
        make=make,
        model=model,
        model_family=model_family(make, model),
        model_year=int(row["MODEL_YR"]),
        body_style=body,
        drive_train=drive,
        vehicle_type=_clean(row.get("VEHICLE_TYPE")),
        vehicle_class=_clean(row.get("VEHICLE_CLASS")),
        production_release=int(_num(row.get("PRODUCTION_RELEASE")) or 1),
        variant_label=variant_label(model, body, drive),
        overall_stars=_num(row.get("OVERALL_STARS")),
        frontal_stars=_num(row.get("OVERALL_FRNT_STARS")),
        side_stars=_num(row.get("OVERALL_SIDE_STARS")),
        rollover_stars=_num(row.get("ROLLOVER_STARS")),
        side_pole_stars=_num(row.get("SIDE_POLE_STARS")),
        rollover_risk_pct=round(rollover_risk, 1) if rollover_risk is not None else None,
        features=features,
        evaluations=evaluations,
        safety_concerns=concerns,
        raw=raw,
    )


def read_safercar_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, keep_default_na=True)
    df = df[df["MAKE"].notna() & df["MODEL"].notna() & df["MODEL_YR"].notna()]
    return df


def load_safercar(session: Session, csv_path: Path | None = None, replace: bool = True) -> int:
    """Load the CSV into ``vehicles``. Returns the number of rows loaded."""
    csv_path = Path(csv_path or get_settings().safercar_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Put Safercar_data.csv there or set SAFERCAR_CSV."
        )
    df = read_safercar_csv(csv_path)
    if replace:
        session.execute(delete(Vehicle))
    vehicles = [row_to_vehicle(i + 1, row) for i, row in enumerate(df.to_dict("records"))]
    session.add_all(vehicles)
    session.commit()
    log.info("Loaded %d vehicles from %s", len(vehicles), csv_path)
    return len(vehicles)


# ------------------------------------------------------------------ evidence

def to_record(vehicle: Vehicle) -> VehicleRecord:
    return VehicleRecord(
        id=vehicle.id,
        make=vehicle.make,
        model=vehicle.model,
        model_family=vehicle.model_family,
        model_year=vehicle.model_year,
        body_style=vehicle.body_style,
        drive_train=vehicle.drive_train,
        vehicle_type=vehicle.vehicle_type,
        variant_label=vehicle.variant_label,
        overall_stars=vehicle.overall_stars,
        frontal_stars=vehicle.frontal_stars,
        side_stars=vehicle.side_stars,
        rollover_stars=vehicle.rollover_stars,
        side_pole_stars=vehicle.side_pole_stars,
        rollover_risk_pct=vehicle.rollover_risk_pct,
        features=vehicle.features or {},
        evaluations=vehicle.evaluations or {},
        safety_concerns=vehicle.safety_concerns or [],
    )


def _stars(value: float | None) -> str:
    return "not rated" if value is None else f"{value:g} stars"


AVAILABILITY_TEXT = {
    "standard": "standard",
    "optional": "optional",
    "varies": "standard on some trims, optional on others",
    "not_available": "not available",
    "unknown": "not reported",
}


def answer_fields(features: list[str]) -> list[str]:
    """Which VehicleRecord fields a question depends on (for clarification)."""
    if features:
        return [f"features.{f}" for f in features] + [f"evaluations.{f}" for f in features if f in EVALUATIONS]
    return list(RATINGS)


def field_value(record: VehicleRecord, field: str) -> Any:
    if field.startswith("features."):
        return record.features.get(field.split(".", 1)[1])
    if field.startswith("evaluations."):
        return record.evaluations.get(field.split(".", 1)[1])
    return getattr(record, field, None)


def vehicle_evidence_text(record: VehicleRecord, features: list[str] | None = None) -> str:
    """Render one vehicle variant as plain-text facts for the LLM."""
    lines = [
        f"Vehicle: {record.model_year} {record.make} {record.model}",
        f"Variant: {record.variant_label}",
        "Source: NHTSA 5-Star Safety Ratings data",
    ]
    for key, (label, _col) in RATINGS.items():
        lines.append(f"{label}: {_stars(getattr(record, key))}")
    if record.rollover_risk_pct is not None:
        lines.append(f"Rollover risk: {record.rollover_risk_pct:g}%")
    keys = list(features or []) + [k for k in FEATURES if k not in (features or [])]
    for key in keys:
        if key not in FEATURES:
            continue
        status = record.features.get(key, "unknown")
        if features and key not in features and status == "unknown":
            continue
        lines.append(f"{FEATURES[key][0]}: {AVAILABILITY_TEXT.get(status, status)}")
        if key in record.evaluations:
            lines.append(f"{EVALUATIONS[key][0]}: {record.evaluations[key]}")
    for concern in record.safety_concerns[:3]:
        lines.append(f"Safety concern noted by NHTSA: {concern}")
    return "\n".join(lines)


def vehicle_evidence(record: VehicleRecord, ref: int, features: list[str] | None = None) -> Evidence:
    return Evidence(
        ref=ref,
        evidence_id=f"vehicle:{record.id}",
        kind="vehicle_record",
        title=f"NHTSA 5-Star Safety Ratings: {record.model_year} {record.make} {record.model} ({record.variant_label})",
        url=vehicle_citation_url(record),
        text=vehicle_evidence_text(record, features),
    )


def vehicle_citation_url(record: VehicleRecord) -> str:
    """Citation link for a ratings row.

    TODO(WP-08): verify NHTSA's per-vehicle page pattern and return a deep
    link. Until then we cite the official ratings page, which is where the
    dataset is published.
    """
    return get_settings().ratings_citation_url


def compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())
