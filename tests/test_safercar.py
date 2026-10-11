import pytest

from structured.safercar import model_family, normalize_availability, to_record, vehicle_evidence_text


@pytest.mark.parametrize("raw,expected", [
    ("S", "standard"), ("Std", "standard"), ("STANDARD", "standard"), ("O", "optional"),
    ("Avl", "optional"), ("optional", "optional"), ("No", "not_available"), ("Not Available", "not_available"),
    ("Both", "varies"), ("None:BASE,TECH Standard:ADV", "varies"), (" ", "unknown"), (None, "unknown"),
])
def test_normalize_availability(raw, expected):
    assert normalize_availability(raw) == expected


@pytest.mark.parametrize("make,model,family", [
    ("FORD", "F-150 SUPERCAB", "F-150"), ("BMW", "3 SERIES GT", "3 SERIES"), ("TESLA", "MODEL S", "MODEL S"),
    ("JEEP", "GRAND CHEROKEE L", "GRAND CHEROKEE"), ("AUDI", "ALL-NEW Q5", "Q5"), ("MAZDA", "MAZDA 2", "MAZDA 2"),
    ("LAND ROVER", "LAND ROVER DISCOVERY", "DISCOVERY"), ("HONDA", "CR-V HYBRID", "CR-V"),
])
def test_model_family(make, model, family):
    assert model_family(make, model) == family


def test_loaded_rows_and_evidence_text(loaded):
    from sqlalchemy import func, select

    from db.models import Vehicle
    from db.session import get_sessionmaker

    with get_sessionmaker()() as db:
        assert db.scalar(select(func.count()).select_from(Vehicle)) == 39
        outback = db.scalars(select(Vehicle).where(Vehicle.model == "OUTBACK", Vehicle.model_year == 2018)).one()
        text = vehicle_evidence_text(to_record(outback), ["blind_spot_detection"])
    assert "Overall rating:" in text and "Blind spot detection:" in text
    from structured.safercar import FEATURES

    labels = {label for label, _cols in FEATURES.values()}
    feature_lines = [line for line in text.splitlines() if line.split(":")[0] in labels]
    assert feature_lines[0].startswith("Blind spot detection")  # the asked-about feature comes first
