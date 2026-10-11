from core.schemas import ResolutionStatus, VehicleSlots
from structured.resolver import VehicleResolver, model_matches


def resolver():
    from db.session import get_sessionmaker
    return VehicleResolver(get_sessionmaker())


def test_model_matches_token_prefix():
    assert model_matches("f150", "F-150 SUPERCAB")
    assert model_matches("CRV", "CR-V HYBRID")
    assert not model_matches("Cherokee", "GRAND CHEROKEE")


def test_ambiguous_variants(loaded):
    r = resolver().resolve(VehicleSlots(make="ford", model="F-150", model_year=2019))
    assert r.status == ResolutionStatus.AMBIGUOUS and len(r.vehicles) == 10


def test_drive_train_narrows_to_one(loaded):
    r = resolver().resolve(VehicleSlots(make="SUBARU", model="OUTBACK", model_year=2018, drive_train="AWD"))
    assert r.status == ResolutionStatus.RESOLVED


def test_missing_year_lists_years(loaded):
    r = resolver().resolve(VehicleSlots(make="FORD", model="FOCUS"))
    assert r.status == ResolutionStatus.MISSING and r.missing_slots == ["model_year"]
    assert r.options[0] == "2018"


def test_make_inferred_from_unique_model(loaded):
    r = resolver().resolve(VehicleSlots(model="CAMRY", model_year=2022))
    assert r.status in (ResolutionStatus.AMBIGUOUS, ResolutionStatus.RESOLVED)
    assert r.vehicles[0].make == "TOYOTA"


def test_year_not_in_data(loaded):
    r = resolver().resolve(VehicleSlots(make="FORD", model="F-150", model_year=2030))
    assert r.status == ResolutionStatus.NOT_FOUND and "2019" in r.options


def test_unknown_make(loaded):
    r = resolver().resolve(VehicleSlots(make="Zorblax", model="X"))
    assert r.status == ResolutionStatus.NOT_FOUND
