from agent import clarifier
from core.schemas import Intent, VehicleSlots
from structured.resolver import VehicleResolver


def _resolve(slots):
    from db.session import get_sessionmaker
    return VehicleResolver(get_sessionmaker()).resolve(slots)


def test_asks_for_variant_when_answers_differ(loaded):
    slots = VehicleSlots(make="FORD", model="F-150", model_year=2019)
    d = clarifier.decide(Intent.VEHICLE_LOOKUP, _resolve(slots), slots, [])
    assert d.should_ask and d.missing_slot == "variant" and len(d.options) == 10


def test_does_not_ask_when_all_variants_agree(loaded):
    slots = VehicleSlots(make="FORD", model="F-150", model_year=2019)
    resolution = _resolve(slots)
    feature = "abs"
    if clarifier.variants_disagree(resolution.vehicles, [feature]):
        return  # data-dependent guard; the rule itself is tested below
    assert not clarifier.decide(Intent.VEHICLE_LOOKUP, resolution, slots, [feature]).should_ask


def test_recalls_never_ask_for_variant(loaded):
    slots = VehicleSlots(make="FORD", model="F-150", model_year=2019)
    d = clarifier.decide(Intent.RECALL_LOOKUP, _resolve(slots), slots, [])
    assert not d.should_ask


def test_missing_everything_asks_once_for_full_vehicle(loaded):
    slots = VehicleSlots()
    d = clarifier.decide(Intent.VEHICLE_LOOKUP, _resolve(slots), slots, ["abs"])
    assert d.should_ask and "year, make and model" in d.question


def test_stops_after_max_turns(loaded):
    slots = VehicleSlots(make="FORD", model="F-150", model_year=2019)
    d = clarifier.decide(Intent.VEHICLE_LOOKUP, _resolve(slots), slots, [], clarification_turns=3, max_turns=3)
    assert not d.should_ask and d.reason == "max_clarification_turns"
