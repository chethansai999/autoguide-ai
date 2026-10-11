import pytest

from agent.nlu import RuleNLU, extract_year
from core.schemas import Intent


@pytest.mark.parametrize("text,intent,slots", [
    ("Does the 2019 Ford F-150 have a backup camera?", Intent.VEHICLE_LOOKUP, {"make": "FORD", "model": "F-150", "model_year": 2019}),
    ("is my 2024 honda crv safe?", Intent.VEHICLE_LOOKUP, {"make": "HONDA", "model": "CR-V", "model_year": 2024}),
    ("Does my car have automatic emergency braking?", Intent.VEHICLE_LOOKUP, {}),
    ("Are there any recalls on my '20 honda accord?", Intent.RECALL_LOOKUP, {"make": "HONDA", "model": "ACCORD", "model_year": 2020}),
    ("How does automatic emergency braking work?", Intent.CONCEPT, {}),
    ("Where can I find information about vehicle recalls?", Intent.CONCEPT, {}),
    ("How much does it cost to replace an airbag?", Intent.OUT_OF_SCOPE, {}),
    ("thanks, that helps!", Intent.SMALLTALK, {}),
    ("What should I focus on when buying a safe car?", Intent.CONCEPT, {}),
    ("ram 1500 2020 side crash rating", Intent.VEHICLE_LOOKUP, {"make": "RAM", "model": "1500", "model_year": 2020}),
])
def test_rule_nlu(vocab, text, intent, slots):
    result = RuleNLU(vocab).parse(text)
    assert result.intent == intent
    got = result.slots.model_dump(exclude_none=True)
    for key, value in slots.items():
        assert got.get(key) == value, (key, got)
    if not slots:
        assert not (got.get("make") or got.get("model"))


def test_short_year():
    assert extract_year("my '19 civic") == 2019
    assert extract_year("a '98 camry") == 1998


def test_function_words_and_sentence_start_capitals_are_not_models():
    """'Is my car safe?' was read as the Lexus IS."""
    from agent.nlu import extract_make_model
    from structured.vocab import VehicleVocab

    vocab = VehicleVocab.from_rows([
        ("LEXUS", "IS 350", "IS", 2019), ("FORD", "FOCUS", "FOCUS", 2018), ("FORD", "ESCAPE", "ESCAPE", 2020),
    ])
    assert extract_make_model("Is my car safe?", vocab) == (None, None)
    assert extract_make_model("Focus on the road", vocab) == (None, None)
    assert extract_make_model("Is the 2019 Lexus IS 350 safe?", vocab)[0] == "LEXUS"
    assert extract_make_model("Is the Focus safe?", vocab) == ("FORD", "FOCUS")
    assert extract_make_model("Escape rating 2020", vocab) == ("FORD", "ESCAPE")


def test_unknown_model_is_not_fuzzy_matched_to_a_real_one():
    """'Tesla Model Z' was answered as the Model S."""
    from agent.nlu import extract_make_model
    from structured.vocab import VehicleVocab

    vocab = VehicleVocab.from_rows([("TESLA", "MODEL S", "MODEL S", 2019), ("TESLA", "MODEL 3", "MODEL 3", 2019),
                                    ("HONDA", "CIVIC", "CIVIC", 2019), ("TOYOTA", "CAMRY", "CAMRY", 2018)])
    assert extract_make_model("Is the 2019 Tesla Model Z safe?", vocab) == ("TESLA", None)
    assert extract_make_model("2019 tesla model 3 rating", vocab) == ("TESLA", "MODEL 3")
    assert extract_make_model("is the 2018 toyota camery safe", vocab) == ("TOYOTA", "CAMRY")  # real typos still fixed
    assert extract_make_model("my '19 civic, is it safe?", vocab) == ("HONDA", "CIVIC")


def test_vehicle_pair_shares_the_year():
    from agent.nlu import extract_vehicle_pair
    from structured.vocab import VehicleVocab

    vocab = VehicleVocab.from_rows([("TOYOTA", "CAMRY", "CAMRY", 2020), ("HONDA", "ACCORD", "ACCORD", 2020)])
    pair = extract_vehicle_pair("Which is safer, the 2020 Toyota Camry or the Honda Accord?", vocab)
    assert [(s.make, s.model, s.model_year) for s in pair] == [("TOYOTA", "CAMRY", 2020), ("HONDA", "ACCORD", 2020)]
    assert extract_vehicle_pair("Is the 2020 Toyota Camry safe?", vocab) is None
    assert extract_vehicle_pair("Does the 2020 Camry have blind spot and lane departure warning?", vocab) is None


def test_typo_makes_and_false_friends():
    from agent.nlu import extract_make_model
    from structured.vocab import VehicleVocab

    vocab = VehicleVocab.from_rows([("TOYOTA", "COROLLA", "COROLLA", 2020), ("AUDI", "A5", "A5", 2020),
                                    ("HONDA", "CIVIC", "CIVIC", 2020)])
    assert extract_make_model("give me totyata sfety rating", vocab)[0] == "TOYOTA"   # typo + car context
    assert extract_make_model("is the toyoto corolla safe", vocab) == ("TOYOTA", "COROLLA")
    assert extract_make_model("How do I audit my car's recalls?", vocab) == (None, None)
    assert extract_make_model("what does a 5 star rating mean", vocab) == (None, None)
    assert extract_make_model("2020 audi a5 rating", vocab) == ("AUDI", "A5")


@pytest.mark.parametrize("text", [
    "How fast is the IIHS small overlap test?",
    "What does a marginal rating mean in the side impact test?",
    "what dummy is used in the side impact test",
    "what is submarining",
])
def test_crash_test_questions_are_concept_questions(vocab, text):
    assert RuleNLU(vocab).parse(text).intent == Intent.CONCEPT
