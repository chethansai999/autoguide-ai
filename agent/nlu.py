"""Natural-language understanding: intent + vehicle slots + features.

Three interchangeable backends behind one interface (``NLU_BACKEND``):

    rules  Gazetteer + regex baseline. No training, no network. Default.
    hf     Fine-tuned DistilBERT intent classifier + BIO slot tagger
           (finetune/train_nlu.py -> models/clarifier/{intent,slots}).
    llm    Prompted LLM returning JSON (or an OpenAI fine-tuned model id
           via OPENAI_MODEL). Uses core.llm.

All return ``NLUResult``; all slot values are canonicalised through the
vehicle vocabulary so downstream code never sees raw spellings.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Protocol

from core.config import Settings, get_settings
from core.schemas import Intent, NLUResult, VehicleSlots
from agent.lexicon import (
    BODY_PATTERNS,
    CAPABILITY,
    COMPARE_CUE,
    COMPARE_SPLIT,
    DOMAIN_WORDS,
    DRIVE_PATTERNS,
    FUNCTION_WORDS,
    OUT_OF_SCOPE,
    POSSESSIVE_VEHICLE,
    RATING_WORDS,
    RECALL_WORDS,
    SMALLTALK,
    SOURCES_QUESTION,
    VEHICLE_INFO_WORDS,
    find_features,
)
from structured.safercar import compact
from structured.vocab import AMBIGUOUS_MAKES, COMMON_WORD_MODELS, VehicleVocab

log = logging.getLogger(__name__)

TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-.'][A-Za-z0-9]+)*")
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
SHORT_YEAR = re.compile(r"(?:^|\s)'(\d{2})\b")
VIN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


class NLUModel(Protocol):
    backend: str

    def parse(self, text: str, history: list[dict] | None = None) -> NLUResult: ...


# --------------------------------------------------------------------------
# slot helpers (shared by all backends)
# --------------------------------------------------------------------------

def extract_year(text: str, vocab: VehicleVocab | None = None) -> int | None:
    match = YEAR.search(text)
    if match:
        return int(match.group(1))
    match = SHORT_YEAR.search(text)
    if match:
        two = int(match.group(1))
        cutoff = (vocab.max_year if vocab else 2030) % 100
        return 2000 + two if two <= cutoff else 1900 + two
    return None


def extract_drive(text: str) -> str | None:
    for pattern, value in DRIVE_PATTERNS:
        if pattern.search(text):
            return value
    return None


def extract_body(text: str) -> str | None:
    for pattern, value in BODY_PATTERNS:
        if pattern.search(text):
            return value
    return None


DETERMINERS = frozenset("my our your his her their the a an this that new used".split())

# Words that never identify a model on their own, so they are excluded from
# fuzzy matching ("Tesla Model Z" is not "Model S").
GENERIC_WORDS = frozenset("""
model models version versions trim edition series class type kind make brand new old used
sedan coupe wagon hatchback convertible truck pickup van minivan suv crossover hybrid electric
diesel premium limited sport touring base standard luxury
""".split())


def extract_make_model(text: str, vocab: VehicleVocab) -> tuple[str | None, str | None]:
    """Longest-match gazetteer scan with false-positive guards."""
    tokens = [(m.group(0), m.start()) for m in TOKEN.finditer(text)]
    used: set[int] = set()
    make: str | None = None
    # Typo'd makes ("totyata", "toyoto") get a looser match only in a message
    # that is plainly about a car: a model year or a rating word.
    make_cutoff = 75 if (extract_year(text) is not None or RATING_WORDS.search(text)) else 85

    for n in (3, 2, 1):  # makes: "land rover", "mercedes benz", "chevy"
        for i in range(len(tokens) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            surface = " ".join(t for t, _ in tokens[i:i + n])
            candidate = vocab.canonical_make(surface, fuzzy=(n == 1 and len(surface) >= 5), cutoff=make_cutoff)
            if not candidate:
                continue
            if candidate in AMBIGUOUS_MAKES and not surface[:1].isupper():
                # accept "ram 1500" / "mini cooper": next token is a model of that make
                following = tokens[i + n][0] if i + n < len(tokens) else ""
                if not following or not vocab.models_matching(following, make=candidate, fuzzy=False):
                    continue
            make = candidate
            used.update(range(i, i + n))
            break
        if make:
            break

    model: str | None = None
    best_len = 0
    for n in (4, 3, 2, 1):
        for i in range(len(tokens) - n + 1):
            span = range(i, i + n)
            if any(j in used for j in span):
                continue
            surface = " ".join(t for t, _ in tokens[i:i + n])
            key = compact(surface)
            if not key or YEAR.fullmatch(surface) or len(key) < 2:
                continue
            hits = vocab.models_matching(surface, make=make, fuzzy=False)
            if not hits:
                continue
            if not make:
                if n == 1 and surface.lower() in FUNCTION_WORDS:  # "Is my car safe?"
                    continue
                if n > 1 and tokens[i][0].lower() in FUNCTION_WORDS:  # "a 5 star rating" is not the Audi A5
                    continue
                # Model names that are also English words ("civic", "focus") need
                # some sign of a car: a capital mid-sentence, a model year, or a
                # safety word ("is my civic safe?"). A sentence-start capital alone
                # doesn't count.
                sentence_start = i == 0 or text[:tokens[i][1]].rstrip().endswith((".", "!", "?"))
                before = tokens[i - 1][0].lower() if i > 0 else ""
                after = tokens[i + n][0] if i + n < len(tokens) else ""
                car_context = bool(
                    extract_year(text) is not None                      # "accord 2018", "my '19 civic"
                    or before in DETERMINERS                            # "my civic", "the focus"
                    or RATING_WORDS.fullmatch(after) or DOMAIN_WORDS.fullmatch(after)  # "civic safety rating"
                )
                capitalized = surface[:1].isupper() and not sentence_start
                if key in COMMON_WORD_MODELS and not (capitalized or car_context):
                    continue
                if key.isdigit():  # "300", "500" need an explicit make
                    continue
            if len(key) > best_len:
                makes = {m for m, _ in hits}
                display = sorted(d for _, d in hits)[0]
                model, best_len = display, len(key)
                if not make and len(makes) == 1:
                    make = makes.pop()
        if model:
            break

    if make and not model:  # fuzzy model match within the make ("camery")
        for surface, _ in tokens:
            generic = surface.lower() in GENERIC_WORDS or DOMAIN_WORDS.fullmatch(surface) or RATING_WORDS.fullmatch(surface)
            if generic:  # "Model Z" must not fuzzy-match "Model S"
                continue
            if len(surface) >= 5 and not YEAR.fullmatch(surface) and compact(surface) != compact(make):
                hits = vocab.models_matching(surface, make=make, fuzzy=True)
                if hits:
                    model = hits[0][1]
                    break
    return make, model


def extract_slots(text: str, vocab: VehicleVocab) -> VehicleSlots:
    make, model = extract_make_model(text, vocab)
    vin = VIN.search(text.upper())
    return VehicleSlots(
        make=make,
        model=model,
        model_year=extract_year(text, vocab),
        body_style=extract_body(text),
        drive_train=extract_drive(text),
        vin=vin.group(1) if vin else None,
    )


# --------------------------------------------------------------------------
# rules backend
# --------------------------------------------------------------------------

def extract_vehicle_pair(text: str, vocab: VehicleVocab) -> list[VehicleSlots] | None:
    """Two vehicles to compare ("2020 Camry or 2020 Accord, which is safer?").

    Returns None unless the message has a comparison cue and two parts that
    each name a make or model. A year given once is shared by both.
    """
    if not COMPARE_CUE.search(text):
        return None
    found = [extract_slots(part, vocab) for part in COMPARE_SPLIT.split(text)]
    vehicles = [s for s in found if s.make or s.model]
    if len(vehicles) < 2:
        return None
    first, second = vehicles[0], vehicles[1]
    if first.model_year and not second.model_year:
        second = second.model_copy(update={"model_year": first.model_year})
    elif second.model_year and not first.model_year:
        first = first.model_copy(update={"model_year": second.model_year})
    if (first.make, first.model, first.model_year) == (second.make, second.model, second.model_year):
        return None
    return [first, second]


def is_capability_question(text: str, slots: VehicleSlots | None = None, features: list[str] | None = None) -> bool:
    """'Can I ask you anything?' with no vehicle, feature or safety topic in it."""
    if SOURCES_QUESTION.search(text) and (slots is None or slots.is_empty()):
        return True
    if not CAPABILITY.search(text) or (slots is not None and not slots.is_empty()):
        return False
    return not (features or find_features(text) or DOMAIN_WORDS.search(text) or RECALL_WORDS.search(text)
                or VEHICLE_INFO_WORDS.search(text))


class RuleNLU:
    """Baseline the fine-tuned model must beat. Transparent and testable."""

    backend = "rules"

    def __init__(self, vocab: VehicleVocab):
        self.vocab = vocab

    def classify_intent(self, text: str, slots: VehicleSlots, features: list[str]) -> tuple[Intent, float]:
        has_vehicle = not slots.is_empty() or bool(POSSESSIVE_VEHICLE.search(text))
        if SMALLTALK.search(text) and slots.is_empty():
            return Intent.SMALLTALK, 0.95
        if is_capability_question(text, slots, features):
            return Intent.SMALLTALK, 0.9
        if OUT_OF_SCOPE.search(text):
            return Intent.OUT_OF_SCOPE, 0.8
        if RECALL_WORDS.search(text):
            return (Intent.RECALL_LOOKUP, 0.85) if has_vehicle else (Intent.CONCEPT, 0.7)
        if has_vehicle and (features or RATING_WORDS.search(text)):
            return Intent.VEHICLE_LOOKUP, 0.85
        if VEHICLE_INFO_WORDS.search(text):
            return (Intent.VEHICLE_INFO, 0.85) if has_vehicle else (Intent.CONCEPT, 0.75)
        if (slots.make or slots.model) and re.search(r"\b(tell me about|information about|info on)\b", text, re.I):
            return Intent.VEHICLE_INFO, 0.85
        if slots.make or slots.model:
            return Intent.VEHICLE_LOOKUP, 0.6
        if features or DOMAIN_WORDS.search(text):
            return Intent.CONCEPT, 0.75
        return Intent.OUT_OF_SCOPE, 0.55

    def parse(self, text: str, history: list[dict] | None = None) -> NLUResult:
        slots = extract_slots(text, self.vocab)
        features = find_features(text)
        intent, confidence = self.classify_intent(text, slots, features)
        return NLUResult(
            intent=intent,
            intent_confidence=confidence,
            slots=slots,
            features=features,
            backend=self.backend,
        )


# --------------------------------------------------------------------------
# fine-tuned Hugging Face backend
# --------------------------------------------------------------------------

class HFNLU:
    """Loads the models produced by ``finetune/train_nlu.py``.

    models/clarifier/intent  AutoModelForSequenceClassification
    models/clarifier/slots   AutoModelForTokenClassification (BIO tags)

    Predicted spans are canonicalised through the vocabulary, so a tagged
    "camery" still becomes model="CAMRY". Falls back to regex for years.
    """

    backend = "hf"

    def __init__(self, vocab: VehicleVocab, model_dir: Path):
        import torch  # noqa: F401  (requirements-finetune.txt)
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        self.vocab = vocab
        self.intent_tok = AutoTokenizer.from_pretrained(model_dir / "intent")
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(model_dir / "intent").eval()
        self.slot_tok = AutoTokenizer.from_pretrained(model_dir / "slots")
        self.slot_model = AutoModelForTokenClassification.from_pretrained(model_dir / "slots").eval()

    def _intent(self, text: str) -> tuple[Intent, float]:
        import torch

        inputs = self.intent_tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            probs = self.intent_model(**inputs).logits.softmax(-1)[0]
        idx = int(probs.argmax())
        label = self.intent_model.config.id2label[idx]
        return Intent(label), float(probs[idx])

    def _spans(self, text: str) -> dict[str, str]:
        import torch

        words = TOKEN.findall(text)
        if not words:
            return {}
        enc = self.slot_tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            pred = self.slot_model(**enc).logits.argmax(-1)[0].tolist()
        labels: list[str] = ["O"] * len(words)
        seen: set[int] = set()
        for pos, word_id in enumerate(enc.word_ids()):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            labels[word_id] = self.slot_model.config.id2label[pred[pos]]
        spans: dict[str, list[str]] = {}
        current = None
        for word, tag in zip(words, labels):
            if tag.startswith("B-"):
                current = tag[2:]
                spans.setdefault(current, []).append(word)
            elif tag.startswith("I-") and current == tag[2:]:
                spans[current][-1] += " " + word
            else:
                current = None
        return {k: v[0] for k, v in spans.items()}

    def parse(self, text: str, history: list[dict] | None = None) -> NLUResult:
        if SMALLTALK.search(text) or is_capability_question(text):  # tiny class in training data; rules are exact here
            return NLUResult(intent=Intent.SMALLTALK, intent_confidence=0.95, backend=self.backend)
        intent, confidence = self._intent(text)
        spans = self._spans(text)
        make = self.vocab.canonical_make(spans.get("MAKE")) if spans.get("MAKE") else None
        model = None
        if spans.get("MODEL"):
            hits = self.vocab.models_matching(spans["MODEL"], make=make)
            if hits:
                model = hits[0][1]
                if not make and len({m for m, _ in hits}) == 1:
                    make = hits[0][0]
        year_text = spans.get("YEAR", "")
        if year_text.isdigit() and len(year_text) == 2:
            year_text = "'" + year_text
        slots = VehicleSlots(
            make=make,
            model=model,
            model_year=extract_year(year_text or text, self.vocab),
            body_style=extract_body(spans.get("BODY", "")) if spans.get("BODY") else extract_body(text),
            drive_train=extract_drive(spans.get("DRIVE", "")) if spans.get("DRIVE") else extract_drive(text),
        )
        return NLUResult(
            intent=intent,
            intent_confidence=confidence,
            slots=slots,
            features=find_features(text),
            backend=self.backend,
        )


# --------------------------------------------------------------------------
# prompted / OpenAI-fine-tuned LLM backend
# --------------------------------------------------------------------------

LLM_NLU_SYSTEM = """You extract structured information from a user's message to a vehicle-safety assistant.
Return JSON with keys:
  intent: one of concept_question | vehicle_lookup | recall_lookup | vehicle_info | out_of_scope | smalltalk
  make, model, model_year, body_style, drive_train: the vehicle the user refers to, or null
  features: list of safety features mentioned (e.g. "automatic_emergency_braking")
Rules: never guess a vehicle that is not mentioned; "my car" alone means make/model/year are null.
concept_question = how/what/why about safety technology or NHTSA programs, not a specific vehicle.
vehicle_lookup = ratings or features of a specific vehicle. recall_lookup = recalls for a vehicle.
vehicle_info = a specific vehicle's engine, specifications, history, fuel economy or emissions.
concept_question also covers general vehicle technology and fuel efficiency questions.
out_of_scope = prices, repair costs, insurance, or anything unrelated to vehicles."""


class LLMNLU:
    backend = "llm"

    def __init__(self, vocab: VehicleVocab, llm=None, settings: Settings | None = None):
        from core.llm import OpenAIClient, get_llm

        settings = settings or get_settings()
        self.vocab = vocab
        if llm is None and settings.nlu_llm_model and settings.resolved_llm_provider() == "openai":
            llm = OpenAIClient(settings)
            llm.model = settings.nlu_llm_model  # e.g. your OpenAI fine-tune
        self.llm = llm or get_llm()
        self.fallback = RuleNLU(vocab)

    def parse(self, text: str, history: list[dict] | None = None) -> NLUResult:
        result = self.llm.complete(LLM_NLU_SYSTEM, [{"role": "user", "content": text}], json_mode=True, max_tokens=200)
        try:
            data = json.loads(result.text)
            intent = Intent(data["intent"])
        except (ValueError, KeyError, TypeError):
            return self.fallback.parse(text, history)
        make = self.vocab.canonical_make(data.get("make")) if data.get("make") else None
        model = None
        if data.get("model"):
            hits = self.vocab.models_matching(str(data["model"]), make=make)
            model = hits[0][1] if hits else str(data["model"]).upper()
            if hits and not make and len({m for m, _ in hits}) == 1:
                make = hits[0][0]
        year = data.get("model_year")
        slots = VehicleSlots(
            make=make,
            model=model,
            model_year=int(year) if str(year or "").isdigit() else extract_year(text, self.vocab),
            body_style=extract_body(text),
            drive_train=extract_drive(text),
        )
        features = [f for f in (data.get("features") or []) if isinstance(f, str)] or find_features(text)
        return NLUResult(intent=intent, intent_confidence=0.9, slots=slots, features=features, backend=self.backend)


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def build_nlu(vocab: VehicleVocab, settings: Settings | None = None) -> NLUModel:
    settings = settings or get_settings()
    if settings.nlu_backend == "hf":
        try:
            return HFNLU(vocab, settings.nlu_model_dir)
        except Exception as error:  # missing model dir or torch
            log.warning("HF NLU unavailable (%s); falling back to rules.", error)
    if settings.nlu_backend == "llm":
        return LLMNLU(vocab, settings=settings)
    return RuleNLU(vocab)
