"""Generate labelled NLU training data from the Safercar vehicles table.

Why synthetic: the CSV holds vehicles, not conversations. We turn it into
thousands of realistic user messages whose labels are known by construction:
intent, BIO slot tags (MAKE/MODEL/YEAR/BODY/DRIVE/FEATURE) and canonical
slot values. Some messages deliberately omit the year / model / make, which
is what teaches the agent when information is missing.

Leakage control: vehicle examples are split by (make, model family) and
non-vehicle examples by template, so the TEST split contains models and
phrasings never seen in training. Synthetic test scores still overstate
real-world accuracy - always also report evaluation/data/nlu_realistic.jsonl.

    python -m finetune.synth_data --n 8000
    -> data/finetune/nlu_{train,dev,test}.jsonl
       data/finetune/openai_{train,dev}.jsonl   (chat format for OpenAI fine-tuning)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.lexicon import FEATURE_SYNONYMS
from agent.nlu import LLM_NLU_SYSTEM, TOKEN
from core.config import get_settings
from structured.vocab import MAKE_ALIASES, VehicleVocab

INTENTS = ["concept_question", "vehicle_lookup", "recall_lookup", "vehicle_info", "out_of_scope", "smalltalk"]
SLOT_TYPES = ["MAKE", "MODEL", "YEAR", "BODY", "DRIVE", "FEATURE"]
TAGS = ["O"] + [f"{p}-{t}" for t in SLOT_TYPES for p in ("B", "I")]

DRIVE_SURFACES = {
    "AWD": ["AWD", "awd", "all-wheel drive", "all wheel drive"],
    "FWD": ["FWD", "fwd", "front-wheel drive"],
    "RWD": ["RWD", "rwd", "rear-wheel drive"],
    "4WD": ["4WD", "4x4", "4wd", "four-wheel drive"],
    "2WD": ["2WD", "4x2", "2wd"],
}
BODY_SURFACES = {
    "SEDAN": ["sedan", "4-door"], "COUPE": ["coupe", "2-door"], "HATCHBACK": ["hatchback", "hatch"],
    "CREWCAB": ["crew cab", "supercrew", "double cab"], "EXTENDEDCAB": ["extended cab", "supercab"],
    "REGULARCAB": ["regular cab"], "CONVERTIBLE": ["convertible"], "WAGON": ["wagon"], "MINIVAN": ["minivan"],
}
RATING_PHRASES = ["safety rating", "crash test rating", "overall rating", "rollover rating", "side crash rating",
                  "frontal crash rating", "star rating", "NCAP rating"]

# {V} = vehicle mention (some slots may be omitted), {F}/{F2} = feature, {R} = rating phrase
TEMPLATES: dict[str, list[str]] = {
    "vehicle_info": [
        "What's the gas mileage of the {V}?", "How much horsepower does the {V} have?",
        "What engine does the {V} use?", "{V} fuel economy", "Tell me the history of the {V}",
        "What transmission is in the {V}?", "What are the specifications of the {V}?",
        "What is my car's MPG?", "Tell me about the {V}",
    ],
    "vehicle_lookup": [
        "Does the {V} have {F}?", "Is {F} standard on the {V}?", "does my {V} come with {F}",
        "What is the {R} of the {V}?", "How safe is the {V}?", "{V} {R}", "{V} {F}?",
        "Is the {V} a safe car?", "What safety rating did the {V} get?", "is {F} available on a {V}",
        "Did the {V} pass its crash tests?", "how many stars does the {V} have", "Tell me about the {V} safety",
        "Does my {V} have {F} and {F2}?", "{R} for {V}", "I drive a {V}. Does it have {F}?",
        "what's the rollover risk for the {V}", "Is the {V} safe for my family?",
        "Does my car have {F}?", "Is my truck safe?", "how safe is my SUV", "does my car have {F} standard",
        "What's my car's {R}?",
    ],
    "recall_lookup": [
        "Are there any recalls on the {V}?", "{V} recalls", "Is my {V} recalled?", "recall info for {V}",
        "Has the {V} had any safety recalls?", "any open recalls on a {V}", "Was there an airbag recall for the {V}?",
        "Are there recalls for my car?", "check recalls on my truck", "Does my {V} have a recall?",
    ],
    "concept_question": [
        "How does {F} work?", "What is {F}?", "What's the difference between {F} and {F2}?",
        "Can {F} prevent a crash?", "What are the limitations of {F}?", "Should I trust {F} in the rain?",
        "How are crash test ratings determined?", "What does a 5-star rating mean?", "How does NHTSA test rollover risk?",
        "Where can I find information about vehicle recalls?", "What is a recall?", "How do airbags protect passengers?",
        "Why is {F} important?", "explain {F}", "is {F} the same as {F2}", "How do I check if my car has a recall?",
        "What does the frontal crash test measure?", "When should a child move to a booster seat?",
        "Does {F} work at night?", "Who decides if a vehicle needs a recall?", "Is {F} required by law?",
        "How does {F} detect other cars?", "what happens when {F} activates", "Can I turn off {F}?",
        "What does NHTSA do?", "How is the side pole crash test done?", "what is electronic stability control for",
        "Do older cars have {F}?", "Why do airbags have a warning label?", "How should I report a vehicle safety problem?",
        "What's the safest way to install a car seat?", "How do I know if my tires are safe?",
        "Benefits of {F} vs {F2}", "is {F} worth it", "Tell me how {F} helps prevent crashes",
        # IIHS crash-test protocols and rating guidelines (ADR-009)
        "How fast is the IIHS small overlap test?", "What is the moderate overlap test?",
        "How does IIHS run the side impact test?", "What's the difference between the small overlap and moderate overlap tests?",
        "What dummy does IIHS use in the side test?", "How does IIHS rate head protection?",
        "What does a marginal rating mean?", "How are IIHS overall ratings weighted?",
        "What is submarining in a crash test?", "How heavy is the IIHS side impact barrier?",
        "What injuries does the side impact test measure?", "How does IIHS measure intrusion?",
        "Why does IIHS put a dummy in the back seat?", "How much overlap is there in the small overlap test?",
        "how is moderate overlap 2.0 different from the original test", "What does a good IIHS rating mean?",
    ],
    "out_of_scope": [
        "How much does a {V} cost?", "What's the price of the {V}?", "How much is insurance for a {V}?",
        "How much to fix the brakes on my {V}?",
        "Should I lease or buy a {V}?", "What's the resale value of a {V}?", "Find me a dealer for the {V}",
        "What's the weather tomorrow?", "Write me a poem about cars", "What's the capital of France?",
        "Recommend a good movie", "Help me with my math homework", "How much does it cost to replace an airbag?",
    ],
    "smalltalk": [
        "hi", "hello", "hey there", "thanks!", "thank you", "good morning", "who are you?", "what can you do?",
        "bye", "hey", "hello!", "thanks a lot", "thank you so much", "good afternoon", "good evening", "ok thanks",
        "cool", "great, thanks", "goodbye", "see you", "hi there!", "what are you?", "how can you help me?", "yo",
    ],
}


@dataclass
class Piece:
    text: str
    tag: str | None = None  # slot type or None


@dataclass
class Example:
    pieces: list[Piece]
    intent: str
    slots: dict = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    template_id: str = ""
    group: str = ""

    def render(self) -> dict:
        text, tokens, tags = "", [], []
        for piece in self.pieces:
            words = TOKEN.findall(piece.text)
            for i, word in enumerate(words):
                tokens.append(word)
                tags.append("O" if not piece.tag else f"{'B' if i == 0 else 'I'}-{piece.tag}")
            text += piece.text
        text = re.sub(r"\s+", " ", text).strip()
        return {
            "text": text, "tokens": tokens, "tags": tags, "intent": self.intent,
            "slots": self.slots, "features": self.features, "template_id": self.template_id,
        }


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _casing(text: str, rng: random.Random) -> str:
    return rng.choice([text.lower(), text.title(), text, text.upper() if len(text) <= 4 else text.lower()])


class Generator:
    def __init__(self, vocab: VehicleVocab, seed: int = 13, typo_rate: float = 0.08):
        self.vocab = vocab
        self.rng = random.Random(seed)
        self.typo_rate = typo_rate
        reverse = {}
        for alias, make in MAKE_ALIASES.items():
            reverse.setdefault(make, []).append(alias.lower())
        self.make_aliases = reverse
        # weight recent years, the way real users ask
        self.pool = [
            (make, family, years)
            for (make, family), years in vocab.years.items()
            if len(family) >= 2 and re.search(r"[A-Z]", family) and family not in {"4X2", "4X4"}
        ]

    # ----------------------------------------------------------- surfaces
    def make_surface(self, make: str) -> str:
        options = [make, make.title(), make.lower()] + self.make_aliases.get(make, [])
        surface = self.rng.choice(options)
        return _typo(surface, self.rng) if self.rng.random() < self.typo_rate else surface

    def model_surface(self, family: str) -> str:
        variants = [family, family.title(), family.lower()]
        if "-" in family:
            variants += [family.replace("-", "").lower(), family.replace("-", " ").lower()]
        surface = self.rng.choice(variants)
        return _typo(surface, self.rng) if self.rng.random() < self.typo_rate else surface

    def year_surface(self, year: int) -> str:
        return self.rng.choice([str(year)] * 4 + [f"'{str(year)[2:]}"])

    def vehicle(self, keep: dict[str, bool]) -> tuple[list[Piece], dict, str]:
        make, family, years = self.rng.choice(self.pool)
        recent = [y for y in years if y >= 2010] or years
        year = self.rng.choice(recent)
        slots = {"make": make if keep["make"] else None, "model": family if keep["model"] else None,
                 "model_year": year if keep["year"] else None, "body_style": None, "drive_train": None}
        parts: dict[str, Piece] = {}
        if keep["year"]:
            parts["YEAR"] = Piece(self.year_surface(year), "YEAR")
        if keep["make"]:
            parts["MAKE"] = Piece(self.make_surface(make), "MAKE")
        if keep["model"]:
            parts["MODEL"] = Piece(self.model_surface(family), "MODEL")
        order = self.rng.choice([["YEAR", "MAKE", "MODEL"], ["MAKE", "MODEL", "YEAR"], ["YEAR", "MODEL"], ["MAKE", "MODEL"]])
        order = [o for o in order if o in parts] + [o for o in ("YEAR", "MAKE", "MODEL") if o in parts and o not in order]
        pieces: list[Piece] = []
        for i, key in enumerate(order):
            if i:
                pieces.append(Piece(" "))
            pieces.append(parts[key])
        if keep["model"] and self.rng.random() < 0.2:
            drive = self.rng.choice(list(DRIVE_SURFACES))
            pieces += [Piece(" "), Piece(self.rng.choice(DRIVE_SURFACES[drive]), "DRIVE")]
            slots["drive_train"] = drive
        elif keep["model"] and self.rng.random() < 0.1:
            body = self.rng.choice(list(BODY_SURFACES))
            pieces += [Piece(" "), Piece(self.rng.choice(BODY_SURFACES[body]), "BODY")]
            slots["body_style"] = body
        return pieces, slots, f"{make}|{family}"

    def feature(self) -> tuple[str, str]:
        key = self.rng.choice(list(FEATURE_SYNONYMS))
        return key, self.rng.choice(FEATURE_SYNONYMS[key])

    # ----------------------------------------------------------- examples
    def example(self, intent: str) -> Example:
        template = self.rng.choice(TEMPLATES[intent])
        template_id = hashlib.md5(f"{intent}|{template}".encode()).hexdigest()[:8]
        pieces: list[Piece] = []
        slots: dict = {"make": None, "model": None, "model_year": None, "body_style": None, "drive_train": None}
        features: list[str] = []
        group = f"template:{template_id}"
        for chunk in re.split(r"(\{V\}|\{F\}|\{F2\}|\{R\})", template):
            if chunk == "{V}":
                r = self.rng.random()
                keep = {"make": r > 0.25, "model": r > 0.1, "year": self.rng.random() > 0.3}
                if not keep["make"] and not keep["model"]:
                    keep["model"] = True
                vehicle_pieces, slots, group = self.vehicle(keep)
                pieces += vehicle_pieces
            elif chunk in ("{F}", "{F2}"):
                key, surface = self.feature()
                while key in features:
                    key, surface = self.feature()
                features.append(key)
                pieces.append(Piece(_casing(surface, self.rng) if self.rng.random() < 0.3 else surface, "FEATURE"))
            elif chunk == "{R}":
                pieces.append(Piece(self.rng.choice(RATING_PHRASES)))
            elif chunk:
                pieces.append(Piece(chunk))
        if self.rng.random() < 0.3 and pieces and pieces[0].tag is None:
            pieces[0] = Piece(pieces[0].text.lower(), None)
        return Example(pieces=pieces, intent=intent, slots=slots, features=features,
                       template_id=template_id, group=group)


def split_for(group: str) -> str:
    bucket = int(hashlib.md5(group.encode()).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


INTENT_MIX = {"vehicle_lookup": 0.30, "recall_lookup": 0.14, "vehicle_info": 0.12, "concept_question": 0.24, "out_of_scope": 0.14, "smalltalk": 0.06}


def generate(vocab: VehicleVocab, n: int = 8000, seed: int = 13) -> list[dict]:
    gen = Generator(vocab, seed=seed)
    rng = random.Random(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 5:
        attempts += 1
        intent = rng.choices(list(INTENT_MIX), weights=list(INTENT_MIX.values()))[0]
        example = gen.example(intent)
        row = example.render()
        if row["text"].lower() in seen:
            continue
        seen.add(row["text"].lower())
        row["id"] = f"syn-{len(rows):05d}"
        row["split"] = split_for(example.group)
        rows.append(row)
    return rows


def openai_record(row: dict) -> dict:
    answer = {"intent": row["intent"], **row["slots"], "features": row["features"]}
    return {"messages": [
        {"role": "system", "content": LLM_NLU_SYSTEM},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": json.dumps(answer)},
    ]}


def write(rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev", "test"):
        part = [r for r in rows if r["split"] == split]
        counts[split] = len(part)
        with (out_dir / f"nlu_{split}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in part)
        if split in ("train", "dev"):
            with (out_dir / f"openai_{split}.jsonl").open("w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(openai_record(r)) + "\n" for r in part)
    (out_dir / "labels.json").write_text(json.dumps({"intents": INTENTS, "tags": TAGS}, indent=2))
    return counts


def main() -> None:
    from db.session import get_sessionmaker, init_db

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=get_settings().data_dir / "finetune")
    args = parser.parse_args()
    init_db()
    with get_sessionmaker()() as db:
        vocab = VehicleVocab.from_session(db)
    if not vocab.makes:
        raise SystemExit("vehicles table is empty: run `python ingest.py load-safercar` first")
    rows = generate(vocab, n=args.n, seed=args.seed)
    counts = write(rows, args.out)
    print(json.dumps({"written": counts, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
