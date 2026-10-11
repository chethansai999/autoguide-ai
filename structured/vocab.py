"""Vehicle vocabulary (gazetteer) built from the ``vehicles`` table.

Used by the rule-based NLU baseline, the resolver, and the synthetic
training-data generator for the fine-tuned clarifier.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Vehicle
from structured.safercar import compact

MAKE_ALIASES = {
    "CHEVY": "CHEVROLET", "VW": "VOLKSWAGEN", "VOLKS": "VOLKSWAGEN",
    "MERCEDES": "MERCEDES-BENZ", "BENZ": "MERCEDES-BENZ", "MERC": "MERCEDES-BENZ",
    "MB": "MERCEDES-BENZ", "LANDROVER": "LAND ROVER", "ALFA": "ALFA ROMEO",
    "ROLLS": "ROLLS-ROYCE", "CHRYSLER": "CHRYSLER", "CADDY": "CADILLAC",
    "OLDS": "OLDSMOBILE", "MITSU": "MITSUBISHI", "HUNDAI": "HYUNDAI", "HYUNDIA": "HYUNDAI",
}

# Makes that are also everyday words: only detected when capitalized.
AMBIGUOUS_MAKES = {"SMART", "MINI", "RAM", "EAGLE", "CODA", "LUCID", "GENESIS", "SATURN", "MERCURY", "GEO", "GM", "SRT", "STI"}

# Model families that are common English words. Without an explicit make
# they are only matched when capitalized in the user's text ("my Focus").
COMMON_WORD_MODELS = {
    "ONE", "FIT", "CUBE", "SOUL", "EDGE", "ESCAPE", "PILOT", "PASSPORT", "JOURNEY",
    "PATRIOT", "COMPASS", "LIBERTY", "RANGER", "EXPLORER", "EXPEDITION", "FRONTIER",
    "ROGUE", "TITAN", "QUEST", "ELEMENT", "INSIGHT", "CLARITY", "ACCENT", "MATRIX",
    "SPARK", "SONIC", "VOLT", "BOLT", "LEAF", "AIR", "ENCORE", "ENVISION", "REGAL",
    "TERRAIN", "CANYON", "EXPRESS", "CITY", "LEGACY", "ASCENT", "VENUE", "ORIGIN",
    "SKY", "AURA", "OUTLOOK", "ION", "RELAY", "SABLE", "COUGAR", "FOCUS", "FUSION",
    "FLEX", "PROBE", "CHARGER", "CHALLENGER", "SPIRIT", "SHADOW", "DART", "NITRO",
    "MAGNUM", "AVENGER", "CALIBER", "NEON", "VOYAGER", "CROSSFIRE", "TRANSIT",
    "KICKS", "VERSA", "JUKE", "ACCORD", "CIVIC", "PRELUDE", "INTEGRA", "LEGEND",
    "ODYSSEY", "MIRAGE", "ECLIPSE", "OUTLANDER", "LANCER", "ENDEAVOR", "FORTE",
    "RIO", "OPTIMA", "STINGER", "CARNIVAL", "TELLURIDE", "SPORTAGE", "VENZA",
    "SEQUOIA", "TUNDRA", "TACOMA", "HIGHLANDER", "MATRIX", "ECHO", "SOLARA",
    "AVALON", "CROWN", "SUPRA", "MODEL", "SERIES", "CLASS", "HYBRID", "SPORT",
    "COUPE", "SEDAN", "WAGON", "VAN", "TRUCK", "CARGO", "PLUS", "LIMITED",
    "TOURING", "PREMIUM", "CLASSIC", "ROADSTER", "CONVERTIBLE", "HATCHBACK",
    "PICKUP", "PASSENGER", "ELECTRIC", "SPYDER", "SPIDER", "GT", "SL", "TT", "IS",
    "ES", "LS", "GS", "HS", "LX", "GX", "RC", "LC", "UX", "NX", "RX", "CT", "SC",
    "S", "X", "Z", "M", "Q", "A", "E", "C", "G", "I", "Y", "K", "F",
}


@dataclass
class VehicleVocab:
    makes: list[str]
    make_counts: Counter
    families: dict[str, list[str]]                       # make -> families
    model_index: dict[str, set[tuple[str, str]]]         # compact -> {(make, display)}
    years: dict[tuple[str, str], list[int]]              # (make, family) -> years
    min_year: int
    max_year: int
    _make_lookup: dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------- builders
    @classmethod
    def from_rows(cls, rows: list[tuple[str, str, str, int]]) -> "VehicleVocab":
        """rows: (make, model, model_family, model_year)."""
        make_counts: Counter = Counter()
        families: dict[str, Counter] = defaultdict(Counter)
        model_index: dict[str, set[tuple[str, str]]] = defaultdict(set)
        years: dict[tuple[str, str], set[int]] = defaultdict(set)
        family_latest: dict[tuple[str, str], int] = {}
        for make, model, family, year in rows:
            make_counts[make] += 1
            families[make][family] += 1
            years[(make, family)].add(year)
            family_latest[(make, family)] = max(year, family_latest.get((make, family), 0))
            model_index[compact(family)].add((make, family))
            model_index[compact(model)].add((make, model))
        ordered_families = {
            make: sorted(counter, key=lambda f: (-family_latest[(make, f)], -counter[f], f))
            for make, counter in families.items()
        }
        all_years = [y for ys in years.values() for y in ys]
        vocab = cls(
            makes=sorted(make_counts),
            make_counts=make_counts,
            families=ordered_families,
            model_index=dict(model_index),
            years={k: sorted(v) for k, v in years.items()},
            min_year=min(all_years) if all_years else 1990,
            max_year=max(all_years) if all_years else 2030,
        )
        vocab._make_lookup = {compact(m): m for m in vocab.makes}
        vocab._make_lookup.update({k: v for k, v in MAKE_ALIASES.items() if v in make_counts})
        return vocab

    @classmethod
    def from_session(cls, session: Session) -> "VehicleVocab":
        rows = session.execute(
            select(Vehicle.make, Vehicle.model, Vehicle.model_family, Vehicle.model_year)
        ).all()
        return cls.from_rows([tuple(r) for r in rows])

    # ---------------------------------------------------------- lookups
    def canonical_make(self, text: str | None, fuzzy: bool = True, cutoff: int = 85) -> str | None:
        """Exact or fuzzy make. Callers pass a lower ``cutoff`` only when the
        message is clearly about a car ("give me totyata safety rating")."""
        if not text:
            return None
        key = compact(text)
        if key in self._make_lookup:
            return self._make_lookup[key]
        if fuzzy and len(key) >= 4:
            match = process.extractOne(key, list(self._make_lookup), scorer=fuzz.ratio, score_cutoff=cutoff)
            # "audit" is not a typo of Audi: a longer word built on the make only
            # counts as a plural ("hondas", "fords").
            if match and key.startswith(match[0]) and key[len(match[0]):] not in ("S", "ES"):
                match = None
            if match:
                return self._make_lookup[match[0]]
        return None

    def models_matching(self, text: str, make: str | None = None, fuzzy: bool = True) -> list[tuple[str, str]]:
        """Return [(make, model_display)] for a model mention."""
        key = compact(text)
        hits = sorted(self.model_index.get(key, set()))
        if make:
            hits = [h for h in hits if h[0] == make]
        if hits or not fuzzy or len(key) < 4:
            return hits
        candidates = (
            {compact(f): (make, f) for f in self.families.get(make, [])}
            if make
            else {k: sorted(v)[0] for k, v in self.model_index.items() if len(k) >= 4}
        )
        match = process.extractOne(key, list(candidates), scorer=fuzz.ratio, score_cutoff=85)
        return [candidates[match[0]]] if match else []

    def makes_for_model(self, text: str) -> list[str]:
        return sorted({make for make, _ in self.models_matching(text)})

    def family_years(self, make: str, family: str) -> list[int]:
        return self.years.get((make, family), [])

    def suggest_makes(self, text: str, limit: int = 5) -> list[str]:
        matches = process.extract(compact(text), list(self._make_lookup), scorer=fuzz.ratio, limit=limit)
        return list(dict.fromkeys(self._make_lookup[m[0]] for m in matches))

    def suggest_families(self, make: str, text: str, limit: int = 6) -> list[str]:
        families = self.families.get(make, [])
        matches = process.extract(compact(text), [compact(f) for f in families], scorer=fuzz.ratio, limit=limit)
        lookup = {compact(f): f for f in families}
        return [lookup[m[0]] for m in matches]

    def popular_makes(self, limit: int = 8) -> list[str]:
        return [m for m, _ in self.make_counts.most_common(limit)]
