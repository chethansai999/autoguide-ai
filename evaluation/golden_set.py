"""Golden question set: schema, validation, stratified 60/40 split.

File: evaluation/data/golden_set.jsonl (committed - it is a deliverable).
Required mix (proposal): 15 single_hop, 6 multi_hop, 4 comparative,
4 temporal, 8 unanswerable, 3 ambiguous_adversarial = 40.

    python -m evaluation.golden_set validate
    python -m evaluation.golden_set split          # assigns dev/test once, stratified
    python -m evaluation.golden_set import-csv data/golden_template.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"

REQUIRED_MIX = {
    "single_hop": 15, "multi_hop": 6, "comparative": 4,
    "temporal": 4, "unanswerable": 8, "ambiguous_adversarial": 3,
}
Category = Literal["single_hop", "multi_hop", "comparative", "temporal", "unanswerable", "ambiguous_adversarial"]


class GoldenQuestion(BaseModel):
    id: str
    category: Category
    question: str
    answerable: bool
    expected_answer: str = ""
    source_urls: list[str] = Field(default_factory=list)       # document-level relevance
    relevant_chunk_ids: list[str] = Field(default_factory=list)  # optional, per strategy
    support_quote: str = ""                                   # <= 25 words
    searched_notes: str = ""                                  # unanswerable: what was searched
    expected_status: Literal["answered", "refused", "needs_clarification"] | None = None
    split: Literal["dev", "test"] | None = None
    verified: bool = False
    verified_by: str = ""

    @field_validator("support_quote")
    @classmethod
    def quote_limit(cls, value: str) -> str:
        if len(value.split()) > 25:
            raise ValueError("support_quote must be 25 words or fewer")
        return value

    @model_validator(mode="after")
    def consistency(self):
        if self.answerable and not self.source_urls:
            raise ValueError(f"{self.id}: answerable questions need source_urls")
        if not self.answerable and not self.searched_notes:
            raise ValueError(f"{self.id}: unanswerable questions need searched_notes")
        return self


def load(path: Path = PATH) -> list[GoldenQuestion]:
    if not path.exists():
        return []
    return [GoldenQuestion.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(rows: list[GoldenQuestion], path: Path = PATH) -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


def validate(rows: list[GoldenQuestion]) -> dict:
    counts = Counter(r.category for r in rows)
    problems = [f"{cat}: have {counts.get(cat, 0)}, need {n}" for cat, n in REQUIRED_MIX.items() if counts.get(cat, 0) < n]
    unverified = [r.id for r in rows if not r.verified]
    duplicates = [q for q, c in Counter(r.question.lower() for r in rows).items() if c > 1]
    return {"total": len(rows), "by_category": dict(counts), "shortfalls": problems,
            "unverified": unverified, "duplicates": duplicates,
            "split": dict(Counter(r.split for r in rows)), "ok": not problems and not unverified and not duplicates}


def stratified_split(rows: list[GoldenQuestion], dev_share: float = 0.6, seed: int = 633) -> list[GoldenQuestion]:
    """Assign dev/test per category (60/40). Never re-splits assigned rows."""
    rng = random.Random(seed)
    by_cat: dict[str, list[GoldenQuestion]] = {}
    for row in rows:
        if row.split is None:
            by_cat.setdefault(row.category, []).append(row)
    for items in by_cat.values():
        rng.shuffle(items)
        n_dev = round(len(items) * dev_share)
        for i, row in enumerate(items):
            row.split = "dev" if i < n_dev else "test"
    return rows


def import_csv(csv_path: Path) -> list[GoldenQuestion]:
    """Import Soujanya's golden_template.csv columns."""
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("question"):
                continue
            answerable = rec.get("category") != "unanswerable"
            rows.append(GoldenQuestion(
                id=rec["id"], category=rec["category"], question=rec["question"], answerable=answerable,
                expected_answer=rec.get("expected_answer", ""),
                source_urls=[u for u in (rec.get("source_url") or "").split("|") if u],
                relevant_chunk_ids=[c for c in (rec.get("relevant_chunk_ids") or "").split("|") if c],
                support_quote=rec.get("evidence_quote", ""),
                searched_notes=rec.get("searched_notes", "") or ("" if answerable else "see notes"),
                split=rec.get("split") or None,
                verified=str(rec.get("verified", "")).lower() in ("true", "1", "yes"),
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("split")
    imp = sub.add_parser("import-csv")
    imp.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.cmd == "validate":
        print(json.dumps(validate(load()), indent=2))
    elif args.cmd == "split":
        rows = stratified_split(load())
        save(rows)
        print(json.dumps(validate(rows), indent=2))
    else:
        rows = import_csv(args.csv)
        save(rows)
        print(f"Imported {len(rows)} questions into {PATH}")


if __name__ == "__main__":
    main()
