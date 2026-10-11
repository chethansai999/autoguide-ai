"""Evaluate an NLU backend: intent, slots, and the clarification decision.

    python -m finetune.eval_nlu                         # NLU_BACKEND from .env (rules)
    python -m finetune.eval_nlu --backend hf
    python -m finetune.eval_nlu --backend llm --limit 200   # costs API calls
    python -m finetune.eval_nlu --set realistic         # hand-written queries

Metrics
  intent_accuracy / intent_macro_f1
  slot_accuracy[make|model|model_year|drive_train|body_style]  exact canonical match
  decision_accuracy   agent asks <=> it should ask (gold slots through the resolver)
  asked_slot_accuracy when both ask, they ask for the same thing
  over_ask_rate / under_ask_rate
  latency_ms p50 / p95
Writes data/finetune/eval_<backend>_<set>.json - paste the summary into
the generation / ethics reports and compare backends on the SAME set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent import clarifier
from agent.nlu import build_nlu
from core.config import get_settings
from core.schemas import Intent, ResolutionStatus, VehicleSlots
from structured.resolver import VehicleResolver
from structured.safercar import compact

SLOTS = ["make", "model", "model_year", "drive_train", "body_style"]


def _same(slot: str, gold, pred, resolver: VehicleResolver) -> bool:
    if gold in (None, "") and pred in (None, ""):
        return True
    if gold in (None, "") or pred in (None, ""):
        return False
    if slot == "model":  # 'F-150' vs 'F-150 SUPERCAB' both acceptable family hits
        return compact(str(pred)).startswith(compact(str(gold))) or compact(str(gold)).startswith(compact(str(pred)))
    return compact(str(gold)) == compact(str(pred))


def _decision(resolver: VehicleResolver, intent: str, slots: dict, features: list[str]):
    if intent not in (Intent.VEHICLE_LOOKUP.value, Intent.RECALL_LOOKUP.value):
        return False, None
    vehicle_slots = VehicleSlots(**{k: v for k, v in slots.items() if k in VehicleSlots.model_fields})
    resolution = resolver.resolve(vehicle_slots)
    if resolution.status == ResolutionStatus.NOT_FOUND and not vehicle_slots.is_empty():
        return True, "not_found"
    decision = clarifier.decide(Intent(intent), resolution, vehicle_slots, features)
    return decision.should_ask, decision.missing_slot


def evaluate(rows: list[dict], backend: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    from db.session import get_sessionmaker, init_db

    settings = get_settings().model_copy(update={"nlu_backend": backend})
    init_db()
    resolver = VehicleResolver(get_sessionmaker())
    nlu = build_nlu(resolver.vocab, settings)

    gold_intents, pred_intents, latencies = [], [], []
    slot_hits = {s: 0 for s in SLOTS}
    decisions = {"agree": 0, "total": 0, "over": 0, "under": 0, "slot_agree": 0, "both_ask": 0}
    errors = []
    for row in rows:
        start = time.perf_counter()
        pred = nlu.parse(row["text"])
        latencies.append((time.perf_counter() - start) * 1000)
        gold_intents.append(row["intent"])
        pred_intents.append(pred.intent.value)
        pred_slots = pred.slots.model_dump()
        for slot in SLOTS:
            slot_hits[slot] += _same(slot, row["slots"].get(slot), pred_slots.get(slot), resolver)
        gold_ask, gold_slot = _decision(resolver, row["intent"], row["slots"], row.get("features", []))
        pred_ask, pred_slot = _decision(resolver, pred.intent.value, pred_slots, pred.features)
        decisions["total"] += 1
        decisions["agree"] += gold_ask == pred_ask
        decisions["over"] += pred_ask and not gold_ask
        decisions["under"] += gold_ask and not pred_ask
        if gold_ask and pred_ask:
            decisions["both_ask"] += 1
            decisions["slot_agree"] += gold_slot == pred_slot
        if pred.intent.value != row["intent"] and len(errors) < 25:
            errors.append({"text": row["text"], "gold": row["intent"], "pred": pred.intent.value})

    n = max(len(rows), 1)
    return {
        "backend": nlu.backend,
        "examples": len(rows),
        "intent_accuracy": round(accuracy_score(gold_intents, pred_intents), 4),
        "intent_macro_f1": round(f1_score(gold_intents, pred_intents, average="macro"), 4),
        "slot_accuracy": {s: round(slot_hits[s] / n, 4) for s in SLOTS},
        "decision_accuracy": round(decisions["agree"] / n, 4),
        "asked_slot_accuracy": round(decisions["slot_agree"] / max(decisions["both_ask"], 1), 4),
        "over_ask_rate": round(decisions["over"] / n, 4),
        "under_ask_rate": round(decisions["under"] / n, 4),
        "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else None,
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
        "sample_intent_errors": errors,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["rules", "hf", "llm"], default=settings.nlu_backend)
    parser.add_argument("--set", choices=["synthetic", "realistic"], default="synthetic")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.set == "synthetic":
        path = settings.data_dir / "finetune" / "nlu_test.jsonl"
    else:
        path = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "nlu_realistic.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()][: args.limit]
    report = evaluate(rows, args.backend)
    out = settings.data_dir / "finetune" / f"eval_{args.backend}_{args.set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sample_intent_errors"}, indent=2))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
