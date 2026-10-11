"""Run baselines B0-B3 over the golden set and write metrics.

    python -m evaluation.run_evaluation --baseline all --split dev
    python -m evaluation.run_evaluation --baseline B3 --split test --confirm-test   # ONCE, after freezing

Baselines (proposal section 7) are ChatOptions on the same ChatService:
    B0  LLM only, no retrieval (closed book)
    B1  BM25 only, fixed chunks, no rerank
    B2  dense only, fixed chunks, no rerank
    B3  hybrid (BM25 + dense, RRF), structure chunks, cross-encoder rerank

Outputs data/eval_runs/<run_id>/{results.jsonl, summary.json} and rows in
eval_runs / eval_results. Answer correctness needs grading: the `correct`
column is left null for humans (or filled by --judge, then spot-checked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

from core.config import get_settings
from core.schemas import ChatOptions, ChatRequest
from evaluation import metrics
from evaluation.golden_set import load

BASELINES = {
    "B0": None,
    "B1": ChatOptions(retrieval_mode="bm25", rerank=False, chunk_strategy="fixed", use_clarifier=False),
    "B2": ChatOptions(retrieval_mode="dense", rerank=False, chunk_strategy="fixed", use_clarifier=False),
    "B3": ChatOptions(retrieval_mode="hybrid", rerank=True, chunk_strategy="structure", use_clarifier=False),
}

JUDGE_PROMPT = """You grade a chatbot answer against a reference answer.
Return JSON {"correct": true|false, "reason": "..."}. "correct" means the answer
states the reference's key facts without contradicting it. Ignore style."""


def ranked_urls(trace: dict) -> list[str]:
    rows = trace.get("candidates") or []
    key = (lambda r: r.get("rerank_rank") or 999) if any(r.get("rerank_rank") for r in rows) else (lambda r: r.get("retrieval_rank") or 999)
    return [r.get("url") for r in sorted(rows, key=key)]


def judge(llm, question: str, expected: str, answer: str) -> bool | None:
    result = llm.complete(JUDGE_PROMPT, [{"role": "user", "content": f"Question: {question}\nReference: {expected}\nAnswer: {answer}"}], json_mode=True, max_tokens=120)
    try:
        return bool(json.loads(result.text)["correct"])
    except (ValueError, KeyError):
        return None


def run(baseline: str, split: str, use_judge: bool = False) -> dict:
    from agent.orchestrator import ChatService
    from core.llm import get_llm
    from db.models import EvalResult, EvalRun
    from db.session import session_scope

    questions = [q for q in load() if q.split == split]
    if not questions:
        raise SystemExit(f"No golden questions with split={split}. Run `python -m evaluation.golden_set split`.")
    service = ChatService()
    service.generator.allow_fallback = False  # an outage must be an error, never a scored answer
    llm = get_llm()
    options = BASELINES[baseline]
    config = {"baseline": baseline, "options": options.model_dump() if options else None,
              "llm": service.settings.resolved_llm_provider(), "nlu": service.settings.nlu_backend,
              "embedding": service.settings.embedding_model, "reranker": service.settings.reranker_model}
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{baseline}-{split}-" + hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    out_dir = get_settings().data_dir / "eval_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in questions:
        start = time.perf_counter()
        if baseline == "B0":
            generated = service.generator.generate_closed_book(q.question)
            status, answer, cited, trace = "answered", generated.text, [], {"llm": generated.llm.as_trace()}
        else:
            response = service.handle(ChatRequest(message=q.question, options=options))
            status, answer = response.status.value, response.answer
            cited = [c.url for c in response.citations]
            trace = response.trace.model_dump(mode="json") if response.trace else {}
        latency = (time.perf_counter() - start) * 1000
        ranking = ranked_urls(trace)
        row = {
            "question_id": q.id, "category": q.category, "answerable": q.answerable, "status": status,
            "refused": status == "refused", "answer": answer, "cited_urls": cited,
            "citation_valid": all(u in ranking for u in cited) if cited else None,
            "recall_at_5": metrics.recall_at_k(ranking, q.source_urls, 5),
            "recall_at_20": metrics.recall_at_k(ranking, q.source_urls, 20),
            "mrr": metrics.mrr(ranking, q.source_urls),
            "ndcg_at_10": metrics.ndcg_at_k(ranking, q.source_urls, 10),
            "latency_ms": round(latency, 1), "cost_usd": (trace.get("llm") or {}).get("cost_usd"),
            "correct": judge(llm, q.question, q.expected_answer, answer) if use_judge and q.answerable and status == "answered" else None,
        }
        rows.append(row)

    summary = {
        "run_id": run_id, "config": config, "questions": len(rows),
        **{m: metrics.mean([r[m] for r in rows]) for m in ("recall_at_5", "recall_at_20", "mrr", "ndcg_at_10")},
        "recall_at_5_ci95": metrics.bootstrap_ci([r["recall_at_5"] for r in rows]),
        "citation_validity": metrics.mean([float(r["citation_valid"]) for r in rows if r["citation_valid"] is not None]),
        **metrics.refusal_rates(rows),
        "answer_correctness": metrics.mean([float(r["correct"]) for r in rows if r["correct"] is not None]),
        "latency_p50_ms": metrics.percentile([r["latency_ms"] for r in rows], 0.5),
        "latency_p95_ms": metrics.percentile([r["latency_ms"] for r in rows], 0.95),
        "cost_per_query_usd": metrics.mean([r["cost_usd"] for r in rows]),
    }
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with session_scope() as db:
        db.add(EvalRun(run_id=run_id, baseline=baseline, split=split, config=config, summary=summary))
        db.add_all(EvalResult(run_id=run_id, question_id=r["question_id"], status=r["status"], answer=r["answer"], metrics=r) for r in rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--confirm-test", action="store_true", help="required for --split test")
    parser.add_argument("--judge", action="store_true", help="LLM-judge correctness (spot-check by hand!)")
    args = parser.parse_args()

    lock = get_settings().data_dir / "eval_runs" / "TEST_SPLIT_USED"
    if args.split == "test":
        if not args.confirm_test:
            raise SystemExit("The test split is run ONCE after freezing the config. Add --confirm-test.")
        if lock.exists():
            raise SystemExit(f"Test split already used ({lock.read_text().strip()}). Report that run.")
    baselines = list(BASELINES) if args.baseline == "all" else [args.baseline]
    summaries = [run(b, args.split, args.judge) for b in baselines]
    if args.split == "test":
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(", ".join(s["run_id"] for s in summaries))
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
