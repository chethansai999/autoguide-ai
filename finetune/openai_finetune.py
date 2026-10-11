"""Optional: fine-tune an OpenAI model for NLU using your API key.

This spends money on your OpenAI account, so it does nothing without --yes.
It prints a token estimate first. Check OpenAI's current fine-tuning docs for
which base models are fine-tunable and what they cost before running.

    python -m finetune.synth_data --n 3000          # smaller set is plenty here
    python -m finetune.openai_finetune               # dry run: shows the estimate
    python -m finetune.openai_finetune --yes         # uploads + starts the job
    python -m finetune.openai_finetune --status JOB_ID

When the job succeeds it prints a model id like ``ft:gpt-4o-mini-...``. Put it
in .env:
    NLU_BACKEND=llm
    NLU_LLM_MODEL=ft:gpt-4o-mini-...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.config import get_settings


def estimate_tokens(path: Path) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        count = lambda text: len(encoding.encode(text))  # noqa: E731
    except Exception:
        count = lambda text: int(len(text.split()) * 1.3)  # noqa: E731
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        for message in json.loads(line)["messages"]:
            total += count(message["content"]) + 4
    return total


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=settings.data_dir / "finetune")
    parser.add_argument("--base", default="gpt-4o-mini-2024-07-18", help="a fine-tunable model id")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--yes", action="store_true", help="actually upload and start the paid job")
    parser.add_argument("--status", help="poll an existing job id")
    args = parser.parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set in .env")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    if args.status:
        job = client.fine_tuning.jobs.retrieve(args.status)
        print(json.dumps({"status": job.status, "model": job.fine_tuned_model, "error": str(job.error)}, indent=2))
        return

    train_path, dev_path = args.data / "openai_train.jsonl", args.data / "openai_dev.jsonl"
    if not train_path.exists():
        raise SystemExit("Run `python -m finetune.synth_data` first")
    lines = train_path.read_text(encoding="utf-8").splitlines()[: args.max_examples]
    trimmed = args.data / "openai_train_trimmed.jsonl"
    trimmed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tokens = estimate_tokens(trimmed)
    print(f"Training examples: {len(lines)}")
    print(f"Estimated training tokens: {tokens:,} x {args.epochs} epochs = {tokens * args.epochs:,}")
    print("Check the current per-token fine-tuning price for your base model before continuing.")
    if not args.yes:
        print("\nDry run only. Re-run with --yes to upload and start the job.")
        return

    train_file = client.files.create(file=trimmed.open("rb"), purpose="fine-tune")
    dev_file = client.files.create(file=dev_path.open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        model=args.base,
        training_file=train_file.id,
        validation_file=dev_file.id,
        suffix="autoguide-nlu",
        hyperparameters={"n_epochs": args.epochs},
    )
    print(f"Started job {job.id}. Polling every 60 s (Ctrl+C is safe; use --status {job.id} later).")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(time.strftime("%H:%M:%S"), job.status)
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(60)
    if job.status == "succeeded":
        print(f"\nDone. Add to .env:\nNLU_BACKEND=llm\nNLU_LLM_MODEL={job.fine_tuned_model}")
    else:
        print(f"Job ended with status {job.status}: {job.error}")


if __name__ == "__main__":
    main()
