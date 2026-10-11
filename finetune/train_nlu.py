"""Fine-tune the clarifier NLU: an intent classifier + a BIO slot tagger.

Base model: distilbert-base-uncased (66M params) - trains in ~10-20 min on a
laptop CPU, ~2 min on a free Colab T4. Override with --base.

    pip install -r requirements-finetune.txt
    python -m finetune.synth_data --n 8000           # make the data first
    python -m finetune.train_nlu                     # -> models/clarifier/{intent,slots}
    python -m finetune.train_nlu --smoke             # 50 steps, to check the setup
    NLU_BACKEND=hf python -m finetune.eval_nlu       # compare against rules

Then set NLU_BACKEND=hf in .env to use it in the chatbot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import get_settings


def load_split(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"nlu_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing - run `python -m finetune.synth_data` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_intent(base: str, train: list[dict], dev: list[dict], intents: list[str], out: Path, args) -> dict:
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    label2id = {label: i for i, label in enumerate(intents)}
    tokenizer = AutoTokenizer.from_pretrained(base)

    def encode(rows):
        ds = Dataset.from_list([{"text": r["text"], "label": label2id[r["intent"]]} for r in rows])
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(intents), id2label=dict(enumerate(intents)), label2id=label2id
    )

    def metrics(pred):
        y_pred = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(pred.label_ids, y_pred),
                "macro_f1": f1_score(pred.label_ids, y_pred, average="macro")}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "_intent_ckpt"), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=64, eval_strategy="epoch",
            save_strategy="no", logging_steps=50, seed=args.seed, report_to=[],
            max_steps=50 if args.smoke else -1,
        ),
        train_dataset=encode(train), eval_dataset=encode(dev),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    trainer.save_model(str(out / "intent"))
    tokenizer.save_pretrained(str(out / "intent"))
    return result


def train_slots(base: str, train: list[dict], dev: list[dict], tags: list[str], out: Path, args) -> dict:
    import numpy as np
    from datasets import Dataset
    from seqeval.metrics import f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    tag2id = {t: i for i, t in enumerate(tags)}
    tokenizer = AutoTokenizer.from_pretrained(base)

    def encode(rows):
        def align(batch):
            enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=64)
            labels = []
            for i, word_tags in enumerate(batch["tags"]):
                previous, row = None, []
                for word_id in enc.word_ids(i):
                    if word_id is None or word_id == previous:
                        row.append(-100)  # special tokens and sub-word continuations
                    else:
                        row.append(tag2id[word_tags[word_id]])
                    previous = word_id
                labels.append(row)
            enc["labels"] = labels
            return enc

        ds = Dataset.from_list([{"tokens": r["tokens"], "tags": r["tags"]} for r in rows if r["tokens"]])
        return ds.map(align, batched=True, remove_columns=["tokens", "tags"])

    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(tags), id2label=dict(enumerate(tags)), label2id=tag2id
    )

    def metrics(pred):
        y_pred = np.argmax(pred.predictions, axis=-1)
        true_tags, pred_tags = [], []
        for p_row, l_row in zip(y_pred, pred.label_ids):
            keep = [(tags[p], tags[l]) for p, l in zip(p_row, l_row) if l != -100]
            pred_tags.append([p for p, _ in keep])
            true_tags.append([l for _, l in keep])
        return {"precision": precision_score(true_tags, pred_tags), "recall": recall_score(true_tags, pred_tags),
                "f1": f1_score(true_tags, pred_tags)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "_slots_ckpt"), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=64, eval_strategy="epoch",
            save_strategy="no", logging_steps=50, seed=args.seed, report_to=[],
            max_steps=50 if args.smoke else -1,
        ),
        train_dataset=encode(train), eval_dataset=encode(dev),
        data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    trainer.save_model(str(out / "slots"))
    tokenizer.save_pretrained(str(out / "slots"))
    return result


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--data", type=Path, default=settings.data_dir / "finetune")
    parser.add_argument("--out", type=Path, default=settings.nlu_model_dir)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smoke", action="store_true", help="50 steps only, to verify the setup")
    args = parser.parse_args()

    labels = json.loads((args.data / "labels.json").read_text())
    train, dev = load_split(args.data, "train"), load_split(args.data, "dev")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "base_model": args.base,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "intent_dev": train_intent(args.base, train, dev, labels["intents"], args.out, args),
        "slots_dev": train_slots(args.base, train, dev, labels["tags"], args.out, args),
        "smoke": args.smoke,
    }
    (args.out / "training_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(f"\nSaved to {args.out}. Use it with NLU_BACKEND=hf")


if __name__ == "__main__":
    main()
