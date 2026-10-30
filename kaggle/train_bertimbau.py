"""Fine-tune BERTimbau to predict the NCM chapter of an item description.

Runs as a Kaggle notebook on a free T4 GPU (see kaggle/kernel-metadata.json), reading the
splits from the classificador-ncm Kaggle dataset, and writes results/bertimbau.json in
the same format as every other model in the repository. The same file runs on a laptop
CPU with a tiny model and a few hundred examples, as a smoke test:

    uv run --group train python kaggle/train_bertimbau.py --data data/dataset \\
        --model prajjwal1/bert-tiny --limit 300 --epochs 1 --out results-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BERTIMBAU = "neuralmind/bert-base-portuguese-cased"
ZERO = {"zero_division": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/kaggle/input/classificador-ncm")
    parser.add_argument("--model", default=BERTIMBAU)
    parser.add_argument("--out", default="/kaggle/working/results")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Use only this many examples.")
    parser.add_argument("--push", default="", help="Hugging Face repo to push the model to.")
    # Notebook runtimes pass their own arguments; ignore them.
    return parser.parse_known_args()[0]


def score(truth: list[str], predicted: list[str], labels: list[str]) -> dict:
    """The same scores as ncm.evaluate.score, so results are comparable."""
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, **ZERO
    )
    mistakes = Counter((t, p) for t, p in zip(truth, predicted, strict=True) if t != p)
    return {
        "examples": len(truth),
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(
            float(f1_score(truth, predicted, labels=labels, average="macro", **ZERO)), 4
        ),
        "weighted_f1": round(
            float(f1_score(truth, predicted, labels=labels, average="weighted", **ZERO)), 4
        ),
        "per_class": {
            label: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusions": [[t, p, n] for (t, p), n in mistakes.most_common(10)],
    }


class WeightedTrainer(Trainer):
    """Cross-entropy with balanced class weights, the same choice as the TF-IDF baseline:
    four chapters are most of the data, and the target is macro-F1."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    splits = {n: pd.read_parquet(data / f"{n}.parquet") for n in ("train", "validation", "test")}
    if args.limit:
        splits = {n: f.sample(min(len(f), args.limit), random_state=0) for n, f in splits.items()}
    labels: list[str] = json.loads((data / "stats.json").read_text(encoding="utf-8"))["classes"]
    index = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=index,
    )

    def encode(frame: pd.DataFrame) -> list[dict]:
        encoded = tokenizer(frame["text"].tolist(), truncation=True, max_length=args.max_length)
        return [
            {"input_ids": ids, "attention_mask": mask, "labels": index[label]}
            for ids, mask, label in zip(
                encoded["input_ids"], encoded["attention_mask"], frame["label"], strict=True
            )
        ]

    train, validation, test = (encode(splits[n]) for n in ("train", "validation", "test"))

    counts = Counter(splits["train"]["label"])
    weights = torch.tensor(
        [len(splits["train"]) / (len(labels) * max(1, counts[label])) for label in labels],
        dtype=torch.float,
    )

    def metrics(prediction) -> dict:
        predicted = [labels[i] for i in np.argmax(prediction.predictions, axis=1)]
        truth = [labels[i] for i in prediction.label_ids]
        return {"macro_f1": f1_score(truth, predicted, labels=labels, average="macro", **ZERO)}

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="checkpoints",
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch * 2,
            # 6% of the steps warm up. Given as steps: transformers 5 dropped warmup_ratio,
            # and the Kaggle image may carry either major version.
            warmup_steps=int(0.06 * args.epochs * math.ceil(len(train) / args.batch)),
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            fp16=torch.cuda.is_available(),
            report_to=[],
            seed=0,
        ),
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        class_weights=weights,
    )
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start

    def predict(rows: list[dict]) -> tuple[list[str], float]:
        start = time.perf_counter()
        logits = trainer.predict(rows).predictions
        return [labels[i] for i in np.argmax(logits, axis=1)], time.perf_counter() - start

    validation_predicted, _ = predict(validation)
    test_predicted, test_seconds = predict(test)
    document = {
        "model": "bertimbau" if args.model == BERTIMBAU else args.model.split("/")[-1],
        "params": {
            "base": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "max_length": args.max_length,
        },
        "train_seconds": round(train_seconds, 1),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ms_per_1000_items": round(1000 * test_seconds / max(1, len(test)) * 1000, 1),
        "validation": score(splits["validation"]["label"].tolist(), validation_predicted, labels),
        "test": score(splits["test"]["label"].tolist(), test_predicted, labels),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{document['model']}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: document[k] for k in ("model", "train_seconds")}))
    print("test macro-F1", document["test"]["macro_f1"], "accuracy", document["test"]["accuracy"])

    if args.push:
        # HF_TOKEN comes from a Kaggle secret, never from the code.
        trainer.model.push_to_hub(args.push, token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(args.push, token=os.environ["HF_TOKEN"])


if __name__ == "__main__":
    main()
