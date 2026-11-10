"""Export the fine-tuned model to ONNX for the browser demo, quantize it, and measure
what the conversion costs.

The demo runs the model in the visitor's browser with transformers.js, because a free
Hugging Face Space can only serve static files. Browsers download the weights, so the
fp32 export (about 440 MB) is quantized to int8 weights (about a quarter of that).

Quantization is not free, and how it is done matters: one scale per weight tensor
squashes the rows with small weights, one scale per output channel does not. Both are
tried, the one whose predictions agree most with fp32 on the VALIDATION set is kept
(choosing on test would make the test score optimistic), and then every file is scored
on the whole test set and compared with the PyTorch model's own predictions in
results/bertimbau.json.

    uv run --with onnx==1.23.2 --with onnxruntime==1.31.0 --with onnxscript==0.7.2 \\
        --group train python scripts/export_onnx.py <saved model dir> <output dir>

The output directory is laid out the way transformers.js expects: config and tokenizer
at the top, the ONNX files under onnx/.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ncm.evaluate import score

MAX_LENGTH = 128  # what the model was trained with


QUANTIZATIONS = {"int8-per-tensor": False, "int8-per-channel": True}


def export(model_dir: Path, out: Path) -> Path:
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, out / name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sample = tokenizer(["seringa descartável"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    fp32 = out / "onnx" / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(sample[n] for n in names),
            fp32,
            input_names=names,
            output_names=["logits"],
            dynamic_axes={n: {0: "batch", 1: "sequence"} for n in names} | {"logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    return fp32


def agreement(a: list[str], b: list[str]) -> float:
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def predict(
    path: Path, model_dir: Path, texts: list[str], labels: list[str]
) -> tuple[list[str], float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    predicted: list[str] = []
    start = time.perf_counter()
    for i in range(0, len(texts), 64):
        batch = tokenizer(
            texts[i : i + 64],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        logits = session.run(["logits"], {k: batch[k].astype(np.int64) for k in batch})[0]
        predicted += [labels[j] for j in logits.argmax(axis=1)]
    return predicted, time.perf_counter() - start


def main() -> None:
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    fp32 = export(model_dir, out)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    validation = pd.read_parquet("data/dataset/validation.parquet")["text"].tolist()
    test = pd.read_parquet("data/dataset/test.parquet")
    texts, truth = test["text"].tolist(), test["label"].tolist()
    reference = json.loads(Path("results/bertimbau.json").read_text(encoding="utf-8"))

    files = {"fp32": fp32}
    fp32_on_validation, _ = predict(fp32, model_dir, validation, labels)
    chosen, best = "", -1.0
    for name, per_channel in QUANTIZATIONS.items():
        path = out / "onnx" / f"model_{name}.onnx"
        quantize_dynamic(fp32, path, weight_type=QuantType.QInt8, per_channel=per_channel)
        files[name] = path
        on_validation, _ = predict(path, model_dir, validation, labels)
        if (agree := agreement(on_validation, fp32_on_validation)) > best:
            chosen, best = name, agree

    report: dict[str, object] = {
        "reference": "results/bertimbau.json (PyTorch, fp32)",
        "chosen": chosen,
        "chosen_by": "agreement with fp32 on the validation set",
        "validation_agreement": round(best, 4),
    }
    for name, path in files.items():
        predicted, seconds = predict(path, model_dir, texts, labels)
        scores = score(truth, predicted, labels)
        report[name] = {
            "megabytes": round(path.stat().st_size / 2**20, 1),
            "macro_f1": round(scores.macro_f1, 4),
            "accuracy": round(scores.accuracy, 4),
            "agrees_with_pytorch": round(agreement(predicted, reference["test_predictions"]), 4),
            "cpu_seconds_per_1000_items": round(1000 * seconds / len(texts), 2),
        }
        print(name, report[name])
    # transformers.js loads onnx/model_quantized.onnx for dtype "q8".
    shutil.move(files[chosen], out / "onnx" / "model_quantized.onnx")
    for name in QUANTIZATIONS:
        if name != chosen:
            files[name].unlink()
    target = Path("results/quantization")
    target.mkdir(parents=True, exist_ok=True)
    (target / "bertimbau-onnx.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
