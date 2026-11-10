from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import joblib
import pandas as pd
import typer

from ncm.baselines import majority, tfidf
from ncm.build import build as build_splits
from ncm.build import read_raw, write
from ncm.collect import RAW, collect_day, days
from ncm.evaluate import score
from ncm.hub import (
    DATASET_NAME,
    MODEL_NAME,
    SPACE_NAME,
    dataset_card,
    model_card,
    publish_dataset,
    publish_model,
    publish_onnx,
    publish_space,
    space_page,
    space_readme,
    space_summary,
    update_model_card,
)
from ncm.llm import BATCH, MAX_CHARS, PROVIDERS, Classifier, sample_indices, to_label
from ncm.pncp import PncpClient, new_http_client
from ncm.results import LABELS_FILE, RESULTS, build_report, record, save

app = typer.Typer(no_args_is_help=True, add_completion=False)

DATASET = Path("data/dataset")


@app.command()
def collect(
    start: Annotated[str, typer.Option("--from", help="First publication day, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--to", help="Last publication day, YYYY-MM-DD.")],
) -> None:
    """Download the items of every pregão eletrônico published in the range, nationwide."""
    first = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    with new_http_client() as http:
        client = PncpClient(http)
        for day in days(first, last):

            def progress(n: int, day: object = day) -> None:
                if n % 100 == 0:
                    typer.echo(f"  {day}: {n} tenders")

            result = collect_day(client, day, progress=progress)
            if result.skipped:
                typer.echo(f"{day}: already collected")
            else:
                typer.echo(
                    f"{day}: {result.tenders} tenders, {result.items} items, "
                    f"{result.failed_tenders} failed"
                )


@app.command()
def build(
    test_days: Annotated[int, typer.Option(help="Last days whose new descriptions are test.")] = 1,
) -> None:
    """Turn data/raw/ into train, validation and test splits in data/dataset/."""
    raw = sorted(RAW.glob("*.jsonl"))
    if not raw:
        raise typer.BadParameter("no raw data; run `ncm collect` first")
    splits, stats = build_splits(read_raw(raw), test_days)
    write(splits, stats, DATASET)
    typer.echo(
        json.dumps({k: v for k, v in stats.__dict__.items() if k != "distribution"}, indent=1)
    )


MODELS = Path("data/models")


def _splits() -> tuple[dict[str, pd.DataFrame], list[str]]:
    splits = {n: pd.read_parquet(DATASET / f"{n}.parquet") for n in ("train", "validation", "test")}
    classes = json.loads((DATASET / "stats.json").read_text(encoding="utf-8"))["classes"]
    return splits, classes


@app.command("baselines")
def baselines_command() -> None:
    """Train the majority and TF-IDF baselines and write their scores to results/."""
    splits, classes = _splits()
    for fitted in (
        majority(splits["train"]),
        tfidf(splits["train"], splits["validation"], classes),
    ):
        document = record(
            fitted.name,
            fitted.predict,
            splits,
            classes,
            {"params": fitted.params, "train_seconds": round(fitted.train_seconds, 1)},
        )
        typer.echo(
            f"{fitted.name}: test macro-F1 {document['test']['macro_f1']:.3f}, "
            f"accuracy {document['test']['accuracy']:.3f}"
        )
        if not isinstance(fitted.model, str):
            MODELS.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted.model, MODELS / f"{fitted.name}.joblib")


@app.command()
def report() -> None:
    """Write results/REPORT.md from every model's scores."""
    (RESULTS / "REPORT.md").write_text(build_report(), encoding="utf-8", newline="\n")
    typer.echo(f"written {RESULTS / 'REPORT.md'}")


def _env(name: str) -> str:
    """A setting from the environment or from .env, which stays out of git."""
    if value := os.environ.get(name):
        return value
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name and value.strip():
                return value.strip()
    raise typer.BadParameter(f"{name} is not set; add it to .env")


@app.command("publish-dataset")
def publish_dataset_command(
    name: Annotated[str, typer.Option(help="Dataset name on the Hub.")] = DATASET_NAME,
) -> None:
    """Publish data/dataset/ to the Hugging Face Hub, with a card generated from its stats."""
    splits, _ = _splits()
    stats = json.loads((DATASET / "stats.json").read_text(encoding="utf-8"))
    seen = pd.concat([frame["first_seen"] for frame in splits.values()]).astype(str)
    repo_id = f"{_env('HF_USERNAME')}/{name}"
    card = dataset_card(stats, (str(seen.min()), str(seen.max())), repo_id)
    typer.echo(f"published {publish_dataset(DATASET, repo_id, _env('HF_TOKEN'), card)}")


@app.command("zero-shot")
def zero_shot(
    model: Annotated[str, typer.Option(help="provider:name, e.g. gemini:gemini-3.5-flash-lite.")],
    sample: Annotated[int, typer.Option(help="Test items to classify, chosen by hash.")] = 1000,
) -> None:
    """Classify a fixed sample of the test set with an LLM given only the chapter list."""
    splits, classes = _splits()
    test = splits["test"]
    indices = sample_indices(test["text"].tolist(), sample)
    texts = test["text"].iloc[indices].tolist()
    truth = test["label"].iloc[indices].tolist()
    provider = model.partition(":")[0]
    key = _env(PROVIDERS[provider][1]) if provider in PROVIDERS else ""
    with httpx.Client(timeout=120) as http:
        classifier = Classifier(http, model, key, Path("data/llm-cache"))
        chapters = classifier.classify(
            texts, progress=lambda n: typer.echo(f"  {n}/{len(texts)}") if n % 250 == 0 else None
        )
    predicted = [to_label(c, classes) for c in chapters]
    usage = classifier.usage
    document = {
        "model": f"{classifier.name}-zero-shot",
        "params": {"base": model, "batch": BATCH, "max_chars": MAX_CHARS},
        "unanswered": sum(c is None for c in chapters),
        "tokens_per_1000_items": {
            "in": round(1000 * usage.tokens_in / len(texts)),
            "out": round(1000 * usage.tokens_out / len(texts)),
        },
        "ms_per_1000_items": round(1000 * sum(usage.latency_ms) / len(texts), 1),
        "validation": None,
        "test": score(truth, predicted, classes).as_dict(),
        "test_predictions": predicted,
        "sample": {"indices": indices, "labels": truth, "of": len(test)},
    }
    save(document)
    typer.echo(
        f"{document['model']}: {len(texts)} items, {usage.requests} requests "
        f"({usage.cached} cached), {document['unanswered']} unanswered, "
        f"macro-F1 {document['test']['macro_f1']:.3f}, accuracy {document['test']['accuracy']:.3f}"
    )


QUANTIZATION = RESULTS / "quantization" / "bertimbau-onnx.json"


def _quantization() -> dict | None:
    if not QUANTIZATION.exists():
        return None
    return json.loads(QUANTIZATION.read_text(encoding="utf-8"))


def _model_card(repo_id: str) -> str:
    """The model card from results/: scores, the comparison with the runner-up, and
    what the int8 export costs once scripts/export_onnx.py has measured it."""
    result = json.loads((RESULTS / "bertimbau.json").read_text(encoding="utf-8"))
    others = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(RESULTS.glob("*.json"))
        if p.name != "bertimbau.json"
    ]
    truth = (RESULTS / LABELS_FILE).read_text(encoding="utf-8").split()
    dataset_repo = f"{_env('HF_USERNAME')}/{DATASET_NAME}"
    return model_card(result, others, dataset_repo, repo_id, truth, _quantization())


@app.command("publish-model")
def publish_model_command(
    model_dir: Annotated[Path, typer.Argument(help="The saved model, from the Kaggle output.")],
    name: Annotated[str, typer.Option(help="Model name on the Hub.")] = MODEL_NAME,
) -> None:
    """Publish the fine-tuned model to the Hugging Face Hub, with a card from results/.
    Uploading 440 MB takes minutes: run it in a terminal, not with a short timeout."""
    repo_id = f"{_env('HF_USERNAME')}/{name}"
    url = publish_model(model_dir, repo_id, _env("HF_TOKEN"), _model_card(repo_id))
    typer.echo(f"published {url}")


@app.command("publish-onnx")
def publish_onnx_command(
    onnx_dir: Annotated[Path, typer.Argument(help="The output of scripts/export_onnx.py.")],
    name: Annotated[str, typer.Option(help="Model name on the Hub.")] = MODEL_NAME,
) -> None:
    """Add the int8 ONNX export to the model repository and refresh its card."""
    if _quantization() is None:
        raise typer.BadParameter(f"{QUANTIZATION} is missing; run scripts/export_onnx.py")
    repo_id = f"{_env('HF_USERNAME')}/{name}"
    publish_onnx(onnx_dir, repo_id, _env("HF_TOKEN"))
    update_model_card(repo_id, _env("HF_TOKEN"), _model_card(repo_id))
    typer.echo(f"published onnx/model_quantized.onnx to {repo_id}")


@app.command("publish-space")
def publish_space_command(
    name: Annotated[str, typer.Option(help="Space name on the Hub.")] = SPACE_NAME,
) -> None:
    """Publish the browser demo, a static page with the numbers from results/."""
    user = _env("HF_USERNAME")
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    truth = (RESULTS / LABELS_FILE).read_text(encoding="utf-8").split()
    dataset_repo, model_repo = f"{user}/{DATASET_NAME}", f"{user}/{MODEL_NAME}"
    summary = space_summary(results, truth, model_repo, dataset_repo, _quantization())
    files = {"README.md": space_readme(model_repo, dataset_repo), "index.html": space_page(summary)}
    typer.echo(f"published {publish_space(f'{user}/{name}', _env('HF_TOKEN'), files)}")
