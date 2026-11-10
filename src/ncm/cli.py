from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
import typer

from ncm.baselines import majority, tfidf
from ncm.build import build as build_splits
from ncm.build import read_raw, write
from ncm.collect import RAW, collect_day, days
from ncm.hub import DATASET_NAME, dataset_card, publish_dataset
from ncm.pncp import PncpClient, new_http_client
from ncm.results import RESULTS, build_report, record

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
    card = dataset_card(stats, (seen.min(), seen.max()), repo_id)
    typer.echo(f"published {publish_dataset(DATASET, repo_id, _env('HF_TOKEN'), card)}")
