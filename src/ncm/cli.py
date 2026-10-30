from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from ncm.build import build as build_splits
from ncm.build import read_raw, write
from ncm.collect import RAW, collect_day, days
from ncm.pncp import PncpClient, new_http_client

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
