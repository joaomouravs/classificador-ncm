"""Download the items of every electronic reverse auction published on a range of days.

Each day goes to data/raw/<day>.jsonl, one item per line, every item kept (with or
without an NCM code): the unlabelled ones are what the model is for, and their share is
part of the story. A day is written as it is fetched and marked done at the end, so an
interrupted collection resumes where it stopped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

from ncm.pncp import Json, PncpClient

RAW = Path("data/raw")


@dataclass
class DayResult:
    day: date
    tenders: int = 0
    items: int = 0
    failed_tenders: int = 0
    skipped: bool = False


def item_record(tender: Json, item: Json) -> Json:
    return {
        "control_number": tender["numeroControlePNCP"],
        "published_at": tender["dataPublicacaoPncp"][:10],
        "uf": tender["unidadeOrgao"]["ufSigla"],
        "item_number": item["numeroItem"],
        "description": item.get("descricao") or "",
        "kind": item.get("materialOuServico"),
        "ncm": item.get("ncmNbsCodigo"),
        "catalog_code": item.get("catalogoCodigoItem"),
        "unit": item.get("unidadeMedida"),
    }


def collect_day(
    client: PncpClient,
    day: date,
    raw: Path = RAW,
    progress: Callable[[int], None] | None = None,
) -> DayResult:
    raw.mkdir(parents=True, exist_ok=True)
    path, done = raw / f"{day.isoformat()}.jsonl", raw / f"{day.isoformat()}.done"
    result = DayResult(day)
    if done.exists():
        result.skipped = True
        return result

    # Tenders already written by an interrupted run are not fetched again. A run killed
    # in the middle of a line leaves it incomplete; it is cut off before appending, or
    # the next line would be glued to it.
    seen: set[str] = set()
    if path.exists():
        content = path.read_bytes()
        if content and not content.endswith(b"\n"):
            content = content[: content.rfind(b"\n") + 1]
            path.write_bytes(content)
        for line in content.decode("utf-8").splitlines():
            seen.add(json.loads(line)["control_number"])

    with path.open("a", encoding="utf-8") as out:
        for tender in client.tenders(day):
            if tender["numeroControlePNCP"] in seen:
                continue
            result.tenders += 1
            try:
                items = client.items(
                    tender["orgaoEntidade"]["cnpj"], tender["anoCompra"], tender["sequencialCompra"]
                )
            except httpx.HTTPError:
                # One broken tender should not stop a two-hour collection; it is counted.
                result.failed_tenders += 1
                continue
            # One write per tender, so a tender is on disk whole or (almost) not at all.
            records = (json.dumps(item_record(tender, i), ensure_ascii=False) for i in items)
            out.write("".join(r + "\n" for r in records))
            out.flush()
            result.items += len(items)
            if progress:
                progress(result.tenders)
    done.touch()
    return result


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(n) for n in range((end - start).days + 1)]
