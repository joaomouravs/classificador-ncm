from __future__ import annotations

import json
from datetime import date

import httpx

from ncm.collect import collect_day, days
from ncm.pncp import PncpClient


def tender(n: int) -> dict:
    return {
        "numeroControlePNCP": f"0000000000000{n}-1-00000{n}/2026",
        "dataPublicacaoPncp": "2026-10-20T04:00:08",
        "unidadeOrgao": {"ufSigla": "RJ"},
        "orgaoEntidade": {"cnpj": f"0000000000000{n}"},
        "anoCompra": 2026,
        "sequencialCompra": n,
    }


class FakePortal:
    def __init__(self, tenders: int = 3, broken: frozenset[int] = frozenset()):
        self.tenders = [tender(n) for n in range(1, tenders + 1)]
        self.broken = broken
        self.item_requests: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/contratacoes/publicacao"):
            return httpx.Response(200, json={"data": self.tenders, "paginasRestantes": 0})
        sequence = int(request.url.path.split("/")[-2])
        self.item_requests.append(sequence)
        if sequence in self.broken:
            return httpx.Response(404)
        return httpx.Response(
            200,
            json=[
                {
                    "numeroItem": 1,
                    "descricao": f"Item do pregão {sequence}",
                    "materialOuServico": "M",
                    "ncmNbsCodigo": "90183119",
                    "catalogoCodigoItem": "123",
                    "unidadeMedida": "Unidade",
                }
            ],
        )


def client(portal: FakePortal) -> PncpClient:
    http = httpx.Client(base_url="https://pncp.test", transport=httpx.MockTransport(portal))
    return PncpClient(http, sleep=lambda _: None)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_day_is_written_item_by_item_and_marked_done(tmp_path):
    result = collect_day(client(FakePortal()), date(2026, 10, 20), tmp_path)

    assert (result.tenders, result.items, result.failed_tenders) == (3, 3, 0)
    rows = lines(tmp_path / "2026-10-20.jsonl")
    assert rows[0] == {
        "control_number": "00000000000001-1-000001/2026",
        "published_at": "2026-10-20",
        "uf": "RJ",
        "item_number": 1,
        "description": "Item do pregão 1",
        "kind": "M",
        "ncm": "90183119",
        "catalog_code": "123",
        "unit": "Unidade",
    }
    assert (tmp_path / "2026-10-20.done").exists()


def test_a_finished_day_is_skipped(tmp_path):
    collect_day(client(FakePortal()), date(2026, 10, 20), tmp_path)
    portal = FakePortal()

    assert collect_day(client(portal), date(2026, 10, 20), tmp_path).skipped
    assert portal.item_requests == []


def test_an_interrupted_day_resumes_without_refetching(tmp_path):
    collect_day(client(FakePortal(tenders=2)), date(2026, 10, 20), tmp_path)
    (tmp_path / "2026-10-20.done").unlink()  # as if it had stopped before the end
    portal = FakePortal(tenders=3)

    collect_day(client(portal), date(2026, 10, 20), tmp_path)

    assert portal.item_requests == [3]
    assert len(lines(tmp_path / "2026-10-20.jsonl")) == 3


def test_a_broken_tender_is_counted_not_fatal(tmp_path):
    result = collect_day(client(FakePortal(broken=frozenset({2}))), date(2026, 10, 20), tmp_path)

    assert (result.items, result.failed_tenders) == (2, 1)


def test_days_are_inclusive():
    assert days(date(2026, 10, 19), date(2026, 10, 21)) == [
        date(2026, 10, 19),
        date(2026, 10, 20),
        date(2026, 10, 21),
    ]


def test_a_line_cut_by_a_crash_is_dropped_before_resuming(tmp_path):
    collect_day(client(FakePortal(tenders=2)), date(2026, 10, 20), tmp_path)
    (tmp_path / "2026-10-20.done").unlink()
    path = tmp_path / "2026-10-20.jsonl"
    path.write_bytes(path.read_bytes() + b'{"control_number": "000000000')  # killed mid-line

    collect_day(client(FakePortal(tenders=3)), date(2026, 10, 20), tmp_path)

    assert [row["control_number"][-6:] for row in lines(path)] == ["1/2026", "2/2026", "3/2026"]
