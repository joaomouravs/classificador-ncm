"""The two public PNCP endpoints this project needs: tenders by day, and their items.

The same client as in Radar de Editais (github.com/joaomouravs/radar-editais), cut down.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import httpx

BASE_URL = "https://pncp.gov.br"
USER_AGENT = "classificador-ncm (+https://github.com/joaomouravs/classificador-ncm)"
PREGAO_ELETRONICO = 6
# The search endpoint answers 400 to any page size above 50.
SEARCH_PAGE_SIZE = 50
ITEMS_PAGE_SIZE = 50

Json = dict[str, Any]


def new_http_client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
    )


class PncpClient:
    def __init__(
        self,
        http: httpx.Client,
        *,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._retries = retries
        self._sleep = sleep

    def tenders(self, day: date, modality: int = PREGAO_ELETRONICO) -> Iterator[Json]:
        """Every tender published on `day`, nationwide."""
        page = 1
        while True:
            resp = self._get(
                "/api/consulta/v1/contratacoes/publicacao",
                {
                    "dataInicial": day.strftime("%Y%m%d"),
                    "dataFinal": day.strftime("%Y%m%d"),
                    "codigoModalidadeContratacao": modality,
                    "pagina": page,
                    "tamanhoPagina": SEARCH_PAGE_SIZE,
                },
            )
            # No results at all comes back as 204 with an empty body.
            if resp.status_code == 204:
                return
            body = resp.json()
            yield from body["data"]
            if body["paginasRestantes"] == 0:
                return
            page += 1

    def items(self, cnpj: str, year: int, sequence: int) -> list[Json]:
        items: list[Json] = []
        page = 1
        while True:
            resp = self._get(
                f"/api/pncp/v1/orgaos/{cnpj}/compras/{year}/{sequence}/itens",
                {"pagina": page, "tamanhoPagina": ITEMS_PAGE_SIZE},
            )
            batch = resp.json() if resp.status_code != 204 else []
            items.extend(batch)
            if len(batch) < ITEMS_PAGE_SIZE:
                return items
            page += 1

    def _get(self, url: str, params: dict[str, str | int]) -> httpx.Response:
        """GET with retries on network errors, 429 and 5xx, backing off 1s, 2s, 4s..."""
        attempt = 0
        while True:
            try:
                resp = self._http.get(url, params=params)
            except httpx.TransportError:
                if attempt == self._retries:
                    raise
            else:
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt == self._retries:
                    resp.raise_for_status()
                    return resp
            self._sleep(2.0**attempt)
            attempt += 1
