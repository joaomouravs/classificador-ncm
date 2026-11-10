"""Zero-shot classification with an LLM: no training data, only the chapter list.

The decisions that matter:

- The model picks among all 96 NCM chapters, not among the dataset's classes. It has
  never seen the training set, so it cannot know which rare chapters were grouped as
  "outros"; the mapping from chapter to class is done here, in code, the same way the
  dataset build does it.
- Descriptions go in batches of BATCH, numbered, and the answer is strict JSON with one
  chapter per number. One request per item would spend the free tier's requests 25
  times faster for the same tokens.
- Every answer is cached on disk by the exact request, so re-scoring never calls the
  API, and an interrupted run resumes where it stopped.
- The free tier caps requests, so the test set is sampled: the same rows every time,
  chosen by a hash of the description, and every other model is scored on them too.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ncm.build import OTHER, normalize
from ncm.chapters import CHAPTERS

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
BATCH = 25
# Long technical specifications are cut: the first sentences name the product, and the
# rest (dimensions, standards, packaging) costs tokens without changing the chapter.
MAX_CHARS = 600

SYSTEM = (
    "You classify goods bought by Brazilian public bodies into chapters of the NCM "
    "(Nomenclatura Comum do Mercosul), the first two digits of the code, which are the "
    "Harmonized System chapters. For every numbered item description, answer the chapter "
    "of the goods it describes, judging by what the item is, not by who buys it or what "
    "it is for. Answer every number exactly once.\n\nChapters:\n"
    + "\n".join(f"{code}: {name}" for code, name in CHAPTERS.items())
)


class LlmError(Exception):
    pass


def sample_indices(texts: Sequence[str], size: int) -> list[int]:
    """`size` rows chosen by a hash of their normalised text: a fixed, unbiased sample
    that does not depend on the order of the file."""
    keyed = sorted(
        range(len(texts)), key=lambda i: hashlib.sha256(normalize(texts[i]).encode()).hexdigest()
    )
    return sorted(keyed[:size])


def to_label(chapter: str | None, classes: Sequence[str]) -> str:
    """A predicted chapter as a dataset class. An unanswered item is "invalid", which
    is never a class and so always counts as an error."""
    if chapter is None:
        return "invalid"
    return chapter if chapter in classes else OTHER


def messages(descriptions: Sequence[str]) -> list[dict[str, str]]:
    numbered = "\n".join(
        f"{i}. {' '.join(d.split())[:MAX_CHARS]}" for i, d in enumerate(descriptions, 1)
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": numbered}]


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "chapter": {"type": "string", "enum": list(CHAPTERS)},
                },
                "required": ["number", "chapter"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}


def parse(content: str, size: int) -> list[str | None]:
    """Chapters by position. A number missing, repeated or out of range, or a chapter
    that does not exist, leaves that item unanswered instead of failing the batch."""
    chapters: list[str | None] = [None] * size
    try:
        answers = json.loads(content).get("answers", [])
    except (ValueError, AttributeError):
        return chapters
    seen: set[int] = set()
    for answer in answers:
        number, chapter = answer.get("number"), answer.get("chapter")
        if not isinstance(number, int) or not 1 <= number <= size or number in seen:
            continue
        seen.add(number)
        if chapter in CHAPTERS:
            chapters[number - 1] = chapter
    return chapters


@dataclass
class Usage:
    requests: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: list[int] = field(default_factory=list)


class Classifier:
    def __init__(
        self,
        http: httpx.Client,
        model: str,
        api_key: str,
        cache_dir: Path,
        *,
        retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        provider, _, self.name = model.partition(":")
        if provider not in PROVIDERS or not self.name:
            raise LlmError(f"model must be provider:name with provider in {list(PROVIDERS)}")
        self._base_url = PROVIDERS[provider][0]
        self._provider = provider
        self._http = http
        self._key = api_key
        self._cache_dir = cache_dir
        self._retries = retries
        self._sleep = sleep
        self.usage = Usage()

    def classify(
        self, texts: Sequence[str], progress: Callable[[int], None] | None = None
    ) -> list[str | None]:
        chapters: list[str | None] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            chapters += parse(self._complete(messages(batch)), len(batch))
            if progress:
                progress(len(chapters))
        return chapters

    def _complete(self, conversation: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": conversation,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "chapters", "strict": True, "schema": SCHEMA},
            },
        }
        key = hashlib.sha256(
            json.dumps([self._provider, body], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = self._cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            self.usage.cached += 1
        else:
            hit = self._post(body)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(hit, ensure_ascii=False), encoding="utf-8")
            self.usage.requests += 1
        self.usage.tokens_in += hit["tokens_in"]
        self.usage.tokens_out += hit["tokens_out"]
        self.usage.latency_ms.append(hit["latency_ms"])
        return hit["content"]

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries. A free tier answers 429 when its per-minute quota is
        spent; the wait it asks for is honoured. Latency is the answering request's."""
        attempt = 0
        while True:
            start = time.monotonic()
            try:
                resp = self._http.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self._key}"},
                )
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise LlmError(repr(exc)) from exc
                wait = 2.0**attempt
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    return {
                        "content": data["choices"][0]["message"]["content"] or "",
                        "tokens_in": usage.get("prompt_tokens", 0),
                        "tokens_out": usage.get("completion_tokens", 0),
                        "latency_ms": round((time.monotonic() - start) * 1000),
                    }
                if resp.status_code != 429 and resp.status_code < 500:
                    raise LlmError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                if attempt == self._retries:
                    raise LlmError(f"HTTP {resp.status_code} after {attempt} retries")
                wait = _retry_after(resp) or (
                    min(15.0 * 2**attempt, 60.0) if resp.status_code == 429 else 2.0**attempt
                )
            self._sleep(wait)
            attempt += 1


def _retry_after(resp: httpx.Response) -> float | None:
    """The wait the provider asked for: a Retry-After header (Groq), or the retryDelay
    Gemini puts in the error body ("retryDelay": "37s")."""
    value = resp.headers.get("retry-after")
    if value is None:
        match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', resp.text)
        value = match.group(1) if match else None
    try:
        return min(float(value), 120.0) if value else None
    except ValueError:
        return None
