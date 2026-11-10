from __future__ import annotations

import json

import httpx
import pandas as pd

from ncm.evaluate import score
from ncm.llm import BATCH, MAX_CHARS, Classifier, messages, parse, sample_indices, to_label
from ncm.results import build_report, record, save

CLASSES = ["30", "90", "outros"]


def answer(*pairs: tuple[int, str]) -> str:
    return json.dumps({"answers": [{"number": n, "chapter": c} for n, c in pairs]})


def test_parse_leaves_bad_answers_unanswered_instead_of_failing_the_batch():
    content = answer((1, "30"), (1, "90"), (3, "99"), (4, "85"), (7, "30"))
    # 1 repeated (first wins), 2 missing, 3 not a chapter, 4 fine, 7 out of range.
    assert parse(content, 4) == ["30", None, None, "85"]
    assert parse("not json", 2) == [None, None]


def test_chapters_outside_the_classes_become_other():
    assert to_label("30", CLASSES) == "30"
    assert to_label("85", CLASSES) == "outros"
    assert to_label(None, CLASSES) == "invalid"


def test_sample_is_fixed_by_content_not_by_order():
    texts = [f"item {i}" for i in range(50)]
    chosen = {texts[i] for i in sample_indices(texts, 10)}
    assert len(chosen) == 10
    assert {texts[::-1][i] for i in sample_indices(texts[::-1], 10)} == chosen


def test_long_descriptions_are_cut_and_numbered():
    user = messages(["a  b\nc", "x" * 5000])[1]["content"]
    assert user.splitlines()[0] == "1. a b c"
    assert len(user.splitlines()[1]) == len("2. ") + MAX_CHARS


def test_classifier_waits_on_429_batches_and_caches(tmp_path):
    calls: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"details": [{"retryDelay": "7s"}]}})
        size = json.loads(request.content)["messages"][1]["content"].count("\n") + 1
        content = answer(*((n, "90") for n in range(1, size + 1)))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    texts = [f"seringa {i}" for i in range(BATCH + 3)]
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        first = Classifier(http, "gemini:test", "key", tmp_path, sleep=waits.append)
        assert first.classify(texts) == ["90"] * len(texts)
        assert waits == [7.0]
        assert first.usage.requests == 2  # two batches

        again = Classifier(http, "gemini:test", "key", tmp_path, sleep=waits.append)
        again.classify(texts)
        assert again.usage.cached == 2
        assert len(calls) == 3  # one 429, then two batches; the rerun used the cache


def test_report_scores_every_model_on_the_llm_sample(tmp_path):
    test = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": ["30", "30", "90", "90"]})
    splits = {"train": test, "validation": test, "test": test}
    record("always-30", lambda texts: ["30"] * len(texts), splits, CLASSES, {}, tmp_path)
    save(
        {
            "model": "llm-zero-shot",
            "ms_per_1000_items": 1.0,
            "validation": None,
            "test": score(["90", "90"], ["90", "90"], CLASSES).as_dict(),
            "test_predictions": ["90", "90"],
            "sample": {"indices": [2, 3], "labels": ["90", "90"], "of": 4},
        },
        tmp_path,
    )
    report = build_report(tmp_path)
    assert "## On the LLM's sample (2 of 4 test items)" in report
    sample_rows = report.split("## On the LLM's sample")[1].splitlines()
    # always-30 is wrong on both sampled rows, which are both chapter 90.
    assert "| always-30 | 0.000 | 0.000 |" in sample_rows
    assert any(r.startswith("| llm-zero-shot |") for r in sample_rows)
    assert "| llm-zero-shot | 2 | 0.333 | 1.000 | n/a |" in report
