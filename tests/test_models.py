from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from ncm.baselines import majority, tfidf
from ncm.evaluate import score
from ncm.results import build_report, record

LABELS = ["30", "40", "90", "outros"]


def test_a_class_never_predicted_still_counts_against_macro_f1():
    truth = ["90", "90", "90", "30"]
    predicted = ["90", "90", "90", "90"]

    scores = score(truth, predicted, LABELS)

    assert scores.accuracy == 0.75
    # 90 has F1 6/7; 30 has 0; 40 and outros have no examples and score 0 as well.
    assert scores.macro_f1 == pytest.approx((6 / 7) / 4)
    assert scores.per_class["30"]["recall"] == 0
    assert scores.confusions == [("30", "90", 1)]


WORDS = {
    "30": ["dipirona", "amoxicilina", "paracetamol", "ibuprofeno", "omeprazol"],
    "40": ["luva latex", "pneu aro", "camara de ar", "borracha", "mangueira"],
    "90": ["seringa", "termometro", "estetoscopio", "cateter", "oximetro"],
    "outros": ["caneta", "papel a4", "grampeador", "pasta", "envelope"],
}


def dataset(seed: int, size: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        label = rng.choice(LABELS)
        rows.append(
            {"text": f"{rng.choice(WORDS[label])} {rng.randint(1, 500)} unidade", "label": label}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def splits():
    return {"train": dataset(1, 400), "validation": dataset(2, 80), "test": dataset(3, 80)}


def test_tfidf_beats_majority_on_separable_text(splits):
    floor = majority(splits["train"])
    model = tfidf(splits["train"], splits["validation"], LABELS)

    texts, truth = splits["test"]["text"].tolist(), splits["test"]["label"].tolist()
    assert score(truth, floor.predict(texts), LABELS).macro_f1 < 0.2
    assert score(truth, model.predict(texts), LABELS).macro_f1 > 0.9
    assert model.params["C"] in (1.0, 4.0, 16.0)


def test_results_and_report(splits, tmp_path):
    model = tfidf(splits["train"], splits["validation"], LABELS)
    record("tfidf-logreg", model.predict, splits, LABELS, {"params": model.params}, tmp_path)
    floor = majority(splits["train"])
    record("majority", floor.predict, splits, LABELS, {}, tmp_path)

    saved = json.loads((tmp_path / "tfidf-logreg.json").read_text(encoding="utf-8"))
    assert saved["test"]["examples"] == 80
    assert saved["ms_per_1000_items"] > 0

    report = build_report(tmp_path)
    rows = [
        line
        for line in report.splitlines()
        if line.startswith("| tfidf") or line.startswith("| majority")
    ]
    assert rows[0].startswith("| tfidf-logreg")  # best first
    assert "## majority" in report
