from __future__ import annotations

import json

import pandas as pd
import pytest

from ncm import build as build_module
from ncm.build import OTHER, build, chapter, normalize


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("85258019", "85"),
        ("852580", "85"),
        ("90", "90"),
        (30049099, "30"),
        ("4029900", "04"),  # lost its leading zero: 0402.99.00
        ("9.018", "90"),
        ("99", None),  # no chapter 99 in the NCM
        ("", None),
        (None, None),
    ],
)
def test_chapter(code, expected):
    assert chapter(code) == expected


def test_normalize_ignores_case_accents_and_spacing():
    assert normalize("  Câmera  Vídeo\nDe Segurança ") == normalize("camera video de seguranca")


def item(description, ncm, day, kind="M", catalog_code=None):
    return {
        "control_number": f"{day}-{description}",
        "published_at": day,
        "uf": "RJ",
        "item_number": 1,
        "description": description,
        "kind": kind,
        "ncm": ncm,
        "catalog_code": catalog_code,
        "unit": "Unidade",
    }


def frame(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def few_examples(monkeypatch):
    monkeypatch.setattr(build_module, "MIN_EXAMPLES", 2)


def test_one_example_per_description_with_the_majority_chapter(few_examples):
    rows = [
        item("Dipirona 500mg", "30049099", "2026-10-20"),
        item("DIPIRONA  500 MG", "3004", "2026-10-20"),  # not the same text: "500 mg"
        item("dipirona 500mg", "30049099", "2026-10-21"),
        item("Dipirona 500mg ", "90189099", "2026-10-21"),  # a wrong code, outvoted
        item("Serviço de limpeza", None, "2026-10-20", kind="S"),
        item("Caneta azul", None, "2026-10-20"),
    ]

    splits, stats = build(frame(rows), test_days=1)

    everything = pd.concat(splits.values())
    dipirona = everything[everything["text"].str.lower().str.startswith("dipirona 500mg")]
    assert len(dipirona) == 1
    assert dipirona.iloc[0]["chapter"] == "30"
    assert dipirona.iloc[0]["occurrences"] == 3
    assert (stats.items, stats.material_items, stats.labelled_items) == (6, 5, 4)
    assert (stats.distinct_descriptions, stats.repeated_descriptions) == (2, 1)
    assert stats.conflicting_descriptions == 1


def test_test_split_is_what_first_appears_on_the_last_day(few_examples):
    rows = [item(f"Seringa {n}ml", "90183119", "2026-10-20") for n in range(5)]
    rows += [item(f"Seringa {n}ml", "90183119", "2026-10-21") for n in range(3)]  # repeats
    rows += [item(f"Luva tamanho {n}", "40151900", "2026-10-21") for n in range(4)]

    splits, stats = build(frame(rows), test_days=1)

    assert stats.test_days == ["2026-10-21"]
    test_texts = set(splits["test"]["text"])
    assert test_texts == {f"Luva tamanho {n}" for n in range(4)}
    train_and_validation = set(splits["train"]["text"]) | set(splits["validation"]["text"])
    assert not test_texts & train_and_validation


def test_a_catalogue_item_reworded_later_stays_out_of_test(few_examples):
    rows = [item(f"Seringa {n}ml", "90183119", "2026-10-20") for n in range(5)]
    # The same CATMAT item, reformatted by another agency on the test day.
    rows += [
        item(
            "Lima odontológica material: aço, comprimento: 21",
            "90184999",
            "2026-10-20",
            catalog_code="473504",
        )
    ]
    rows += [
        item(
            "Lima Odontológica, Material:Aço, Comprimento:21 MM",
            "90184999",
            "2026-10-21",
            catalog_code="473504",
        )
    ]
    # A catalogue item new on the test day: both of its wordings are test.
    rows += [item("Luva material: látex", "40151900", "2026-10-21", catalog_code="443645")]
    rows += [item("Luva, Material:Látex", "40151900", "2026-10-21", catalog_code="443645")]

    splits, stats = build(frame(rows), test_days=1)

    assert set(splits["test"]["catalog_code"]) == {"443645"}
    assert len(splits["test"]) == 2
    seen = pd.concat([splits["train"], splits["validation"]])
    assert seen["catalog_code"].tolist().count("473504") == 2
    assert (stats.catalog_items, stats.groups) == (2, 7)


def test_rare_chapters_become_other(few_examples):
    rows = [item(f"Seringa {n}ml", "90183119", "2026-10-20") for n in range(30)]
    rows += [item("Pneu aro 15", "40111000", "2026-10-20")]
    rows += [item("Luva nova", "40151900", "2026-10-21")]

    splits, stats = build(frame(rows), test_days=1)

    assert stats.classes == ["90", OTHER]
    everything = pd.concat(splits.values())
    assert set(everything.loc[everything["chapter"] == "40", "label"]) == {OTHER}


def test_validation_is_stable_and_disjoint(few_examples):
    rows = [item(f"Seringa {n}ml", "90183119", "2026-10-20") for n in range(400)]
    rows += [item("Luva nova", "40151900", "2026-10-21")]

    first, _ = build(frame(rows), test_days=1)
    second, _ = build(frame(rows), test_days=1)

    assert first["validation"]["text"].tolist() == second["validation"]["text"].tolist()
    assert 20 < len(first["validation"]) < 60  # about 10% of 400
    assert not set(first["validation"]["text"]) & set(first["train"]["text"])


def test_read_raw_keeps_descriptions_with_unicode_line_separators(tmp_path):
    rows = [{"description": "Caneta\u2028azul", "kind": "M"}, {"description": "Lápis", "kind": "M"}]
    path = tmp_path / "2026-10-19.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")

    assert build_module.read_raw([path])["description"].tolist() == ["Caneta\u2028azul", "Lápis"]
