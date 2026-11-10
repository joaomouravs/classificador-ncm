from __future__ import annotations

from ncm.chapters import CHAPTERS, OTHER_TITLE, title
from ncm.hub import dataset_card

STATS = {
    "items": 1000,
    "material_items": 800,
    "labelled_items": 300,
    "distinct_descriptions": 250,
    "repeated_descriptions": 40,
    "conflicting_descriptions": 5,
    "splits": {"train": 200, "validation": 20, "test": 30},
    "test_days": ["2026-10-23"],
    "classes": ["30", "85", "outros"],
    "distribution": {
        "train": {"30": 120, "85": 60, "outros": 20},
        "validation": {"30": 12, "85": 8},
        "test": {"30": 15, "85": 10, "outros": 5},
    },
}


def test_every_chapter_but_the_reserved_one_has_a_title():
    assert set(CHAPTERS) == {f"{n:02d}" for n in range(1, 98)} - {"77"}
    assert title("85").startswith("Máquinas")
    assert title("outros") == OTHER_TITLE


def test_card_numbers_come_from_the_stats():
    card = dataset_card(STATS, ("2026-10-19", "2026-10-23"), "someone/pncp-ncm-chapters")
    assert card.startswith("---\nlanguage:\n- pt\n")
    assert "- n<1K" in card
    assert "  - split: test\n    path: data/test.parquet" in card
    assert 'load_dataset("someone/pncp-ncm-chapters")' in card
    assert "| `30` | Produtos farmacêuticos | 120 | 12 | 15 |" in card
    # A class absent from a split shows zero instead of breaking the table.
    assert "| `outros` | Outros capítulos (poucos exemplos no treino) | 20 | 0 | 5 |" in card
    assert "from 2026-10-19 to 2026-10-23" in card
