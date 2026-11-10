"""From raw items to train, validation and test splits.

The decisions that matter, each against a problem seen in the data:

- One example per distinct description. Government catalogue text repeats across
  tenders (873 distinct descriptions among 1,087 labelled items in the first probe), and
  keeping every copy would weight the model, and the metrics, towards whatever happened
  to be bought most.
- The label is the NCM chapter, the first two digits. When the same description carries
  different chapters in different tenders, the majority wins, and how often that happens
  is reported: it is a lower bound on the label noise.
- Splits are decided per catalogue item, not per description. Almost every item with
  an NCM code also has a CATMAT catalogue code, and the NCM belongs to the catalogue
  item: one code, one chapter, in every tender. Agencies reformat the same item's text
  ("material: aço inoxidável" against "Material:Aço Inoxidável"), so a split by text
  alone put a third of the test items' catalogue entries in training too. All the
  descriptions of one catalogue item (or one description, without a code) are a group,
  and a group is in exactly one split.
- Test is the groups first seen on the last collected day(s), so the test set looks
  like what comes next rather than a random slice of the past.
- Chapters with fewer than MIN_EXAMPLES training examples become "outros".
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

MIN_EXAMPLES = 50
OTHER = "outros"
VALIDATION_SHARE = 0.1


def normalize(description: str) -> str:
    """The key that decides whether two descriptions are the same text."""
    text = unicodedata.normalize("NFKD", description.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def chapter(code: Any) -> str | None:
    """The NCM chapter ("01" to "97"), or None if the code is missing or invalid.

    A code that lost its leading zero somewhere upstream (seven digits instead of eight)
    is padded back to an even length before taking the first two digits.
    """
    if code is None:
        return None
    digits = re.sub(r"\D", "", str(code))
    if not digits:
        return None
    if len(digits) % 2:
        digits = "0" + digits
    head = digits[:2]
    return head if 1 <= int(head) <= 97 else None


@dataclass
class Stats:
    items: int = 0
    material_items: int = 0
    labelled_items: int = 0
    distinct_descriptions: int = 0
    repeated_descriptions: int = 0
    conflicting_descriptions: int = 0
    labelled_with_catalog: int = 0
    catalog_items: int = 0
    groups: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    test_days: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    distribution: dict[str, dict[str, int]] = field(default_factory=dict)


def read_raw(paths: Iterable[Path]) -> pd.DataFrame:
    # Split on newlines only: str.splitlines() also breaks on U+2028 and other
    # separators, which json.dumps(ensure_ascii=False) leaves unescaped in a description.
    rows = [
        json.loads(line)
        for p in paths
        for line in p.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    return pd.DataFrame(rows)


def build(items: pd.DataFrame, test_days: int = 1) -> tuple[dict[str, pd.DataFrame], Stats]:
    stats = Stats(items=len(items))
    material = items.loc[items["kind"] == "M"].copy()
    stats.material_items = len(material)
    material["chapter"] = material["ncm"].map(chapter)
    labelled = material.loc[material["chapter"].notna() & (material["description"].str.len() > 0)]
    labelled = labelled.assign(key=labelled["description"].map(normalize))
    if "catalog_code" not in labelled:
        labelled = labelled.assign(catalog_code=None)
    # Series.map turns a None result into NaN, so missing codes are tested with isna.
    code = labelled["catalog_code"].map(lambda c: None if pd.isna(c) or c == "" else str(c))
    group = [
        f"text:{k}" if pd.isna(c) else f"cat:{c}"
        for c, k in zip(code, labelled["key"], strict=True)
    ]
    labelled = labelled.assign(catalog_code=code, group=group)
    stats.labelled_items = len(labelled)
    stats.labelled_with_catalog = int(code.notna().sum())
    stats.catalog_items = len(set(code.dropna()))

    groups = labelled.groupby("key", sort=True)
    examples = pd.DataFrame(
        {
            # The first wording seen; copies differ only in case, accents or spacing.
            "text": groups["description"].first(),
            "chapter": groups["chapter"].agg(lambda c: Counter(c).most_common(1)[0][0]),
            "chapters_seen": groups["chapter"].nunique(),
            "occurrences": groups.size(),
            "first_seen": groups["published_at"].min(),
            "catalog_code": groups["catalog_code"].first(),
            "group": groups["group"].first(),
        }
    ).reset_index()
    stats.distinct_descriptions = len(examples)
    repeated = examples[examples["occurrences"] > 1]
    stats.repeated_descriptions = len(repeated)
    stats.conflicting_descriptions = len(repeated.loc[repeated["chapters_seen"] > 1])

    stats.groups = len(set(examples["group"]))

    days = sorted(examples["first_seen"].unique())
    stats.test_days = [str(d) for d in days[-test_days:]]
    group_first_seen = examples.groupby("group")["first_seen"].transform("min")
    is_test = group_first_seen.isin(stats.test_days)
    # A stable hash of the group, so validation never changes between builds and a
    # catalogue item's descriptions all land on the same side.
    bucket = examples["group"].map(lambda g: int(hashlib.sha256(g.encode()).hexdigest()[:8], 16))
    is_validation = ~is_test & (bucket % 1000 < VALIDATION_SHARE * 1000)
    split = pd.Series("train", index=examples.index)
    split[is_validation] = "validation"
    split[is_test] = "test"
    examples["split"] = split

    train_counts = examples.loc[examples["split"] == "train", "chapter"].value_counts()
    keep = sorted(train_counts[train_counts >= MIN_EXAMPLES].index)
    stats.classes = [*keep, OTHER]
    examples["label"] = examples["chapter"].where(examples["chapter"].isin(keep), OTHER)

    columns = ["text", "label", "chapter", "catalog_code", "occurrences", "first_seen"]
    splits = {
        name: examples.loc[examples["split"] == name, columns].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    stats.splits = {name: len(frame) for name, frame in splits.items()}
    stats.distribution = {
        name: {k: int(v) for k, v in frame["label"].value_counts().sort_index().items()}
        for name, frame in splits.items()
    }
    return splits, stats


def write(splits: dict[str, pd.DataFrame], stats: Stats, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats.__dict__, indent=1) + "\n", encoding="utf-8")
