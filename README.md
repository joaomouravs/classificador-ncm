# Classificador NCM

[![CI](https://github.com/joaomouravs/classificador-ncm/actions/workflows/ci.yml/badge.svg)](https://github.com/joaomouravs/classificador-ncm/actions/workflows/ci.yml)

Predicts the NCM chapter of a good bought by a Brazilian public body from its free-text
description. NCM (Nomenclatura Comum do Mercosul) is the product code used for taxes and
trade statistics; its first two digits are the chapter: 30 is pharmaceuticals, 84
machinery, 85 electrical equipment, 90 optical and medical instruments.

The labels cost nothing: some items published on the national procurement portal (PNCP)
already carry their NCM code. The project builds a dataset from them
([on Hugging Face](https://huggingface.co/datasets/joaovux/pncp-ncm-chapters)), and compares three
ways of solving a text classification problem in 2026 on the same held-out test set: a
classical model, a fine-tuned transformer and an LLM with no training data at all.

## Results

1,492 descriptions of catalogue items first seen on 23 October 2026, none of them, in
any wording, in training. 14 chapters plus `outros`. Per-class scores and the most
common confusions are in [results/REPORT.md](results/REPORT.md).

| Model | Macro-F1 | Accuracy | Training | Time per 1,000 items |
| --- | --: | --: | --- | --: |
| **BERTimbau, fine-tuned** ([on Hugging Face](https://huggingface.co/joaovux/ncm-chapter-bertimbau)) | **0.914** | **0.949** | 3.6 min on a Kaggle T4 | 2.1 s, GPU |
| TF-IDF character n-grams + logistic regression | 0.903 | 0.935 | about a minute on a laptop CPU | 0.5 s, CPU |
| Gemini 3.5 Flash-Lite, zero-shot | 0.611 | 0.757 | none: the 96 chapter titles | 131 s, API |
| Majority class | 0.036 | 0.337 | | |

Macro-F1 is the headline: four chapters are 85% of the training data, and a model that
only learned those would look fine on accuracy.

- **Fine-tuning wins on accuracy, not clearly on macro-F1.** A paired bootstrap over the
  test items puts BERTimbau's accuracy edge over TF-IDF at +1.4 points (95% interval
  +0.2 to +2.7; McNemar p = 0.026: 51 items only BERTimbau gets right, 30 only TF-IDF),
  but its macro-F1 edge at +1.0 points with an interval from −1.7 to +3.3. The small
  chapters, 3 to 15 test examples each, are too few to tell the two apart, and on
  validation TF-IDF is ahead (0.927 against 0.921).
- **A classical baseline is hard to beat on catalogue text.** Item descriptions are
  catalogue entries full of units, part numbers and abbreviations, which character
  n-grams handle without a tokenizer. TF-IDF gets within a point of a 110-million
  parameter transformer, trains in a minute on a laptop and needs no GPU to serve.
- **The LLM reads the material; the NCM reads the function.** Gemini's mistakes have a
  pattern. A plastic tongue depressor is a medical instrument (90), and it answered
  plastics (39) 26 times; a toner cartridge is a printer part (84), and it answered
  photographic chemicals (37). The NCM's interpretation rules classify by what an item
  is for, and a model trained on labels learns that convention. Some of the
  disagreements are the catalogue's own convention, though: a bottle of printer ink,
  which the catalogue files under printers (84), is arguably ink (32).
- **Zero-shot is the expensive option here.** About 140,000 input tokens per 1,000
  items, most of them the chapter list repeated in every request, for 29 points less
  macro-F1. It is still what you get on day one with no labelled data.

BERTimbau trains with [kaggle/train_bertimbau.py](kaggle/train_bertimbau.py) as a Kaggle
script on one T4 (`kaggle kernels push -p kaggle` with the dataset attached), and the
same file smoke-tests on a laptop CPU with a tiny model.

## What the data taught me

Every number comes from the 8,098 *pregões eletrônicos* with items published nationwide
from 19 to 23 October 2026: 214,743 items, 194,855 of them goods.

- **Only 9% of goods carry an NCM code.** A first probe of 150 tenders suggested 35%;
  five days nationwide gave 17,297 labelled items out of 194,855.
- **The code belongs to the catalogue, not to the buyer.** 17,296 of the 17,298 items
  with an NCM code also have a CATMAT code (the federal goods catalogue), and a CATMAT
  code carries the same NCM code in every tender: 3,233 codes seen more than once, zero
  disagreements. The plan assumed buyers type the code by hand and some get it wrong; in
  fact the label is the catalogue's mapping.
- **Splitting by text leaked a third of the test set.** Agencies reformat the same
  catalogue entry ("material: aço inoxidável" against "Material:Aço Inoxidável"). With
  one row per normalised description and test as the descriptions first seen on the last
  day, 766 of 2,309 test rows were catalogue items already in training. The split is now
  by catalogue item: all wordings of an entry are in the same split.
- **A class absent from the test set should not count.** Chapter 40 (rubber) has
  training examples but none on the test day. Counting its F1 as 0 lowered every
  model's macro-F1 by a fifteenth, for a class the test cannot measure.
- **`str.splitlines()` is not "split on newlines".** It also breaks on U+2028, which
  `json.dumps(ensure_ascii=False)` writes unescaped; one description in 214,743 had it.

## How it works

```
collect ──> build ──> train / ask ──> score ──> report
 PNCP API    dedupe,   TF-IDF (CPU),   same       results/REPORT.md
 items of    split by  BERTimbau (GPU), metrics    from results/*.json
 each tender catalogue Gemini (API)    for all
```

- **collect** pages through the PNCP search API for every *pregão eletrônico* of a day
  and downloads each tender's items, one JSON line per item, resumable by day.
- **build** keeps goods with an NCM code, labels each with its chapter, keeps one row
  per normalised description, groups rows by catalogue item, and splits by group: test
  is the catalogue items first seen on the last day, validation a stable 10% of the rest
  by hash. Chapters with fewer than 50 training examples become `outros`.
- **baselines** trains the majority class and the TF-IDF model, choosing the
  regularisation strength on validation.
- **zero-shot** sends the test descriptions to an LLM in batches of 25 with the 96
  chapter titles and a strict JSON schema whose only allowed answers are chapter codes.
  The chapter is mapped to a class in code, so the model never has to guess what
  `outros` means. Answers are cached on disk; re-scoring calls nothing.
- Every model writes the same JSON to `results/`, with its test predictions, and the
  report is built only from those files, whether the model ran on a laptop, on Kaggle
  or behind an API. The report also asks whether the best model beats the runner-up by
  more than the test set's noise: a paired bootstrap over test items for both metrics,
  and McNemar's exact test for accuracy.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run ncm collect --from 2026-10-19 --to 2026-10-23   # a couple of hours, resumable
uv run ncm build
uv run ncm baselines
uv run ncm zero-shot --model gemini:gemini-3.5-flash-lite --sample 1492   # GEMINI_API_KEY in .env
uv run ncm report
```

`uv run ncm publish-dataset` uploads the splits to the Hugging Face Hub with a dataset
card generated from the build's statistics, and `uv run ncm publish-model <dir>` the
fine-tuned model downloaded from Kaggle, with a model card generated from `results/`
(`HF_USERNAME` and `HF_TOKEN` in `.env`). Neither card has a number typed by hand.

The design and its milestones are in [docs/design.md](docs/design.md).

## Limitations

- **Catalogue text only.** Every labelled description is a catalogue entry. 79% of the
  goods are bought without a catalogue code, described in the agency's own words, and
  carry no NCM at all; they are where a classifier would be useful, and the test set
  says nothing about them.
- **The catalogue's mapping is taken as truth.** A CATMAT entry with the wrong NCM code
  is wrong in every row that uses it.
- **Five days of tenders.** Seasonal purchases outside the window are missing, and seven
  of the fifteen classes have fewer than 20 test examples (one has none), so their
  per-class scores are noisy.

## License

MIT for the code. PNCP data is Brazilian open government data.
