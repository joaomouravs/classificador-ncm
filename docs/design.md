# Classificador NCM: design

Status: M0, M1 and M3 implemented; M2 (BERTimbau) and M4 (Space, cards) next. The data
changed three decisions below; see "Deviations" at the end.

## Problem

Every item a Brazilian public body buys is described in free text ("Câmera Vídeo De
Segurança tipo sensor: cmos 1/3\", resolução horizontal..."). The PNCP API can also carry
the item's NCM code, the Mercosur product classification used for taxes and trade
statistics, but buyers fill it in for only about a third of the items: 35% of the 3,133
material items in a probe of 150 tenders from 22 states on 20 October 2026.

This project trains a model that reads an item description and predicts its NCM chapter
(the first two digits: 30 is pharmaceuticals, 85 electrical equipment, 90 optical and
medical instruments). The items that do carry a code are the labels, so the dataset
costs nothing to annotate.

It is also a fair comparison of three ways of solving a text classification problem in
2026: a classical model, a fine-tuned small transformer, and a large language model
asked zero-shot.

## Goals

1. A public dataset of item descriptions with NCM chapters, built from PNCP, published
   on Hugging Face and Kaggle with a dataset card.
2. Three models evaluated on the same held-out test set:
   - TF-IDF over character n-grams with logistic regression (scikit-learn, CPU),
   - BERTimbau (a Portuguese BERT) fine-tuned on a free Kaggle GPU, in a public notebook,
   - an LLM (Gemini Flash-Lite) given the chapter list, zero-shot.
3. The best small model published on Hugging Face with a model card, and a demo Space.
4. A comparison table: macro-F1, accuracy, latency and cost per 1,000 items.

## Non-goals

- The full 8-digit NCM. Chapter level has enough examples per class to learn and to
  measure; deeper levels are a later step if the data supports it.
- Services. They are classified by NBS, a different taxonomy with far fewer filled codes.

## Data

- **Source.** The PNCP search API for tenders, then each tender's items endpoint; only
  material items with an NCM code. One request per tender at about one per second, so a
  week of tenders nationwide is a background job of a couple of hours.
- **Labels.** The first two digits of `ncmNbsCodigo`. Buyers sometimes fill only the
  chapter, so codes of two digits or more all count.
- **Noise.** Buyers pick codes by hand, and some are wrong. A sample of test items is
  checked by hand and the label error rate reported, because it bounds the accuracy any
  model can show.
- **Leakage.** Descriptions repeat (873 unique among 1,087 labelled items in the probe):
  the government catalogue's text is reused across tenders. Splitting rows at random
  would put the same description in training and test. The split is by time (test is the
  last week collected) and by normalised description (a description in test never
  appears in training).
- **Classes.** Chapters with at least 50 training examples; the rest grouped as
  `outros`. The distribution is skewed (chapters 90, 85, 84 and 30 are most of it), so
  the headline metric is macro-F1, with accuracy and a confusion matrix beside it.

## Models

| Model | Where it runs | Why it is here |
| --- | --- | --- |
| Majority class | anywhere | the floor every model must beat |
| TF-IDF + logistic regression | laptop CPU, seconds | strong on catalogue-like text; the baseline that justifies anything heavier |
| BERTimbau base, fine-tuned | Kaggle T4, public notebook | does pretrained Portuguese help beyond character n-grams? |
| Gemini Flash-Lite, zero-shot | API, free tier | what you get with no training data at all, and what it costs |

The LLM is scored on a fixed sample of the test set (the free tier limits requests), and
the sample is reported with its size.

## Publishing

- Dataset: Hugging Face Datasets and Kaggle, with source, license (PNCP data is open
  government data), collection dates, split rules and known noise.
- Model: Hugging Face Hub with a model card (intended use, metrics per class,
  limitations, label noise).
- Demo: a Gradio Space on the free CPU tier, typing a description and seeing the top
  chapters with probabilities.
- Write-up: the comparison and what it says about when fine-tuning is worth it.

## Stack

Python 3.12 with uv, httpx, pandas, scikit-learn, Hugging Face `datasets`,
`transformers` and `evaluate` (training on Kaggle), Gradio for the demo, pytest.

## Milestones

- **M0**: collection script and dataset build (cleaning, dedupe, splits), with tests.
- **M1**: majority and TF-IDF baselines, evaluation module, first report.
- **M2**: BERTimbau fine-tuning notebook on Kaggle; model on Hugging Face.
- **M3**: LLM zero-shot on the test sample; comparison table; label-noise check.
- **M4**: Gradio Space, dataset and model cards, README, write-up.

## Deviations

Measured on the full collection (19 to 23 October 2026), and implemented:

- **Coverage.** 9% of goods carry an NCM code, not 35%: 17,297 labelled items out of
  194,855 goods. The dataset is about 13,000 distinct descriptions instead of the tens
  of thousands planned.
- **Labels and noise.** The NCM code comes with the CATMAT catalogue entry (17,296 of
  17,298 labelled items have one, and a catalogue code never changes chapter between
  tenders). The planned hand check of label noise becomes a limitation instead: the
  catalogue's mapping is taken as truth.
- **Leakage.** Splitting by normalised description put 766 of 2,309 test rows on
  catalogue items already in training, reworded. Splits are now by catalogue item.
- **Test window.** Test is the catalogue items first seen on the last day, not the last
  week: five days collected, and one day already gives 1,492 test rows.
- **LLM sample.** The whole test set fits the free tier (60 batched requests), so the
  LLM is scored on all of it; the sampling code stays for larger test sets.
- **Macro-F1** leaves out classes with no test examples, which would otherwise count as
  0 for every model.
