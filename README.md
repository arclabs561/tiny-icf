# tiny-icf

Tiny byte-level model for estimating word commonality (ICF).

Licensed under MIT.

ICF is normalized to \([0, 1]\): **0.0 = very common**, **1.0 = very rare**.

```bash
uv sync --extra dev

# Train
uv run tiny-icf-train --help

# Predict
uv run tiny-icf-predict --help
```

## Quick smoke-test (no external downloads)

This trains a toy model from a tiny CSV. The model won’t be good, but it proves the end-to-end pipeline works.

```bash
mkdir -p data models

python3 - <<'PY'
import csv

rows = [
    ("the", 100000),
    ("and", 80000),
    ("apple", 1000),
    ("xylophone", 10),
    ("qzxbjk", 1),
]

with open("data/toy_word_frequency.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["word", "count"])
    w.writerows(rows)

print("wrote data/toy_word_frequency.csv")
PY

uv run tiny-icf-train --data data/toy_word_frequency.csv --epochs 1 --output models/toy.pt
uv run tiny-icf-predict --model models/toy.pt --words "the apple xylophone qzxbjk café 北京" --detailed
```

## Quick real-data run (small downloads)

This downloads two small public frequency lists (10k + 50k words), trains a model, and reports
task-shaped metrics (common-word filtering + gibberish-vs-common), plus baseline comparisons.

```bash
mkdir -p data models

# Downloads into data/ (and writes data/word_frequency.csv)
uv run python scripts/download_datasets.py

# Train a reasonably-good CPU model (start here)
uv run tiny-icf-train --data data/word_frequency.csv --epochs 20 --output models/universal_50k_20ep.pt --device cpu

# Baseline comparisons (Spearman/MAE + ranking overlap)
uv run python scripts/evaluate_with_baselines.py --model models/universal_50k_20ep.pt --data data/word_frequency.csv

# Downstream harness (OOV-style split + AUROC tasks + Jabberwocky)
uv run python scripts/evaluate_downstream.py --model models/universal_50k_20ep.pt --data data/word_frequency.csv
```

## Data and models

This repo intentionally does **not** include training data or trained model files (they’re large and user-specific).
See `docs/guides/DATA_AND_MODELS.md` for download/training workflows.

Training data format: CSV with `word,count` (header optional). See `tiny_icf.data.load_frequency_list`.

## Evaluate (including Jabberwocky Protocol)

```bash
uv run scripts/evaluate_model.py --model models/toy.pt --data data/toy_word_frequency.csv
uv run scripts/evaluate_model.py --model models/toy.pt --jabberwocky-only
```

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run black --check src tests
```

## Docs

Start with:
- `docs/PROJECT_OVERVIEW.md`
- `docs/guides/QUICK_START.md`
- `docs/guides/TRAINING_GUIDE.md`
