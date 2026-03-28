# tiny-icf

Compressed character-level model for word commonality estimation.

Licensed under MIT.

ICF is normalized to \([0, 1]\): **0.0 = very common**, **1.0 = very rare**. The Rust `icf-estimator` and Python models use a max word length of 20 bytes; longer words are truncated.

```bash
uv sync --extra dev
# Differentiable Spearman (diffsort) is a default dependency; backend logged at train start

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

# OOV-focused prediction: avoid clamp-to-1.0 saturation on pseudo-words/composed words
uv run tiny-icf-predict \
  --model models/universal_50k_20ep.pt \
  --words "unfriendliness flimjam qzxbjk" \
  --detailed \
  --saturation-fix

# Optional: tune the saturation-fix parameters (defaults are Jabberwocky-safe)
uv run tiny-icf-predict \
  --model models/universal_50k_20ep.pt \
  --words "unfriendliness flimjam qzxbjk" \
  --detailed \
  --saturation-fix \
  --fix-center 1.23 \
  --fix-scale 0.15 \
  --fix-conf-weight 16

# (For evaluation) you can also pass these knobs to the downstream harness:
uv run python scripts/evaluate_downstream.py \
  --model models/universal_50k_20ep.pt \
  --data data/word_frequency.csv \
  --fix-center 1.23 \
  --fix-scale 0.15 \
  --fix-conf-weight 16
```

## Multi-task training

Optional auxiliary heads: token hygiene classification, language/era classification, temporal ICF prediction. See `scripts/train_all_fronts.py --help` for options. Monitor with `just check-training`.

## Data and models

No training data or model files are committed (large, user-specific). Train locally; artifacts go in `models/` (gitignored).

- **Pre-trained:** Model selection table and S3 download: `docs/guides/DATA_AND_MODELS.md`.
- **Publish:** `./scripts/upload_model_to_s3.sh models/<name>.pt s3://your-bucket/tiny-icf/`. After training: `just fit-calibration` then `just sync-s3` (or the sync command in DATA_AND_MODELS).
- **Data format:** CSV with `word,count` (optional header). See `tiny_icf.data.load_frequency_list`.

## Evaluate (Jabberwocky + MAE/Spearman)

```bash
# Full: Jabberwocky protocol + dataset metrics (add --calibration <path> for calibrated MAE)
uv run python scripts/evaluate_model.py --model models/<name>.pt --data data/word_frequency.csv

# Jabberwocky only (13 probe words)
uv run python scripts/evaluate_model.py --model models/<name>.pt --jabberwocky-only
```
Use `models/toy.pt` with `data/toy_word_frequency.csv` for the smoke-test model; use `multitask_all_fronts_v3b.pt` (or v3/v4) with `data/word_frequency.csv` for pre-trained. For English-only models with calibration: `just eval-en` or `just eval-en-spearman`.

## Development

```bash
uv pip install -e .   # optional: editable install for scripts that import tiny_icf
just ci               # lint (ruff + black) + pytest
```

Further documentation in `docs/`: `PROJECT_OVERVIEW.md`, `guides/QUICK_START.md`, `guides/TRAINING_GUIDE.md`, `guides/CALIBRATION_AND_RANKING_GUIDE.md`.
