# Quick start

This repo is intentionally lean: it does **not** ship training data or trained checkpoints. You bring a frequency list (or download one), train, then predict/evaluate.

## Install (uv)

```bash
uv sync --extra dev
```

## Get data

The training input is a CSV with `word,count` (header optional).

Options:
- Download helpers: `./scripts/download_data.sh` or `uv run scripts/download_datasets.py`
- Provide your own frequency list

More details: `DATA_AND_MODELS.md` and `DATA_PREP.md`.

## Train

```bash
mkdir -p models

uv run tiny-icf-train \
  --data data/word_frequency.csv \
  --epochs 50 \
  --batch-size 64 \
  --output models/model.pt
```

## Predict

```bash
uv run tiny-icf-predict \
  --model models/model.pt \
  --words "the apple xylophone qzxbjk café 北京" \
  --detailed

uv run tiny-icf-predict \
  --model models/model.pt \
  --words "the apple xylophone qzxbjk café 北京" \
  --json
```

## Evaluate (including Jabberwocky Protocol)

```bash
uv run scripts/evaluate_model.py --model models/model.pt --data data/word_frequency.csv
uv run scripts/evaluate_model.py --model models/model.pt --jabberwocky-only
```

## Next

- `TRAINING_GUIDE.md` for training strategies/variants
- `../PROJECT_OVERVIEW.md` for motivation + design context
