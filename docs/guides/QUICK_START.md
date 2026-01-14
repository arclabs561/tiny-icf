# Quick Start Guide

## Best Models

### For General ICF Prediction
```bash
# Use rank_weight=5.0 model
uv run scripts/evaluate_model.py \
    --model models/model_diagnostic_rank5.pt \
    --data data/word_frequency.csv
```
**Results:** Spearman=0.28, MAE=0.31, Jabberwocky=40%

### For Rare Word Detection
```bash
# Use rank_weight=10.0 model
uv run scripts/evaluate_model.py \
    --model models/model_diagnostic_rank10.pt \
    --data data/word_frequency.csv
```
**Results:** Spearman=0.24, MAE=0.33, Jabberwocky=80%

## Training

### Diagnostic Training
```bash
uv run scripts/train_diagnostic.py \
    --data data/word_frequency.csv \
    --epochs 30 \
    --rank-weight 5.0 \
    --output models/model.pt
```

### Compare All Experiments
```bash
./scripts/evaluate_all_experiments.sh
python3 scripts/compare_experiments.py
```

## Results Summary

- ✅ rank_weight=5.0: Best overall (Spearman=0.28, MAE=0.31)
- ✅ rank_weight=10.0: Best rare words (Jabberwocky=80%)
- ⚠️ Calibrated: Needs investigation

See `FINAL_REPORT.md` for complete details.
