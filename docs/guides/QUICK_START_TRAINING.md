# Quick Start: Training with Fixed Model

## What Was Fixed

Following `GOALS_CRITIQUE_AND_REFINEMENT.md` recommendations:

1. **Collapse Detection**: Monitors prediction variance, stops training if collapse detected
2. **Loss Component Logging**: Separately tracks Huber and Ranking loss
3. **Improved Initialization**: Better weight initialization to prevent collapse
4. **Higher Ranking Weight**: Default 5.0 (was 2.0) for stronger ranking signal
5. **Comprehensive Diagnostics**: Logs prediction stats, loss components, collapse events

## Quick Start

### Local Training (Recommended First)

```bash
# Single diagnostic run (30 epochs)
uv run --python 3.12 scripts/train_diagnostic.py \
    --data data/word_frequency.csv \
    --epochs 30 \
    --batch-size 64 \
    --rank-weight 5.0 \
    --output models/model_diagnostic.pt \
    --history training_history/diagnostic.json

# Or run all experiments
./scripts/train_local_experiments.sh
```

### RunPod Training (When Pod is Available)

```bash
# Start training
./scripts/train_runpod_diagnostic.sh

# Monitor (from another terminal)
ssh -i ~/.ssh/id_ed25519 -p 31179 root@38.80.152.76 \
    'tail -f /root/idf-est/training_history/runpod_diagnostic.log'
```

## What to Expect

### Good Signs ✅
- Prediction std > 0.01 (no collapse)
- Ranking loss decreasing over time
- Spearman improving (target: > 0.4)
- MAE decreasing (target: < 0.25)

### Red Flags ⚠️
- Collapse detected (pred_std < 0.01)
- Ranking loss = 0 (pairs not generated)
- Ranking loss not decreasing (weight too low)
- Spearman not improving (may need architecture changes)

## Initial Results (5 epochs)

- **No collapse**: Pred_std = 0.35-0.37 ✅
- **Spearman improving**: 0.105 → 0.164 ✅
- **Ranking loss working**: 14.97 → 11.61 ✅
- **MAE**: 0.27-0.29 (close to target < 0.25)

## Goals

### Phase 1: Fix Collapse ✅
- ✅ Non-zero predictions
- ✅ Meaningful range
- ✅ Spearman > 0.1

### Phase 2: Basic Learning (In Progress)
- ⚠️ MAE < 0.25 (currently 0.27-0.29)
- ⚠️ Spearman > 0.4 (currently 0.16, improving)
- ⏳ Jabberwocky 3/5+ tests

## Files

- `scripts/train_diagnostic.py`: Main diagnostic training script
- `scripts/train_local_experiments.sh`: Local experiment runner
- `scripts/train_runpod_diagnostic.sh`: RunPod training script
- `TRAINING_PLAN.md`: Detailed training plan
- `TRAINING_PROGRESS.md`: Current progress tracking

