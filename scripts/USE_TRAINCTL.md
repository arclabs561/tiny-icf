# ⚠️ Training Scripts Moved to trainctl

**All training scripts have been moved to `../trainctl/training/scripts/`**

## Do NOT use scripts in this directory for training

Use trainctl instead:

```bash
cd /Users/arc/Documents/dev/trainctl
uv run training/scripts/train_flexible_opportunistic.py --data ../idf-est/data/word_frequency.csv
```

## What Was Moved

- `train_flexible_opportunistic.py` → `../trainctl/training/scripts/`
- `scale_gpu_training.sh` → `../trainctl/training/scripts/`
- `monitor_aws_training.sh` → `../trainctl/training/scripts/`
- `show_training_results.sh` → `../trainctl/training/scripts/`
- `monitor_residual_experiments.sh` → `../trainctl/training/scripts/`

## Why?

- `trainctl` is the unified training control system
- `idf-est` focuses on model implementation
- All training orchestration happens in trainctl

See `../trainctl/training/README.md` for details.

