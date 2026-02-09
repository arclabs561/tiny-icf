# Training Scripts Moved to trainctl

All training scripts and AWS management scripts have been moved to `../trainctl/training/`.

## New Location

- **Training Script**: `../trainctl/training/scripts/train_flexible_opportunistic.py`
- **AWS Scripts**: `../trainctl/training/scripts/scale_gpu_training.sh`, `monitor_aws_training.sh`, etc.

## Why?

The `trainctl` project is designed to be the unified training control system. All training orchestration should happen there, while `idf-est` remains focused on the model implementation.

## Usage

From `trainctl`:

```bash
cd /Users/arc/Documents/dev/trainctl
uv run training/scripts/train_flexible_opportunistic.py --data ../idf-est/data/word_frequency.csv
```

The scripts automatically reference `../idf-est/src` for the model source code.

