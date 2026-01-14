# idf-est Scripts

This directory contains utility scripts for the idf-est project.

## ⚠️ Training Scripts Moved

**All training scripts have been moved to `../trainctl/training/scripts/`**

Do NOT use scripts in this directory for training. Use trainctl instead.

## What's Here

- **Analysis scripts**: For analyzing model results (some moved to trainctl)
- **Data scripts**: For processing and preparing data
- **Utility scripts**: Various helper scripts

## Training

Use trainctl for all training:

```bash
cd /Users/arc/Documents/dev/trainctl
uv run training/scripts/train_flexible_opportunistic.py --data ../idf-est/data/word_frequency.csv
```

See `../trainctl/training/README.md` for details.

