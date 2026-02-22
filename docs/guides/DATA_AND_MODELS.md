# Data and Models

This repository does not include training data or model files to keep the repo size manageable.

## Getting Data

### Option 1: Download Script
```bash
./scripts/download_data.sh
```

### Option 2: Python Script
```bash
uv run python scripts/download_best_data.py
```

### Option 3: Manual Download
Place your training data in the `data/` directory:
- `data/word_frequency.csv` - Main training data (word, frequency columns)
- `data/word_frequency_modern.csv` - With modern words added
- Other data files as needed

For multilingual training (language-prefixed keys), this repo can also use:
- `data/word_frequency_multilingual.csv` - Keys like `en:word`, `es:palabra`, etc

## Getting Models

### Option 1: Train Your Own
```bash
# Basic training
python -m tiny_icf.train --data data/word_frequency.csv --epochs 50 --output models/model.pt

# Curriculum training (recommended)
python -m tiny_icf.train_curriculum \
  --data data/word_frequency.csv \
  --epochs 50 \
  --output models/model.pt
```

### Option 2: Download Script
```bash
./scripts/download_models.sh
```

(Note: Pre-trained models may be available via releases or external hosting)

## Directory Structure

```
tiny-icf/
├── data/              # Training data (gitignored, use download scripts)
├── models/            # Trained models (gitignored, train locally)
├── src/tiny_icf/      # Source code (committed)
├── tests/             # Tests (committed)
└── scripts/           # Scripts including download helpers (committed)
```

## Model selection (pre-trained)

| Model | Jabberwocky | MAE | Spearman | Use when |
|-------|-------------|-----|----------|----------|
| `multitask_all_fronts_v3.pt` | 46% | 0.26 | 0.14 | OOV calibration, pseudo-words |
| `multitask_all_fronts_v3b.pt` | 31% | 0.09 | 0.18–0.27 | Dataset fit, ranking quality (best ckpt, e.g. ep24) |
| `multitask_all_fronts_v4.pt` | 31% | 0.28 | 0.07 | Better "the"/common-word calibration (124K params) |

Download from S3: `aws s3 cp s3://arclabs-backups/tiny-icf/models/<name>.pt models/`

Sync to S3 (as needed): `aws s3 sync models/ s3://arclabs-backups/tiny-icf/models/ --exclude "*" --include "multitask_all_fronts*.pt" --include "v3_base*.pt"`. When a run finishes, the train script exports from best checkpoint; then run this sync.

## Model Storage: Repo vs Local vs S3

| Location | Status |
|----------|--------|
| **Repo** | No. `models/`, `*.pt`, `*.ckpt` are gitignored. |
| **Local** | Yes. Training writes to `models/` (or `--output-dir`). |
| **S3** | Optional. Use `scripts/upload_model_to_s3.sh` after training. |

To publish a model for others: train locally, then upload to your S3 bucket (or GitHub Releases). The upload script requires `aws` CLI and `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or equivalent).

## Why Excluded?

- **Data files**: Large, change frequently, user-specific
- **Model files**: Large, generated from data, user-specific
- **Log files**: Temporary, generated during training
- **Temporary docs**: Analysis/status files, not needed for code

This keeps the repository focused on code and essential documentation.

