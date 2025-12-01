# Data and Models

This repository does not include training data or model files to keep the repo size manageable.

## Getting Data

### Option 1: Download Script
```bash
./scripts/download_data.sh
```

### Option 2: Python Script
```bash
python scripts/download_datasets.py
```

### Option 3: Manual Download
Place your training data in the `data/` directory:
- `data/word_frequency.csv` - Main training data (word, frequency columns)
- `data/word_frequency_modern.csv` - With modern words added
- Other data files as needed

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

## Why Excluded?

- **Data files**: Large, change frequently, user-specific
- **Model files**: Large, generated from data, user-specific
- **Log files**: Temporary, generated during training
- **Temporary docs**: Analysis/status files, not needed for code

This keeps the repository focused on code and essential documentation.

