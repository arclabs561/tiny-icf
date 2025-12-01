# tiny-icf: Tiny ICF Model

A highly compressed, character-level model that predicts normalized ICF (Inverse Collection Frequency) for arbitrary words. The model learns the statistical structure of word commonality from character-level features, satisfying Kolmogorov complexity constraints.

## Core Concept

Instead of storing a massive frequency dictionary, this model learns to predict word commonality from:
- Byte-level character sequences
- Morphological patterns (prefixes, suffixes, roots)
- Structural validity (English phonotactics)

**Target**: Normalized ICF score where `1.0 = rare/unique` and `0.0 = stopword`

## Architecture

- **Input**: UTF-8 bytes (0-255)
- **Encoder**: Parallel 1D CNNs (kernel sizes 3, 5, 7) for morphological n-grams
- **Parameters**: < 50k (fits in L2 cache)
- **Output**: Sigmoid-normalized ICF (0.0 to 1.0)

## Usage

```bash
# Train on frequency list
python -m tiny_icf.train --data path/to/frequencies.csv --epochs 50

# Predict ICF for words
python -m tiny_icf.predict --words "the apple xylophone qzxbjk"

# Run Jabberwocky Protocol validation
pytest tests/test_jabberwocky.py
```

## Validation: Jabberwocky Protocol

The model must correctly predict scores for non-existent words:
- `"the"` → ~0.0 (common)
- `"xylophone"` → ~0.9 (rare but valid)
- `"flimjam"` → ~0.6-0.8 (rare, looks English)
- `"qzxbjk"` → ~0.99 (impossible structure)
- `"unfriendliness"` → ~0.4-0.6 (composed of common parts)

