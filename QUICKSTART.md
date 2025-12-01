# Quick Start Guide

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Step 1: Prepare Frequency Data

You need a CSV file with word frequencies. See `DATA_PREP.md` for sources.

Format: `word,count` (one per line)

Example `frequencies.csv`:
```csv
the,1000000
be,800000
to,700000
of,600000
and,500000
```

## Step 2: Train the Model

```bash
python -m tiny_icf.train \
    --data frequencies.csv \
    --epochs 50 \
    --batch-size 64 \
    --output model.pt
```

Training will:
- Load frequency list
- Compute normalized ICF scores
- Create stratified sample (handles Zipfian distribution)
- Train byte-level CNN model
- Save best model based on validation loss

## Step 3: Test Predictions

```bash
python -m tiny_icf.predict \
    --model model.pt \
    --words "the apple xylophone qzxbjk unfriendliness"
```

Expected output:
```
Word                 ICF Score  Interpretation
------------------------------------------------------------
the                  0.0234     Very Common (stopword-like)
apple                0.3456     Common
xylophone            0.8765     Very Rare/Unique
qzxbjk               0.9876     Very Rare/Unique
unfriendliness       0.5432     Rare
```

## Step 4: Validate Generalization

Run the Jabberwocky Protocol to test if the model learned word structure:

```bash
python tests/test_jabberwocky.py model.pt
```

This tests the model on non-existent words to verify it learned morphological patterns rather than just memorizing frequencies.

## Model Architecture Summary

- **Input**: UTF-8 bytes (0-255)
- **Encoder**: 3 parallel 1D CNNs (kernel sizes 3, 5, 7)
- **Parameters**: < 50k (fits in L2 cache)
- **Output**: Normalized ICF (0.0 = common, 1.0 = rare)

## Expected Training Time

- **Small dataset** (100k words): ~5-10 minutes on CPU
- **Medium dataset** (1M words): ~30-60 minutes on CPU
- **Large dataset** (10M+ words): Use GPU, ~2-4 hours

## Troubleshooting

**Issue**: Model predicts all words as ~0.5
- **Solution**: Increase training epochs or check data quality

**Issue**: Jabberwocky Protocol fails
- **Solution**: Train on larger, more diverse frequency list

**Issue**: Out of memory during training
- **Solution**: Reduce batch size or max_length

