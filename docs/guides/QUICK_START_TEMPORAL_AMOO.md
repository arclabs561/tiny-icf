# Quick Start: Temporal AMOO Training

## Overview

This guide shows how to train the ICF model with:
- **Historical n-gram data** (1800-2019)
- **Aligned Multi-Objective Optimization (AMOO)** for adaptive weighting
- **Temporal loss functions** for historical consistency

## Step 1: Test Setup

Verify everything works:

```bash
uv run scripts/test_temporal_amoo.py
```

Expected output:
```
✓ All tests passed!
```

## Step 2: Download Historical Data (Optional)

If you want to use historical data:

```bash
./scripts/setup_historical_data.sh
```

Or manually:

```bash
uv run scripts/download_historical_ngrams.py \
    --output-dir data/historical_ngrams \
    --years 1800 1900 2000 \
    --ngram-type 1gram
```

**Note**: Google Books n-gram files are large (~500MB each). The script downloads sample files (letters 'a' and 't') for testing. For full dataset, you'll need to download all letters (a-z).

## Step 3: Train with Temporal AMOO

### Basic Training (No Historical Data)

```bash
uv run scripts/train_temporal_amoo.py \
    --data data/word_frequency.csv \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3
```

### With Historical Data

```bash
uv run scripts/train_temporal_amoo.py \
    --data data/word_frequency.csv \
    --historical-data data/historical_ngrams/historical_icf_1gram.csv \
    --use-temporal \
    --adaptive-weights \
    --temporal-alpha 0.1 \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3
```

### Parameters

- `--use-temporal`: Enable temporal loss (requires `--historical-data`)
- `--adaptive-weights`: Use AMOO adaptive weighting
- `--temporal-alpha`: Weight for temporal consistency (default: 0.1)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate

## Step 4: Monitor Training

Training output shows:
- **Train loss**: Combined AMOO loss
- **ICF loss**: Base ICF prediction loss
- **Temporal loss**: Historical consistency loss (if enabled)
- **Spearman**: Rank correlation
- **MAE**: Mean absolute error

Example output:
```
Epoch 1/20
  Train - Loss: 0.1234, ICF: 0.1000, Temporal: 0.0234, Spearman: 0.2500
  Val   - Loss: 0.1100, Spearman: 0.2800, MAE: 0.3100
```

## Step 5: Use Trained Model

The trained model is saved to `models/model_temporal_amoo.pt`.

Use it with existing prediction scripts:

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_temporal_amoo.pt \
    --words "thou computer selfie" \
    --data data/word_frequency.csv \
    --json
```

## Understanding AMOO

### What is AMOO?

**Aligned Multi-Objective Optimization** exploits when multiple objectives share a common solution. Instead of fixed weights, AMOO adaptively weights objectives based on:

1. **Gradient alignment**: Objectives with aligned gradients reinforce each other
2. **Curvature**: Objectives with better local curvature get higher weight
3. **Convergence**: Faster convergence when objectives are aligned

### Our Objectives

1. **ICF Prediction**: Primary objective - predict current ICF score
2. **Temporal Consistency**: Secondary - predictions should match historical trends

When these are aligned (e.g., a word's ICF is consistent across decades), AMOO gives better convergence than fixed weighting.

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Install dependencies:
```bash
uv pip install torch numpy pandas scipy tqdm
```

### "Historical data file not found"

Either:
1. Download historical data (Step 2)
2. Train without `--use-temporal` flag

### "AMOO loss is NaN"

This can happen if:
- Losses are too large → reduce learning rate
- Gradients explode → gradient clipping is enabled by default
- Temporal data has NaN values → check historical data file

### Low Spearman Correlation

Try:
- Increase `--temporal-alpha` (more weight on temporal consistency)
- Use `--adaptive-weights` (let AMOO find optimal weights)
- Train for more epochs
- Check data quality

## Advanced Usage

### Custom Objective Weights

Modify `AlignedMultiObjectiveLoss` initialization in `train_temporal_amoo.py`:

```python
amoo_loss = AlignedMultiObjectiveLoss(
    objectives=['icf', 'temporal'],
    initial_weights={'icf': 0.8, 'temporal': 0.2},  # Custom weights
    adaptive=True,
)
```

### Multiple Decades

Extend to more decades:

```python
decades = [1800, 1850, 1900, 1950, 2000]
```

Update `TemporalICFDataset` and temporal loss functions accordingly.

### Language-Specific Temporal Data

Download n-gram data for other languages:

```bash
# Spanish
uv run scripts/download_historical_ngrams.py \
    --ngram-type 1gram \
    --language spa
```

## Performance Tips

1. **Start small**: Test with sample data before full download
2. **Use GPU**: Set `--device cuda` if available
3. **Batch size**: Larger batches (256+) for faster training
4. **Early stopping**: Monitor validation Spearman, stop if not improving

## Next Steps

1. **Evaluate temporal accuracy**: Measure how well model predicts historical ICF
2. **Compare with baseline**: Train without temporal data, compare results
3. **Experiment with weights**: Try different `temporal_alpha` values
4. **Extend to more objectives**: Add language detection, era classification

## Files Reference

- `scripts/train_temporal_amoo.py` - Main training script
- `scripts/download_historical_ngrams.py` - Download historical data
- `scripts/test_temporal_amoo.py` - Test setup
- `src/tiny_icf/temporal_loss.py` - AMOO and temporal losses
- `src/tiny_icf/data_temporal.py` - Temporal dataset
- `MULTI_OBJECTIVE_AND_TEMPORAL.md` - Detailed documentation

