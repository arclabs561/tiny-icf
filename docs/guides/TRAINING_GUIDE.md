# Training Guide

Complete guide to training the tiny-icf model with different strategies.

## Quick Start

```bash
# 1. Prepare data
# Place word_frequency.csv in data/ directory

# 2. Train standard model
python -m tiny_icf.train \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 64 \
    --output models/model.pt

# 3. Evaluate
python scripts/evaluate_model.py \
    --model models/model.pt \
    --data data/word_frequency.csv
```

## Training Strategies

### 1. Standard Training (Baseline)

Uses `CombinedLoss` (Huber + Ranking):

```bash
python -m tiny_icf.train \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --output models/model_standard.pt
```

**When to use**: Initial training, baseline comparison

### 2. Multi-Loss Training (Enhanced)

Uses `EnhancedMultiLoss` with 5 components:
- Huber loss (absolute accuracy)
- Ranking loss (relative ordering)
- Contrastive loss (common/rare separation)
- Consistency loss (similar words → similar ICF)
- Calibration loss (frequency distribution matching)

```bash
python -m tiny_icf.train_multi_loss \
    --data data/word_frequency.csv \
    --epochs 100 \
    --batch-size 64 \
    --multi-loss \
    --output models/model_multi.pt
```

**When to use**: Better ranking, improved generalization

### 3. Curriculum Multi-Loss (Progressive)

Progressive loss addition:
- Stage 1 (0-33% epochs): Huber + Ranking
- Stage 2 (33-66% epochs): + Contrastive
- Stage 3 (66-100% epochs): + Consistency + Calibration

```bash
python -m tiny_icf.train_multi_loss \
    --data data/word_frequency.csv \
    --epochs 100 \
    --curriculum \
    --output models/model_curriculum.pt
```

**When to use**: More stable training, better convergence

## Training Variations

### Train Multiple Architectures

```bash
python scripts/train_variations.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --output-dir models/variations
```

This trains:
- `UniversalICF` (baseline, ~40k params)
- `HierarchicalICF` (smaller, ~16k params)
- `BoxEmbeddingICF` (smallest, ~14k params)

## Monitoring Training

### Real-time Progress

```bash
# One-time check
python scripts/monitor_training_progress.py --log training.log

# Watch mode (auto-update every 10 seconds)
python scripts/monitor_training_progress.py --log training.log --watch --interval 10
```

### Evaluation During Training

```bash
# Quick Jabberwocky test (fast)
python scripts/evaluate_model.py \
    --model models/model.pt \
    --jabberwocky-only

# Full evaluation (slower, more comprehensive)
python scripts/evaluate_model.py \
    --model models/model.pt \
    --data data/word_frequency.csv \
    --max-samples 1000
```

## Hyperparameters

### Learning Rate

- **Default**: `1e-3`
- **Too high**: Loss explodes, training unstable
- **Too low**: Slow convergence, may get stuck
- **Recommended**: Start with `1e-3`, adjust based on loss curve

### Batch Size

- **Default**: `64`
- **Larger**: More stable gradients, faster training, more memory
- **Smaller**: More gradient noise, slower training, less memory
- **Recommended**: 32-128 depending on available memory

### Epochs

- **Default**: `50-100`
- **Early stopping**: Monitor validation loss, stop if not improving
- **Recommended**: Train until validation loss plateaus

### Augmentation Probability

- **Default**: `0.1` (10% of samples augmented)
- **Higher**: More robust to typos, slower training
- **Lower**: Faster training, less robust
- **Recommended**: 0.1-0.2 for balanced performance

## Troubleshooting

### Model Predicts Similar Values for All Words

**Symptoms**: All predictions in narrow range (e.g., 0.5-0.6)

**Causes**:
- Not enough training epochs
- Learning rate too low
- Model capacity too small
- Data quality issues

**Solutions**:
- Train longer (100+ epochs)
- Increase learning rate (try `2e-3`)
- Use multi-loss training (better ranking)
- Check data quality

### Validation Loss Not Decreasing

**Symptoms**: Validation loss stuck or increasing

**Causes**:
- Overfitting (train loss decreasing, val loss increasing)
- Learning rate too high
- Model capacity too large for data size

**Solutions**:
- Add dropout (already in model)
- Reduce learning rate
- Use curriculum training
- Early stopping

### Jabberwocky Protocol Failing

**Symptoms**: Model fails on pseudo-words

**Causes**:
- Model memorizing frequencies, not learning structure
- Not enough training data
- Model too simple

**Solutions**:
- Train longer
- Use multi-loss training (contrastive + consistency)
- Add more diverse training data
- Try hierarchical architecture

## Best Practices

1. **Start with standard training** to establish baseline
2. **Monitor validation loss** - save best model
3. **Use multi-loss** if ranking is poor
4. **Use curriculum** if training is unstable
5. **Evaluate regularly** with Jabberwocky Protocol
6. **Compare models** using `compare_training.py`
7. **Train multiple variations** to find best architecture

## Expected Results

### Good Model Performance

- **MAE**: < 0.1 (mean absolute error)
- **Spearman correlation**: > 0.8 (ranking accuracy)
- **Jabberwocky Protocol**: 4/5 or 5/5 tests pass
- **Inference speed**: < 1ms per word (CPU)

### Training Time

- **Small dataset** (10k words): ~5-10 minutes (CPU)
- **Medium dataset** (100k words): ~30-60 minutes (CPU)
- **Large dataset** (1M+ words): Use GPU, ~2-4 hours

## Next Steps

After training:
1. Evaluate on test set
2. Test on Jabberwocky Protocol
3. Compare with other models
4. Optimize for deployment (quantization, pruning)

