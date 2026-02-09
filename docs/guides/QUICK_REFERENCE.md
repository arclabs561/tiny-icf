# Quick Reference Guide

Essential commands and workflows for tiny-icf.

## Training

```bash
# Standard training (with weighted sampling & smooth rewards)
python -m tiny_icf.train --data data/word_frequency.csv --epochs 100 --output models/model.pt

# Adaptive training with early stopping (recommended)
python scripts/train_adaptive.py --data data/word_frequency.csv --epochs 100 --scheduler adaptive --early-stop --output models/model_adaptive.pt

# Multi-loss training
python -m tiny_icf.train_multi_loss --data data/word_frequency.csv --epochs 100 --multi-loss --output models/model_multi.pt

# Training with mid-epoch evaluation
python -m tiny_icf.train_with_eval --data data/word_frequency.csv --epochs 100 --eval-interval 5 --use-scheduler --output models/model_with_eval.pt

# Quick test improvements (5 epochs, 5k words)
python scripts/quick_test_improvements.py

# Compare loss configurations
python scripts/compare_loss_configs.py

# Run batch experiments
python scripts/run_batch_experiments.py --data data/word_frequency.csv --quick
```

## Prediction

```bash
# Quick interactive prediction
python scripts/quick_predict.py --model models/model.pt

# Batch prediction
python scripts/quick_predict.py --model models/model.pt --words "the apple xylophone"

# From file
python scripts/quick_predict.py --model models/model.pt --file words.txt --output results.csv

# Using predict module
python -m tiny_icf.predict --model models/model.pt --words "the apple xylophone"
```

## Evaluation

```bash
# Comprehensive evaluation with error analysis
python scripts/comprehensive_eval.py --model models/model.pt --data data/word_frequency.csv --output eval_results.json

# Standard evaluation
python scripts/evaluate_model.py --model models/model.pt --data data/word_frequency.csv

# Quick Jabberwocky test
python scripts/evaluate_model.py --model models/model.pt --jabberwocky-only

# Compare multiple models
python scripts/compare_models.py --models baseline:models/model1.pt improved:models/model2.pt --data data/word_frequency.csv

# Compare two models
python scripts/compare_training.py --model1 models/model1.pt --model2 models/model2.pt --data data/word_frequency.csv

# Analyze training dynamics
python scripts/analyze_training_dynamics.py --model models/model.pt --data data/word_frequency.csv --n-batches 10
```

## Monitoring & Analysis

```bash
# Training dashboard (real-time monitoring)
python scripts/training_dashboard.py --watch --log training.log --interval 5

# Plot training history
python scripts/training_dashboard.py --history training_history.json --plot training_plot.png

# Monitor training progress
python scripts/monitor_training_progress.py --log training.log --watch --interval 10

# Analyze training dynamics (gradients, loss components)
python scripts/analyze_training_dynamics.py --model models/model.pt --data data/word_frequency.csv --output dynamics.json
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_jabberwocky.py

# With coverage
pytest tests/ --cov=src/tiny_icf
```

## Data & Models

```bash
# Download data
./scripts/download_data.sh

# Download models (if available)
./scripts/download_models.sh
```

## Benchmarking & Visualization

```bash
# Benchmark training configurations
python scripts/benchmark_training.py --data data/word_frequency.csv --epochs 5 --output benchmark.json

# Visualize training history
python scripts/visualize_training.py --history training_history.json --output training_plot.png

# Plot predictions vs targets
python scripts/visualize_training.py --model models/model.pt --data data/word_frequency.csv --type predictions

# Export model for deployment
python scripts/export_model.py --model models/model.pt --format all --output-dir export/
```

## Common Workflows

### Complete Training Pipeline

```bash
# 1. Prepare data
./scripts/download_data.sh

# 2. Train model
python -m tiny_icf.train_multi_loss \
    --data data/word_frequency.csv \
    --epochs 100 \
    --multi-loss \
    --output models/model.pt

# 3. Evaluate
python scripts/evaluate_model.py \
    --model models/model.pt \
    --data data/word_frequency.csv

# 4. Test predictions
python scripts/quick_predict.py --model models/model.pt
```

### Quick Model Comparison

```bash
# Train two variants
python -m tiny_icf.train --data data/word_frequency.csv --epochs 50 --output models/standard.pt
python -m tiny_icf.train_multi_loss --data data/word_frequency.csv --epochs 50 --multi-loss --output models/multi.pt

# Compare
python scripts/compare_training.py \
    --model1 models/standard.pt \
    --model2 models/multi.pt \
    --data data/word_frequency.csv
```

## Model Performance Targets

- **MAE**: < 0.1 (mean absolute error)
- **Spearman**: > 0.8 (ranking correlation)
- **Jabberwocky**: 4/5 or 5/5 tests pass
- **Inference**: < 1ms per word (CPU)

## Troubleshooting

### Model predicts similar values for all words
- Train longer (100+ epochs)
- Use multi-loss training
- Check data quality

### Training loss not decreasing
- Reduce learning rate
- Use curriculum training
- Check data preprocessing

### Jabberwocky Protocol failing
- Train longer
- Use multi-loss (contrastive + consistency)
- Add more diverse training data

## File Locations

- **Models**: `models/`
- **Data**: `data/`
- **Logs**: `training*.log`
- **Source**: `src/tiny_icf/`
- **Tests**: `tests/`
- **Scripts**: `scripts/`

## Documentation

- **`README.md`**: Overview, install, CLI, smoke-test
- **`docs/guides/QUICK_START.md`**: Quick start workflow
- **`docs/guides/TRAINING_GUIDE.md`**: Detailed training guide
- **Jabberwocky**: `scripts/evaluate_model.py --jabberwocky-only` (or `tests/test_jabberwocky.py`)
- **`docs/guides/QUICK_REFERENCE.md`**: This file

