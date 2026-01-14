# Aim Experiment Tracking Quick Start

Aim is integrated into the ICF training scripts for experiment tracking, visualization, and comparison.

## Installation

Aim is included in the project dependencies. Install it with:

```bash
uv sync
# or
pip install aim
```

## Basic Usage

### Standard Training Script

Enable Aim tracking with the `--aim` flag:

```bash
# Basic training with Aim
uv run python -m tiny_icf.train \
    --data data/word_frequency.csv \
    --epochs 100 \
    --aim \
    --aim-experiment "baseline-experiments"

# Best practices training with Aim
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --aim \
    --aim-experiment "optimization-sweep"
```

### PyTorch Lightning Training

```bash
uv run python -m tiny_icf.train_lightning \
    --data data/word_frequency.csv \
    --output-dir models/lightning \
    --epochs 100 \
    --aim \
    --aim-experiment "lightning-training"
```

## Starting the Aim UI

After training, start the Aim UI to view your experiments:

```bash
aim up
```

This starts a web server (default: http://127.0.0.1:43800) where you can:
- Compare runs side-by-side
- Visualize metrics over time
- Filter and search experiments
- Analyze hyperparameter effects

## What Gets Tracked

### Hyperparameters
- Training configuration (epochs, batch size, learning rate, etc.)
- Model architecture parameters
- Data configuration (max length, augmentation probability)
- Device and seed information

### Metrics
- **Training**: `train_loss`, `train_huber_loss`, `train_ranking_loss`
- **Validation**: `val_loss`, `val_mae`, `val_rmse`, `val_spearman_corr`, `val_pearson_corr`
- **Jabberwocky Protocol**: Pass rate and individual word predictions
- **Prediction Statistics**: Mean, std, min, max of predictions
- **Learning Rate**: Current learning rate (when using schedulers)

### Best Model Tracking
- Best validation loss
- Best Spearman correlation
- Best epoch number

## Example Workflow

1. **Run multiple experiments** with different hyperparameters:

```bash
# Experiment 1: Baseline
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --lr 1e-3 \
    --aim --aim-experiment "hyperparameter-sweep"

# Experiment 2: Higher learning rate
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --lr 2e-3 \
    --aim --aim-experiment "hyperparameter-sweep"

# Experiment 3: Different scheduler
uv run scripts/train_best_practices.py \
    --data data/word_frequency.csv \
    --epochs 100 \
    --scheduler cosine \
    --aim --aim-experiment "hyperparameter-sweep"
```

2. **View results** in Aim UI:

```bash
aim up
```

3. **Compare experiments**:
   - Use the UI to filter by experiment name
   - Compare metrics across runs
   - Identify best hyperparameter combinations

## Programmatic Access

You can also query runs programmatically using the Aim SDK:

```python
from aim import Run

# Get all runs from an experiment
runs = Run.filter(experiment="hyperparameter-sweep")

for run in runs:
    print(f"Run: {run.name}")
    print(f"  Best Spearman: {run['best_spearman']}")
    print(f"  Learning Rate: {run['hparams']['learning_rate']}")
    print(f"  Batch Size: {run['hparams']['batch_size']}")
```

## Tips

- **Organize experiments**: Use descriptive experiment names to group related runs
- **Track everything**: All hyperparameters are automatically logged
- **Compare systematically**: Run multiple experiments with the same experiment name to compare them easily
- **Use tags**: You can add tags to runs in the UI for better organization

## Troubleshooting

If Aim tracking fails, the training will continue without tracking (warnings will be printed). Common issues:

- **Aim not installed**: Install with `uv sync` or `pip install aim`
- **Permission errors**: Check write permissions for the Aim repository (default: `~/.aim`)
- **Port conflicts**: If `aim up` fails, use `aim up --port 43801` to use a different port

## Further Reading

- [Aim Documentation](https://aimstack.readthedocs.io/)
- [Aim GitHub](https://github.com/aimhubio/aim)

