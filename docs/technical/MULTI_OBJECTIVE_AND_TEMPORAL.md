# Multi-Objective Optimization & Historical Data Integration

## Overview

This document describes the integration of:
1. **Aligned Multi-Objective Optimization (AMOO)** for training with multiple objectives
2. **Historical n-gram data** for temporal ICF prediction
3. **Temporal loss functions** that leverage historical patterns

## Multi-Objective Optimization

### AMOO Framework

Based on the paper "Aligned Multi-Objective Optimization" (arXiv:2502.14096), we can optimize multiple aligned objectives simultaneously:

**Key Insight**: When objectives share a common solution, we can get better convergence by adaptively weighting them based on:
- **Curvature**: Functions with better local curvature get higher weight
- **Gradient alignment**: Objectives with aligned gradients reinforce each other

### Objectives for ICF Prediction

1. **ICF Prediction** (primary): Predict current ICF score
2. **Temporal Consistency**: Predictions should be consistent with historical trends
3. **Language Detection**: Model should learn language-specific patterns
4. **Era Classification**: Temporal patterns should align with era detection

### Implementation

```python
from tiny_icf.temporal_loss import AlignedMultiObjectiveLoss

# Define objectives
objectives = ['icf', 'temporal', 'language', 'era']

# Create AMOO loss
amoo_loss = AlignedMultiObjectiveLoss(
    objectives=objectives,
    adaptive=True,  # Use adaptive weighting
    curvature_weight=0.1,
)

# During training
losses = {
    'icf': icf_loss,
    'temporal': temporal_consistency_loss,
    'language': language_loss,
    'era': era_loss,
}

total_loss = amoo_loss(losses, gradients=gradients)
```

## Historical Data Integration

### Google Books N-gram Dataset

We can download historical word frequency data from Google Books n-gram dataset:
- **Coverage**: 1800-2019, decade-level granularity
- **Format**: Word frequencies by year
- **Size**: ~500MB per year for 1-grams

### Setup

```bash
# Run setup script
./scripts/setup_historical_data.sh

# Or manually
uv run scripts/download_historical_ngrams.py \
    --output-dir data/historical_ngrams \
    --years 1800 1900 2000 \
    --ngram-type 1gram
```

### Processing Historical Data

The script:
1. Downloads n-gram files from Google Books
2. Parses word frequencies by decade
3. Computes ICF scores for each decade
4. Saves to CSV: `data/historical_ngrams/historical_icf_1gram.csv`

### Using Historical Data in Training

```python
from tiny_icf.data_temporal import TemporalICFDataset

# Load dataset with historical data
dataset = TemporalICFDataset.from_files(
    current_data_path=Path('data/word_frequency.csv'),
    historical_data_path=Path('data/historical_ngrams/historical_icf_1gram.csv'),
    decades=[1800, 1900, 2000],
)

# Each sample includes:
# - 'icf': Current ICF score
# - 'icf_1800': Historical ICF for 1800s
# - 'icf_1900': Historical ICF for 1900s
# - 'icf_2000': Historical ICF for 2000s
```

## Temporal Loss Functions

### 1. Temporal ICF Loss

Encourages predictions to match historical ICF patterns:

```python
from tiny_icf.temporal_loss import temporal_icf_loss

loss = temporal_icf_loss(
    predictions=model_output,
    targets=current_icf,
    temporal_targets={
        '1800': historical_icf_1800,
        '1900': historical_icf_1900,
        '2000': historical_icf_2000,
    },
    alpha=0.1,  # Weight for temporal consistency
)
```

### 2. Multi-Decade ICF Loss

Predict ICF across multiple decades simultaneously:

```python
from tiny_icf.temporal_loss import multi_decade_icf_loss

# Model outputs predictions for each decade
predictions = {
    '1800': model_1800_output,
    '1900': model_1900_output,
    '2000': model_2000_output,
}

targets = {
    '1800': historical_icf_1800,
    '1900': historical_icf_1900,
    '2000': historical_icf_2000,
}

loss = multi_decade_icf_loss(predictions, targets)
```

### 3. Temporal Consistency Loss

Encourages smooth transitions across decades:

```python
from tiny_icf.temporal_loss import compute_temporal_consistency_loss

loss = compute_temporal_consistency_loss(
    predictions=current_predictions,
    historical_predictions={
        '1800': hist_pred_1800,
        '1900': hist_pred_1900,
        '2000': hist_pred_2000,
    },
    smoothness_weight=0.1,
)
```

## Training with Multi-Objective & Temporal Data

### Example Training Loop

```python
from tiny_icf.temporal_loss import AlignedMultiObjectiveLoss
from tiny_icf.data_temporal import TemporalICFDataset

# Load dataset
dataset = TemporalICFDataset.from_files(
    current_data_path=Path('data/word_frequency.csv'),
    historical_data_path=Path('data/historical_ngrams/historical_icf_1gram.csv'),
    decades=[1800, 1900, 2000],
)

# Create AMOO loss
amoo_loss = AlignedMultiObjectiveLoss(
    objectives=['icf', 'temporal'],
    adaptive=True,
)

# Training loop
for batch in dataloader:
    words = batch['bytes']
    current_icf = batch['icf']
    
    # Model prediction
    prediction = model(words)
    
    # Compute losses
    icf_loss = mse_loss(prediction, current_icf)
    
    # Temporal loss (if historical data available)
    temporal_loss = 0.0
    if 'icf_1800' in batch:
        # Encourage consistency with historical trends
        temporal_loss = temporal_icf_loss(
            prediction,
            current_icf,
            temporal_targets={
                '1800': batch['icf_1800'],
                '1900': batch['icf_1900'],
                '2000': batch['icf_2000'],
            },
        )
    
    # Combined loss with adaptive weighting
    total_loss = amoo_loss(
        losses={'icf': icf_loss, 'temporal': temporal_loss},
        gradients=gradients,
    )
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

## Benefits

### 1. Better Generalization
- Historical patterns help model learn temporal trends
- Multi-objective training improves robustness

### 2. Improved Convergence
- AMOO adaptively weights objectives based on curvature
- Faster convergence when objectives are aligned

### 3. Richer Predictions
- Model can predict ICF for different time periods
- Temporal consistency improves accuracy

## Next Steps

1. **Download Historical Data**: Run setup script to download n-gram data
2. **Integrate into Training**: Modify training scripts to use temporal data
3. **Experiment with AMOO**: Test adaptive weighting vs fixed weights
4. **Evaluate Temporal Accuracy**: Measure how well model predicts historical ICF

## Files

- `scripts/download_historical_ngrams.py` - Download and process n-gram data
- `scripts/setup_historical_data.sh` - Setup script
- `src/tiny_icf/temporal_loss.py` - Temporal loss functions
- `src/tiny_icf/data_temporal.py` - Temporal dataset
- `MULTI_OBJECTIVE_AND_TEMPORAL.md` - This file

