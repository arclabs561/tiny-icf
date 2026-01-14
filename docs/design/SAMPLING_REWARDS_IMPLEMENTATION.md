# Sampling-Based Rewards with Smooth Weighting

## Overview

Implemented enhanced ranking loss with sampling-based rewards and smooth weighting, as suggested by the user. This improves the model's ability to learn from meaningful word pairs while providing smoother gradients.

## Key Improvements

### 1. Weighted Sampling (`generate_ranking_pairs`)

**Before**: Random uniform sampling of pairs, potentially including pairs with very similar ICF scores (noise).

**After**: 
- Builds all valid pairs with their ICF differences
- Samples pairs with probability proportional to their ICF difference
- Uses softmax-based probability distribution: `probs = softmax(diffs * 5.0)`
- Larger differences = higher sampling probability = stronger learning signal

**Benefits**:
- Focuses learning on pairs with meaningful differences
- Reduces noise from similar-frequency words
- Provides stronger gradient signals

### 2. Smooth Ranking Loss

**Before**: Hard ReLU-based loss: `loss = max(0, margin - diff)`

**After**: Smooth sigmoid-based loss: `loss = sigmoid((margin - diff) * temperature)`

**Benefits**:
- Smooth gradients even when predictions are close to the margin
- More stable training dynamics
- Configurable temperature (default: 10.0) controls sharpness

### 3. Weighted Loss by Target Difference

**New Feature**: Loss is weighted by the actual ICF difference between pairs.

**Implementation**:
- Uses softmax weighting: `weights = softmax(target_diffs * 5.0)`
- Pairs with larger ICF differences contribute more to the loss
- Provides adaptive importance weighting

**Benefits**:
- Emphasizes learning from pairs with clear frequency distinctions
- Reduces impact of noisy pairs with small differences
- Better alignment with ranking objectives

## Code Changes

### `src/tiny_icf/loss.py`

- `ranking_loss()`: Added `target_diff`, `smooth`, and `temperature` parameters
- `CombinedLoss.forward()`: Added `pair_target_diffs` and `smooth_ranking` parameters

### `src/tiny_icf/train.py`

- `generate_ranking_pairs()`: Complete rewrite with weighted sampling
  - Returns `(pairs, diffs)` tuple instead of just `pairs`
  - Builds all valid pairs first, then samples according to weights
  - Falls back to uniform sampling if no valid pairs found

### `src/tiny_icf/train_multi_loss.py`

- Updated to import `generate_ranking_pairs` from `train.py`
- Updated to pass `pair_target_diffs` to `EnhancedMultiLoss`

### `src/tiny_icf/loss_multi.py`

- `EnhancedMultiLoss.forward()`: Added `pair_target_diffs` parameter
- Updated `ranking_loss` call to use smooth rewards and target diffs

## Usage

### Standard Training

```python
from tiny_icf.train import generate_ranking_pairs
from tiny_icf.loss import CombinedLoss

# Generate pairs with weighted sampling
pairs, diffs = generate_ranking_pairs(
    targets, n_pairs=32, min_diff=0.05, use_weighted_sampling=True
)

# Use in loss
criterion = CombinedLoss()
loss = criterion(
    predictions, targets,
    pairs=pairs,
    pair_target_diffs=diffs,
    smooth_ranking=True,
)
```

### Multi-Loss Training

```python
from tiny_icf.loss_multi import EnhancedMultiLoss

criterion = EnhancedMultiLoss()
loss = criterion(
    predictions, targets,
    pairs=pairs,
    pair_target_diffs=diffs,
    # ... other parameters
)
```

## Testing

All functionality has been tested and verified:
- ✅ Weighted sampling generates pairs with larger differences
- ✅ Smooth ranking loss provides smooth gradients
- ✅ Weighted loss emphasizes important pairs
- ✅ Integration with CombinedLoss and EnhancedMultiLoss

## Expected Impact

1. **Better Spearman Correlation**: Weighted sampling and smooth rewards should improve ranking performance
2. **Faster Convergence**: Stronger signals from meaningful pairs accelerate learning
3. **More Stable Training**: Smooth gradients reduce training instability
4. **Better Generalization**: Focus on meaningful differences improves model robustness

## Next Steps

- Run training experiments to measure impact on Spearman correlation
- Tune temperature and weighting scale factors if needed
- Consider adaptive temperature scheduling during training
- Monitor training dynamics to ensure smooth rewards are helping

