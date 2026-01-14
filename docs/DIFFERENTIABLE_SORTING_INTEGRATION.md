# Differentiable Sorting Integration

## Overview

Integrated two state-of-the-art differentiable sorting libraries to directly optimize Spearman correlation:

1. **diffsort** ([GitHub](https://github.com/Felix-Petersen/diffsort)) - Differentiable Sorting Networks (ICLR 2022, ICML 2021)
2. **fast-soft-sort** ([GitHub](https://github.com/google-research/fast-soft-sort)) - Fast Differentiable Sorting and Ranking (ICML 2020)

## Why Differentiable Sorting?

**Problem**: Spearman correlation requires sorting, which is non-differentiable. Current approaches:
- Pairwise ranking loss: Optimizes local pairs, not global ranking
- Listwise losses (LambdaRank, ApproxNDCG): Approximate ranking, still indirect

**Solution**: Differentiable sorting allows gradients to flow through the sorting operation, enabling **direct optimization of Spearman correlation**.

## Implementation

### Core Loss Function (`src/tiny_icf/loss_diffsort.py`)

**`DifferentiableSortingLoss`**:
- Uses differentiable sorting to get soft ranks
- Computes Spearman correlation on ranks (Pearson correlation of ranks)
- Loss = 1 - Spearman (so lower is better, Spearman higher is better)
- Combines with Huber loss for absolute accuracy

**Key Features**:
- Supports both `diffsort` and `fast-soft-sort` backends
- Automatic backend selection if both available
- Configurable regularization/steepness parameters
- Combines ranking (Spearman) with absolute accuracy (Huber)

### Integration Points

1. **Training Script** (`scripts/train_diffsort.py`):
   - Full training pipeline with differentiable sorting loss
   - Automatic library detection
   - Early stopping and checkpointing
   - RBO metrics integration

2. **Ablation Study** (`scripts/ablation_loss_study.py`):
   - Now includes differentiable sorting configurations
   - Compares all loss methods systematically

## Installation

### diffsort
```bash
pip install diffsort
# or
uv pip install diffsort
```

### fast-soft-sort
```bash
git clone https://github.com/google-research/fast-soft-sort.git
# Copy fast_soft_sort/ to src/tiny_icf/
```

## Usage

### Training with Differentiable Sorting

```bash
# Using diffsort (if installed)
uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --method diffsort \
    --huber-weight 0.3 \
    --output models/model_diffsort.pt

# Using fast-soft-sort (if installed)
uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --method fast_soft_sort \
    --regularization-strength 1.0 \
    --output models/model_fast_soft_sort.pt

# Auto-detect (uses first available)
uv run scripts/train_diffsort.py \
    --data data/word_frequency.csv \
    --epochs 50 \
    --method auto \
    --output models/model_auto_sort.pt
```

### In Code

```python
from tiny_icf.loss_diffsort import DifferentiableSortingLoss, check_differentiable_sorting_available

# Check availability
available = check_differentiable_sorting_available()
print(available)  # {'diffsort': True, 'fast_soft_sort': False}

# Create loss
loss_fn = DifferentiableSortingLoss(
    method="diffsort",  # or "fast_soft_sort"
    steepness=5.0,  # For diffsort (higher = sharper)
    regularization_strength=1.0,  # For fast-soft-sort
    huber_weight=0.3,  # 30% Huber, 70% Spearman
)

# Use in training
loss = loss_fn(predictions, targets)
loss.backward()
```

## Expected Benefits

1. **Direct Spearman Optimization**: Gradients flow through sorting, directly optimizing the metric we care about
2. **Better Ranking Signal**: No approximation needed - true differentiable ranking
3. **Improved Convergence**: More direct optimization path should converge faster
4. **Higher Spearman Correlation**: Should outperform pairwise/listwise losses

## Comparison with Other Methods

| Method | Spearman Optimization | Gradient Flow | Complexity |
|--------|----------------------|---------------|------------|
| Pairwise Ranking | Indirect (local pairs) | Through pairs | Low |
| Listwise (LambdaRank) | Indirect (NDCG proxy) | Through NDCG | Medium |
| Listwise (ApproxNDCG) | Indirect (soft ranking) | Through softmax | Medium |
| **Differentiable Sorting** | **Direct** | **Through sorting** | **Medium-High** |

## Research References

1. **diffsort**: Petersen et al. "Monotonic Differentiable Sorting Networks" (ICLR 2022)
   - Uses classic sorting networks (bitonic, odd-even) with relaxation
   - Leverages permutation matrices for differentiable sorting

2. **fast-soft-sort**: Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020)
   - Uses regularization-based approach
   - Faster than diffsort for large batches
   - More flexible regularization strength

## Next Steps

1. **Run Ablation Study**: Compare all loss methods including differentiable sorting
2. **Hyperparameter Tuning**: Optimize `steepness` (diffsort) and `regularization_strength` (fast-soft-sort)
3. **Batch Size Considerations**: Differentiable sorting may have different optimal batch sizes
4. **Performance Analysis**: Measure training speed and memory usage

## Notes

- Differentiable sorting is more computationally expensive than pairwise losses
- May require larger batches for stable gradients
- Regularization/steepness parameters need tuning per dataset
- Both libraries are research code - may have edge cases


