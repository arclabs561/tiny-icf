# Adaptive Loss Implementation Summary

## Overview

Based on comprehensive research on multi-objective loss functions, we've implemented enhanced loss weighting strategies and monitoring capabilities.

## What Was Implemented

### 1. Enhanced CombinedLoss ✅

**Location**: `src/tiny_icf/loss.py`

**New Features**:
- **Component Tracking**: Tracks individual loss components (Huber, Ranking, NeuralNDCG, Listwise)
- **Component Statistics**: `get_component_stats()` method returns mean, std, and ratios
- **Imbalance Detection**: Automatically detects when one loss dominates (>70%)

**Usage**:
```python
loss_fn = CombinedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
    track_components=True,  # Enable tracking
)

# During training
loss = loss_fn(predictions, targets, pairs=pairs)

# Get statistics
stats = loss_fn.get_component_stats()
print(f"Huber ratio: {stats['huber_ratio']:.2%}")
print(f"Ranking ratio: {stats['ranking_ratio']:.2%}")
```

### 2. Real-Time Normalized Loss ✅

**Location**: `src/tiny_icf/loss_adaptive.py`

**Strategy**: Normalizes each loss by its current magnitude: `w_i = 1 / L_i`

**Advantages**:
- Simple and effective baseline
- Automatically balances loss scales
- No hyperparameter tuning needed

**Usage**:
```python
from tiny_icf.loss_adaptive import RealTimeNormalizedLoss

loss_fn = RealTimeNormalizedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
)

loss, diagnostics = loss_fn(predictions, targets, pairs=pairs)
# diagnostics contains component values and ratios
```

### 3. Uncertainty-Weighted Loss ✅

**Location**: `src/tiny_icf/loss_adaptive.py`

**Strategy**: Learns task uncertainty as parameters: `L = Σ(1/(2σ²) * L_i + log(σ))`

**Advantages**:
- Automatically learns optimal weights
- Provides interpretable uncertainty estimates
- Works well with different noise levels

**Usage**:
```python
from tiny_icf.loss_adaptive import UncertaintyWeightedLoss

loss_fn = UncertaintyWeightedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
)

loss, diagnostics = loss_fn(predictions, targets, pairs=pairs)
# diagnostics includes learned uncertainty (sigma) values
```

### 4. Loss Monitoring Utilities ✅

**Location**: `src/tiny_icf/loss_monitoring.py`

**Features**:
- `compute_loss_component_metrics()`: Comprehensive metrics (ratios, dominance warnings, balance scores)
- `detect_loss_imbalance()`: Detects if any component dominates
- `compute_gradient_balance()`: Computes gradient norms and balance
- `log_loss_components()`: Structured logging for loss components

**Usage**:
```python
from tiny_icf.loss_monitoring import (
    compute_loss_component_metrics,
    detect_loss_imbalance,
)

loss_components = {
    'huber': 0.05,
    'ranking': 0.15,
    'neural_ndcg': 0.02,
}

# Check for imbalance
is_imbalanced, dominant = detect_loss_imbalance(loss_components, threshold=0.7)
if is_imbalanced:
    print(f"Warning: {dominant} is dominating!")

# Get comprehensive metrics
metrics = compute_loss_component_metrics(loss_components)
print(f"Balance score: {metrics['balance_score']:.4f}")
```

### 5. Enhanced Training Utilities ✅

**Location**: `src/tiny_icf/training_utils.py`

**New Features**:
- Automatic imbalance detection during training (every 10 batches)
- Component statistics included in training metrics
- Integration with loss monitoring utilities

**What Happens**:
- Training automatically checks for loss imbalance
- Warns if any component dominates (>70%)
- Includes component stats in returned metrics

## Integration with Existing Code

### Backward Compatibility

All changes are **backward compatible**:
- `CombinedLoss` defaults to `track_components=True` but works without it
- Existing training scripts continue to work
- New features are opt-in via imports

### Recommended Usage

**For New Training Scripts**:
```python
from tiny_icf.loss import CombinedLoss
from tiny_icf.loss_monitoring import detect_loss_imbalance

# Use enhanced CombinedLoss
loss_fn = CombinedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
    track_components=True,  # Enable monitoring
)

# During training, check stats periodically
if epoch % 10 == 0:
    stats = loss_fn.get_component_stats()
    is_imbalanced, dominant = detect_loss_imbalance({
        'huber': stats.get('huber_mean', 0),
        'ranking': stats.get('ranking_mean', 0) * loss_fn.rank_weight,
        'neural_ndcg': stats.get('neural_ndcg_mean', 0) * loss_fn.neural_ndcg_weight,
    })
    if is_imbalanced:
        print(f"⚠️  Loss imbalance: {dominant}")
```

**For Advanced Use Cases**:
```python
# Try real-time normalization
from tiny_icf.loss_adaptive import RealTimeNormalizedLoss

loss_fn = RealTimeNormalizedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
)

# Or uncertainty weighting
from tiny_icf.loss_adaptive import UncertaintyWeightedLoss

loss_fn = UncertaintyWeightedLoss(
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,
)
```

## Research-Backed Improvements

### 1. Loss Scale Normalization
- **Problem**: Different losses operate at different scales
- **Solution**: Real-time normalization or adaptive weighting
- **Result**: All losses contribute meaningfully

### 2. Gradient Balance Monitoring
- **Problem**: Gradient conflicts can slow convergence
- **Solution**: Monitor gradient norms per component
- **Result**: Early detection of imbalance issues

### 3. Automatic Imbalance Detection
- **Problem**: One loss can dominate without notice
- **Solution**: Automatic detection with configurable threshold
- **Result**: Proactive warnings during training

### 4. Component Statistics
- **Problem**: Hard to debug multi-objective training
- **Solution**: Comprehensive statistics per component
- **Result**: Better visibility into training dynamics

## Testing

All implementations are tested in `scripts/test_adaptive_losses.py`:

✅ CombinedLoss with component tracking
✅ Real-time normalized loss
✅ Uncertainty-weighted loss
✅ Loss monitoring utilities

## Next Steps (Optional)

1. **GradNorm Implementation**: Full gradient normalization (more complex, requires additional hyperparameter)
2. **Pareto Optimization**: Multi-objective optimization approach (computationally expensive)
3. **Gradient Surgery**: PCGrad for resolving conflicting gradients
4. **Sequential Weighting**: Adjust weights based on training stage

## Files Created/Modified

**New Files**:
- `src/tiny_icf/loss_adaptive.py` - Adaptive loss weighting strategies
- `src/tiny_icf/loss_monitoring.py` - Loss monitoring utilities
- `docs/technical/MULTI_OBJECTIVE_LOSS_RESEARCH.md` - Research findings
- `docs/technical/ADAPTIVE_LOSS_IMPLEMENTATION.md` - This file
- `scripts/test_adaptive_losses.py` - Test script

**Modified Files**:
- `src/tiny_icf/loss.py` - Enhanced CombinedLoss with tracking
- `src/tiny_icf/training_utils.py` - Added imbalance detection
- `src/tiny_icf/__init__.py` - Added exports for new modules

## Summary

✅ **Implemented**: Real-time normalization, uncertainty weighting, component tracking, imbalance detection
✅ **Tested**: All new features working correctly
✅ **Documented**: Research findings and implementation details
✅ **Integrated**: Backward compatible with existing code
✅ **Ready to Use**: Can be used in training scripts immediately

The implementation follows research best practices and provides multiple strategies for balancing multi-objective losses, with comprehensive monitoring and diagnostics.

