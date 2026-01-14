# Additional Loss Function Criteria and Research Review

## Executive Summary

After reviewing rank-relax documentation, research literature, and our current implementation, we've identified **several important criteria we haven't fully explored**:

1. **Monotonicity Constraints** - Enforcing structure (e.g., longer words → higher ICF)
2. **Focal Loss for Hard Examples** - Downweighting easy examples
3. **Multiple Ranking Methods from rank-relax** - We only use sigmoid, but neural_sort/probabilistic might be better
4. **Adaptive Regularization Strength** - Tuning based on data scale
5. **Entropy Regularization** - Minimizing representation complexity
6. **Quantile Regression** - For uncertainty intervals
7. **Temperature Scaling** - Post-hoc calibration (we have calibration loss but not temperature scaling)
8. **Distribution Matching Beyond KL** - Wasserstein distance, etc.

---

## 1. What We Already Have ✅

### Calibration
- ✅ `calibration_loss` in `loss_multi.py` (KL divergence on binned distributions)
- ✅ `expected_calibration_error` in `eval.py`
- ✅ Calibration metrics tracking

### Uncertainty Quantification
- ✅ `eval_uncertainty.py` with bootstrap confidence intervals
- ✅ Quantile regression intervals
- ✅ Ensemble uncertainty (if multiple models)

### Robustness
- ✅ `eval_robustness.py` with adversarial, OOD, noise testing
- ✅ Robustness score computation

### Confidence Estimation
- ✅ Confidence estimates from feature activations in `model.py`
- ✅ Confidence in predictions

### Distribution Matching
- ✅ Calibration loss uses KL divergence
- ✅ Distribution similarity metrics in `eval.py`

### Consistency
- ✅ `consistency_loss` for similar words in `loss_multi.py`

---

## 2. What We're Missing ⚠️

### 2.1 Monotonicity Constraints

**Research Finding**: Monotonicity constraints improve generalization and interpretability.

**What We Could Enforce**:
- **Word length → ICF**: Longer words tend to be rarer (with exceptions)
- **Character frequency → ICF**: Words with rare characters tend to be rarer
- **Morphological complexity → ICF**: More complex words tend to be rarer

**Implementation**:
```python
def monotonicity_loss(predictions, features, constraints):
    """
    Enforce monotonicity constraints.
    
    Args:
        predictions: [batch, 1] ICF predictions
        features: Dict of feature tensors (length, char_freq, etc.)
        constraints: Dict mapping feature names to monotonicity direction
    
    Returns:
        Loss penalizing monotonicity violations
    """
    loss = 0.0
    for feature_name, direction in constraints.items():
        feature = features[feature_name]  # [batch]
        # If direction='increasing': longer words should have higher ICF
        # If direction='decreasing': rare chars should have higher ICF
        
        # Compute correlation between feature and predictions
        # Penalize if correlation has wrong sign
        correlation = compute_correlation(feature, predictions)
        if direction == 'increasing' and correlation < 0:
            loss += -correlation  # Penalize negative correlation
        elif direction == 'decreasing' and correlation > 0:
            loss += correlation  # Penalize positive correlation
    
    return loss
```

**Research Support**: 
- Monotonicity constraints improve generalization (ICML 2009)
- Constrained architectures provide hard guarantees (arXiv 2022)

### 2.2 Focal Loss for Hard Examples

**Research Finding**: Focal loss downweights easy examples, focusing on hard cases.

**Current State**: We have asymmetric loss, but not focal weighting.

**Implementation**:
```python
def focal_icf_loss(predictions, targets, gamma=2.0):
    """
    Focal loss for ICF: focus on hard examples (large errors).
    
    Args:
        predictions: [batch, 1] model predictions
        targets: [batch, 1] ground truth
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Focal loss value
    """
    error = torch.abs(predictions - targets)
    base_loss = F.smooth_l1_loss(predictions, targets)
    
    # Focal weighting: large errors get exponentially more weight
    focal_weight = (1.0 + error) ** gamma
    
    return (focal_weight * base_loss).mean()
```

**Research Support**: 
- Focal loss for object detection (arXiv 2017)
- Effective for class imbalance and hard example mining

### 2.3 Multiple Ranking Methods from rank-relax

**Current State**: We only use `rank_relax.soft_rank` (sigmoid method).

**Available Methods** (from rank-relax):
1. **Sigmoid** (default): O(n²), general purpose
2. **NeuralSort**: Temperature-scaled softmax, sharper rankings
3. **Probabilistic (SoftRank)**: Gaussian smoothing, uncertainty modeling
4. **SmoothI**: Exponential scaling, alternative gradient profiles

**Research Finding**: Different methods have different gradient profiles and may be better for different tasks.

**Implementation**:
```python
# In loss_unified.py, we could try different methods:
ranks = rank_relax.soft_rank_with_method(
    values, 
    regularization_strength=1.0,
    method="neural_sort"  # or "probabilistic", "smooth_i"
)
```

**Recommendation**: Experiment with `neural_sort` for sharper rankings, `probabilistic` for uncertainty-aware ranking.

### 2.4 Adaptive Regularization Strength

**From rank-relax PARAMETER_TUNING.md**:
> "Match the parameter to the scale of differences in your values: `regularization_strength ≈ 1.0 / typical_difference_between_values`"

**Current State**: We use fixed `regularization_strength=1.0` or `1e-2`.

**Better Approach**: Adaptive based on batch statistics:
```python
def adaptive_regularization_strength(predictions, targets):
    """
    Adaptively set regularization strength based on data scale.
    
    Rule of thumb: reg_strength ≈ 1.0 / typical_difference
    """
    # Compute typical difference between values
    pred_diff = torch.std(predictions)
    target_diff = torch.std(targets)
    typical_diff = (pred_diff + target_diff) / 2.0
    
    # Set regularization strength
    reg_strength = 1.0 / (typical_diff + 1e-6)
    
    # Clamp to reasonable range
    return torch.clamp(reg_strength, 0.1, 100.0)
```

**Research Support**: rank-relax documentation emphasizes matching regularization to data scale.

### 2.5 Entropy Regularization

**Research Finding**: Entropy regularization minimizes representation complexity, improving efficiency and robustness.

**Implementation**:
```python
def entropy_regularization_loss(features):
    """
    Entropy regularization: minimize representation complexity.
    
    Encourages sparse, structured representations.
    """
    # Compute entropy of feature activations
    # Higher entropy = more uniform = less structured
    # Lower entropy = more peaked = more structured
    
    # Normalize features to probabilities
    probs = F.softmax(features, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    
    # Penalize high entropy (encourage structure)
    return entropy
```

**Research Support**: 
- Entropy regularization for sparsity (arXiv 2025)
- Improves both efficiency and robustness

### 2.6 Quantile Regression for Uncertainty

**Research Finding**: Quantile regression provides principled uncertainty intervals.

**Current State**: We have `quantile_regression_intervals` in `eval_uncertainty.py`, but not as a loss function.

**Implementation**:
```python
def quantile_loss(predictions, targets, quantile=0.5):
    """
    Quantile regression loss.
    
    Args:
        predictions: [batch, 1] model predictions
        targets: [batch, 1] ground truth
        quantile: Desired quantile (0.5 = median, 0.9 = 90th percentile)
    
    Returns:
        Quantile loss value
    """
    error = predictions - targets
    
    # Asymmetric weighting
    loss = torch.max(
        quantile * error,
        (quantile - 1.0) * error
    )
    
    return loss.mean()
```

**Research Support**: 
- Quantile regression for uncertainty (OpenReview 2023)
- Calibration-guided quantile regression improves both sharpness and calibration

### 2.7 Temperature Scaling for Calibration

**Research Finding**: Temperature scaling is simple and effective for post-hoc calibration.

**Current State**: We have calibration loss during training, but not temperature scaling.

**Implementation**:
```python
class TemperatureScaledModel(nn.Module):
    """
    Wraps model with learnable temperature parameter for calibration.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.temperature
```

**Research Support**: 
- Temperature scaling for calibration (ICLR 2017)
- Simple, effective, single parameter

### 2.8 Distribution Matching Beyond KL

**Research Finding**: Wasserstein distance may be better than KL for distribution matching.

**Current State**: We use KL divergence in calibration loss.

**Alternative**:
```python
def wasserstein_loss(pred_dist, target_dist):
    """
    Wasserstein distance for distribution matching.
    
    Better than KL when distributions have disjoint support.
    """
    # For 1D distributions, Wasserstein = L1 distance of CDFs
    pred_cdf = torch.cumsum(pred_dist, dim=-1)
    target_cdf = torch.cumsum(target_dist, dim=-1)
    
    return torch.abs(pred_cdf - target_cdf).mean()
```

**Research Support**: 
- Wasserstein distance for GANs (arXiv 2017)
- Better gradient flow than KL divergence

---

## 3. rank-relax Capabilities We're Not Using

### 3.1 Multiple Ranking Methods

**Available** (from rank-relax README):
- `soft_rank_with_method(values, reg_strength, method="neural_sort")`
- Methods: `"sigmoid"`, `"neural_sort"`, `"probabilistic"`, `"smooth_i"`

**Current Usage**: Only `soft_rank` (sigmoid default)

**Recommendation**: Experiment with:
- **NeuralSort**: For sharper rankings (better for late training)
- **Probabilistic**: For uncertainty-aware ranking
- **SmoothI**: For alternative gradient profiles

### 3.2 Analytical Gradients

**Available**: `rank_relax.spearman_loss_gradient()` provides analytical gradients.

**Current Usage**: We rely on autograd (which is fine, but analytical might be faster/more stable)

**Recommendation**: Consider using analytical gradients for:
- Faster training
- More stable gradients
- Better numerical precision

### 3.3 Batch Processing

**Available**: `rank_relax.soft_rank_batch()` for parallel processing.

**Current Usage**: We process batches sequentially in Python.

**Recommendation**: Use batch processing for:
- Faster training on large batches
- Better GPU utilization

### 3.4 Adaptive Regularization

**From PARAMETER_TUNING.md**:
- Match `regularization_strength` to data scale
- Use annealing: start low, increase over time
- Test by comparing soft ranks to discrete ranks

**Current Usage**: Fixed `regularization_strength=1.0` or `1e-2`

**Recommendation**: 
- Compute adaptive strength: `1.0 / typical_difference`
- Use annealing schedule
- Validate on held-out data

---

## 4. Research We Should Review

### 4.1 Monotonicity in Neural Networks

**Key Papers**:
- "Monotonicity in Neural Networks" (ICML 2009)
- "Monotonic Neural Networks" (arXiv 2022)

**Key Insights**:
- Constrained architectures provide hard guarantees
- Loss function penalties provide probabilistic guarantees
- Monotonicity improves interpretability and generalization

**Application to ICF**:
- Enforce: longer words → higher ICF (with exceptions)
- Enforce: rare characters → higher ICF
- Could improve performance on edge cases

### 4.2 Focal Loss and Hard Example Mining

**Key Papers**:
- "Focal Loss for Dense Object Detection" (arXiv 2017)
- "Hard Example Mining" (various)

**Key Insights**:
- Downweighting easy examples improves focus on hard cases
- Particularly effective for class imbalance
- Can be combined with other losses

**Application to ICF**:
- Focus on words where prediction is difficult
- Downweight easy cases (very common/very rare words)
- Could improve performance on ambiguous words

### 4.3 Calibration-Guided Quantile Regression

**Key Papers**:
- "Calibration-Guided Quantile Regression" (OpenReview 2023)

**Key Insights**:
- Quantile regression + calibration improves both sharpness and calibration
- Provides principled uncertainty intervals
- Better than separate calibration and quantile regression

**Application to ICF**:
- Provide uncertainty intervals for predictions
- Improve calibration while maintaining accuracy
- Enable risk-aware decision making

### 4.4 Entropy Regularization

**Key Papers**:
- "Entropy Regularization for Structured Sparsity" (arXiv 2025)

**Key Insights**:
- Minimizes representation complexity
- Induces structured sparsity
- Improves both efficiency and robustness

**Application to ICF**:
- Encourage sparse, structured character patterns
- Reduce model complexity
- Improve generalization

### 4.5 Wasserstein Distance for Distribution Matching

**Key Papers**:
- "Wasserstein GAN" (arXiv 2017)
- "Optimal Transport" (Cuturi 2013)

**Key Insights**:
- Better gradient flow than KL divergence
- Respects geometric structure
- Continuous even when distributions have disjoint support

**Application to ICF**:
- Better distribution matching than KL
- Smoother optimization landscape
- Could improve calibration

---

## 5. Priority Recommendations

### Priority 1: High Impact, Low Effort

1. **Adaptive Regularization Strength** ⭐⭐⭐
   - Match to data scale: `1.0 / typical_difference`
   - Easy to implement, significant impact

2. **Temperature Scaling** ⭐⭐⭐
   - Single parameter, post-hoc calibration
   - Simple wrapper around model

3. **Try NeuralSort Method** ⭐⭐
   - Just change `method="neural_sort"` in rank-relax calls
   - Might provide sharper rankings

### Priority 2: Medium Impact, Medium Effort

4. **Focal Loss Component** ⭐⭐
   - Add to `AsymmetricICFLoss`
   - Focus on hard examples

5. **Monotonicity Constraints** ⭐⭐
   - Enforce word length → ICF relationship
   - Could improve edge cases

6. **Quantile Regression Loss** ⭐⭐
   - For uncertainty intervals
   - Principled uncertainty quantification

### Priority 3: Lower Priority, Higher Effort

7. **Entropy Regularization** ⭐
   - Minimize representation complexity
   - Nice to have, less critical

8. **Wasserstein Distance** ⭐
   - Alternative to KL divergence
   - Might be better, but KL works

9. **Analytical Gradients from rank-relax** ⭐
   - Faster/more stable
   - But autograd works fine

---

## 6. Implementation Plan

### Phase 1: Quick Wins (1-2 days)
- [ ] Adaptive regularization strength
- [ ] Temperature scaling wrapper
- [ ] Try NeuralSort method in experiments

### Phase 2: Medium Effort (3-5 days)
- [ ] Add focal loss to `AsymmetricICFLoss`
- [ ] Implement monotonicity constraints
- [ ] Add quantile regression loss

### Phase 3: Advanced (1 week+)
- [ ] Entropy regularization
- [ ] Wasserstein distance for distribution matching
- [ ] Analytical gradients from rank-relax

---

## 7. rank-relax Notes Summary

### Key Insights from rank-relax Documentation

1. **Parameter Tuning**:
   - Match `regularization_strength` to data scale
   - Rule: `reg_strength ≈ 1.0 / typical_difference`
   - Use annealing: start low, increase over time

2. **Multiple Methods**:
   - Sigmoid: General purpose, O(n²)
   - NeuralSort: Sharper rankings, different gradient profile
   - Probabilistic: Uncertainty-aware
   - SmoothI: Alternative gradient profiles

3. **Mathematical Foundations**:
   - Permutahedron projection: O(n log n), exact
   - Optimal Transport: O(n²), dense gradients
   - Sorting Networks: O(n log² n), parallel-friendly
   - LapSum: O(n log n), fastest, closed-form

4. **Current Implementation**:
   - Uses sigmoid method (O(n²))
   - Could upgrade to faster methods (Permutahedron, LapSum)
   - But sigmoid works fine for our scale (n < 1000)

---

## 8. Summary: What We Should Do Next

### Immediate Actions

1. **Implement adaptive regularization strength** - Easy, high impact
2. **Add temperature scaling** - Simple, effective calibration
3. **Experiment with NeuralSort** - Just change method parameter

### Short-term

4. **Add focal loss to asymmetric loss** - Focus on hard examples
5. **Implement monotonicity constraints** - Enforce structure
6. **Add quantile regression loss** - Uncertainty intervals

### Research to Review

- Monotonicity in neural networks (ICML 2009, arXiv 2022)
- Focal loss (arXiv 2017)
- Calibration-guided quantile regression (OpenReview 2023)
- Entropy regularization (arXiv 2025)
- Wasserstein distance (arXiv 2017)

---

**Status**: We have good coverage of basic criteria, but missing several advanced techniques that could improve performance. Priority: Adaptive regularization and temperature scaling (easy wins).

