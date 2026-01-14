# Multi-Objective Loss Function Research & Best Practices

## Executive Summary

This document synthesizes research findings on combining multiple loss functions, focusing on practical strategies for balancing Huber + Ranking + NeuralNDCG losses in our ICF prediction model.

## Key Findings

### 1. The Core Challenge: Loss Scale Mismatch

**Problem**: Different loss functions operate at different scales:
- Huber loss: Typically 0.01-0.1 range
- Ranking loss: Can be 0.1-1.0 range  
- NeuralNDCG: Often 0.01-0.5 range

**Impact**: Without proper weighting, the largest-magnitude loss dominates optimization, effectively ignoring other objectives.

**Solution**: Normalize losses to comparable scales before combining, or use adaptive weighting strategies.

### 2. Loss Weighting Strategies

#### A. Static Weighting (Current Approach)
```python
L_total = w1 * L_huber + w2 * L_ranking + w3 * L_neural_ndcg
```

**Pros**: Simple, interpretable, easy to debug

**Cons**: 
- Requires manual tuning
- Doesn't adapt to training dynamics
- May become suboptimal as training progresses

**Best Practices**:
1. Start with equal weights (1.0, 1.0, 1.0)
2. Monitor individual loss magnitudes during first few epochs
3. Adjust weights to balance magnitudes: `w_i = scale_factor / L_i_magnitude`
4. Typical starting point: `[1.0, 2.0, 0.5]` for [Huber, Ranking, NeuralNDCG]

#### B. Gradient-Based Balancing (GradNorm)

**Principle**: Balance gradients across losses rather than loss values.

**How it works**:
- Compute gradients for each loss component
- Normalize gradients to have similar magnitudes
- Adjust weights to maintain gradient balance

**Advantages**:
- Automatically adapts to training dynamics
- Prevents one loss from dominating
- Ensures all objectives contribute meaningfully

**Implementation**:
```python
# Pseudo-code
gradients = [compute_grad(L_i) for L_i in losses]
grad_norms = [g.norm() for g in gradients]
target_norm = mean(grad_norms)
weights = [target_norm / (g_norm + eps) for g_norm in grad_norms]
```

**Research Finding**: GradNorm has shown significant improvements in multi-task learning, particularly when tasks have different learning rates or difficulty levels.

#### C. Uncertainty Weighting (Homoscedastic Uncertainty)

**Principle**: Weight losses based on task uncertainty estimates.

**How it works**:
- Tasks with higher uncertainty (harder to learn) get higher weights
- Tasks with lower uncertainty (easier to learn) get lower weights
- Uncertainty is learned as a model parameter

**Formula**:
```
L_total = Σ (1/(2*σ_i²) * L_i + log(σ_i))
```
where σ_i is the learned uncertainty for task i.

**Advantages**:
- Automatically balances task difficulty
- Provides interpretable uncertainty estimates
- Works well when tasks have different noise levels

**Disadvantages**:
- Requires learning additional parameters
- Can be sensitive to initialization

#### D. Adaptive Real-Time Weighting

**Principle**: Dynamically adjust weights based on historical loss values.

**How it works**:
- Track loss values over recent epochs
- Adjust weights inversely proportional to loss magnitude
- Use exponential moving average for stability

**Implementation**:
```python
# Track moving averages
ema_losses = [EMA(L_i) for L_i in losses]
# Normalize to sum to 1
weights = softmax([1.0 / (ema + eps) for ema in ema_losses])
```

### 3. Gradient Conflicts and Solutions

#### Problem: Conflicting Gradients

Different losses can produce contradictory gradient signals:
- **Huber loss**: Pushes predictions toward mean (robust to outliers)
- **Ranking loss**: Cares about relative order, not absolute values
- **NeuralNDCG**: Focuses on top-k ranking quality

**Impact**: Gradients can cancel out, slowing or preventing convergence.

#### Solutions:

1. **Gradient Clipping**: Limit gradient magnitudes to prevent extreme conflicts
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Gradient Masking**: Only apply certain losses to specific samples
   - Use Huber for all samples
   - Use Ranking only for pairs with sufficient ICF difference
   - Use NeuralNDCG only for batches with diverse ICF values

3. **Sequential Weighting**: Emphasize different losses at different training stages
   - Early: Focus on Huber (stability)
   - Mid: Emphasize Ranking (ordering)
   - Late: Emphasize NeuralNDCG (top-k quality)

### 4. Scale Normalization Strategies

#### A. Loss Value Normalization
```python
# Normalize each loss to [0, 1] range
L_normalized = (L - L_min) / (L_max - L_min + eps)
```

#### B. Gradient Magnitude Normalization
```python
# Normalize by gradient norms
grad_norms = [g.norm() for g in gradients]
L_weighted = [w * L for w, L in zip(normalized_weights, losses)]
```

#### C. Relative Weighting
```python
# Weight by inverse of loss magnitude
weights = [1.0 / (L_i.mean() + eps) for L_i in losses]
weights = weights / sum(weights)  # Normalize to sum to 1
```

### 5. Monitoring and Diagnostics

#### Critical Metrics to Track:

1. **Individual Loss Values**: Each component separately
2. **Loss Ratios**: `L_i / L_total` to detect dominance
3. **Gradient Magnitudes**: Per-loss gradient norms
4. **Gradient Cosines**: Angle between gradients (conflict detection)
5. **Validation Metrics**: Task-specific metrics (Spearman, NDCG, MAE)

#### Warning Signs:

- **Loss Plateau**: One loss stops decreasing while others continue
- **Gradient Vanishing**: One loss's gradients become negligible
- **Oscillating Weights**: Weights fluctuate wildly (instability)
- **Metric Divergence**: Training loss decreases but validation metrics don't improve

### 6. Practical Recommendations for Our Model

#### Current Configuration Analysis:

```python
CombinedLoss(
    huber_delta=0.1,
    rank_margin=0.1,
    rank_weight=2.0,  # Ranking emphasized
    use_neural_ndcg=True,
    neural_ndcg_weight=0.5,  # NeuralNDCG moderate
    use_listwise_ranking=False,  # Not currently used
)
```

#### Recommended Improvements:

1. **Add Gradient Monitoring**:
   ```python
   # Track gradient norms per loss component
   grad_norms = {
       'huber': huber_grad.norm(),
       'ranking': ranking_grad.norm(),
       'neural_ndcg': ndcg_grad.norm(),
   }
   ```

2. **Implement Adaptive Weighting** (Optional):
   ```python
   # Adjust weights based on gradient balance
   if grad_norms['ranking'] / grad_norms['neural_ndcg'] > 10:
       # Ranking dominates, increase NeuralNDCG weight
       neural_ndcg_weight *= 1.1
   ```

3. **Add Loss Component Logging**:
   - Log individual losses separately
   - Track loss ratios over time
   - Alert when one component dominates (>80% of total)

4. **Consider Sequential Weighting**:
   ```python
   if epoch < 10:
       # Early: Focus on stability
       rank_weight = 1.0
       neural_ndcg_weight = 0.1
   elif epoch < 50:
       # Mid: Balance both
       rank_weight = 2.0
       neural_ndcg_weight = 0.5
   else:
       # Late: Emphasize ranking quality
       rank_weight = 2.0
       neural_ndcg_weight = 1.0
   ```

### 7. Advanced Techniques (Future Exploration)

#### A. Pareto Multi-Task Learning
- Optimize for Pareto-optimal solutions
- Maintain multiple models on Pareto frontier
- Let user choose trade-off point

#### B. Multi-Gradient Descent
- Compute gradients for each objective separately
- Find descent direction that improves all objectives
- More complex but theoretically sound

#### C. Curriculum Learning
- Start with easier objectives (Huber)
- Gradually introduce harder ones (Ranking, NeuralNDCG)
- Helps with convergence and stability

### 8. Common Pitfalls to Avoid

1. **Ignoring Loss Scales**: Always normalize or weight appropriately
2. **Static Weights Forever**: Revisit weights periodically
3. **Not Monitoring Components**: Track each loss separately
4. **Overweighting Easy Losses**: Don't let simple losses dominate
5. **Ignoring Gradient Conflicts**: Monitor gradient directions
6. **Validation Metric Mismatch**: Ensure losses align with evaluation metrics

### 9. Research-Backed Best Practices

1. **Start Simple**: Begin with static, equal weights
2. **Monitor Closely**: Track all components for first 10-20 epochs
3. **Adjust Gradually**: Make small weight adjustments (10-20% changes)
4. **Validate Changes**: Always check validation metrics after weight changes
5. **Document Decisions**: Record why weights were chosen/changed
6. **Consider Task Difficulty**: Harder tasks may need higher weights
7. **Balance is Key**: No single loss should dominate (>70% of total)

## References

1. GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018)
2. Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
3. Pareto Multi-Task Learning (Lin et al., 2019)
4. Strategies for Balancing Multiple Loss Functions (Medium, 2024)
5. Multi-Objective Loss Balancing for Physics-Informed Deep Learning (ScienceDirect, 2025)
6. Adaptive Real-Time Multi-Loss Function Optimization (arXiv, 2024)

## Implementation Checklist

- [ ] Add gradient norm monitoring per loss component
- [ ] Implement loss component logging
- [ ] Add loss ratio tracking
- [ ] Create diagnostic plots (loss components over time)
- [ ] Document current weight choices and rationale
- [ ] Set up alerts for loss dominance (>80%)
- [ ] Consider implementing adaptive weighting (optional)
- [ ] Validate weight changes on held-out set
- [ ] Document weight tuning process

