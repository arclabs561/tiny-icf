# Parallel Filters and Loss Function Design

## 1. Different Filters for Different "Views"

### Yes! We Have Parallel Filters

**We use 3 parallel 1D CNNs with different kernel sizes** - these are indeed different "views" of the input:

```python
# From src/tiny_icf/model.py
self.conv3 = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)  # Trigrams
self.conv5 = nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2)    # 5-grams
self.conv7 = nn.Conv1d(emb_dim, conv_channels, kernel_size=7, padding=3)   # 7-grams
```

### What Each Filter Captures

**Kernel Size 3 (Trigrams)**:
- Short character patterns: "ing", "pre", "tion", "ed"
- Prefixes/suffixes: "un-", "-ly", "-er"
- Common morphological markers

**Kernel Size 5 (5-grams)**:
- Medium patterns: "graph", "phone", "ology"
- Word roots and stems
- Common character sequences

**Kernel Size 7 (7-grams)**:
- Longer patterns: "graphic", "phonic", "ological"
- Complex affixes and compound structures
- Multi-character morphological units

### How They're Combined

**Multi-scale pooling** combines all views:
```python
# Each filter produces features at different scales
p3_max, p3_mean, p3_last = pool(conv3_output)  # 3 views from kernel 3
p5_max, p5_mean, p5_last = pool(conv5_output)  # 3 views from kernel 5
p7_max, p7_mean, p7_last = pool(conv7_output)  # 3 views from kernel 7

# Concatenate: 3 kernels × 3 pooling methods = 9 feature sets
combined = [p3_max, p3_mean, p3_last, p5_max, p5_mean, p5_last, p7_max, p7_mean, p7_last]
```

**Why this works**:
- Different kernel sizes capture patterns at different scales
- Multi-scale pooling preserves information from different positions
- Combining all views gives richer representation than single filter

### Could We Add More Views?

**Potential enhancements**:
1. **More kernel sizes**: 2, 4, 6, 8, 9 (currently only 3, 5, 7)
2. **Dilated convolutions**: Capture longer-range patterns without increasing parameters
3. **Multi-resolution**: Process input at different resolutions (subsampling)
4. **Character-level attention**: Learn which character positions matter most

**Trade-off**: More views = more parameters = larger model (we want <50k params)

---

## 2. Is Rank Alignment the Best? Asymmetric Penalties

### Current Loss Functions (Symmetric)

**We currently use**:
1. **Huber Loss**: Symmetric, treats all errors equally
   ```python
   loss = huber(pred - target)  # Same penalty for +0.5 and -0.5 error
   ```

2. **Ranking Loss**: Margin-based, but symmetric
   ```python
   loss = max(0, margin - (pred_diff * sign(target_diff)))  # Same penalty for wrong direction
   ```

3. **Spearman Loss**: Ranking correlation, symmetric
   ```python
   loss = 1 - spearman_corr(pred, target)  # Treats all ranking errors equally
   ```

### The Problem: Some Errors Are Worse Than Others

**Your insight is correct!** Consider:

**Example 1: Polar Opposite vs. Slight Error**
- Word: "the" (should be 0.0)
  - Prediction: 1.0 → **Polar opposite** (terrible!)
  - Prediction: 0.1 → **Slightly off** (acceptable)
- **Current loss**: Both might have similar penalty (wrong!)
- **Should be**: Polar opposite penalized MUCH more

**Example 2: Large vs. Small Errors**
- Word: "xylophone" (should be 0.95)
  - Prediction: 0.05 → **Huge error** (0.9 off)
  - Prediction: 0.90 → **Small error** (0.05 off)
- **Current loss**: Linear penalty (0.9 vs 0.05)
- **Should be**: Exponential penalty for large errors

**Example 3: Direction Matters**
- Predicting common word as rare: **Bad** (filters out important words)
- Predicting rare word as common: **Less bad** (keeps word, just wrong weight)
- **Current loss**: Same penalty for both directions
- **Should be**: Asymmetric penalty

### Proposed Solutions

#### 1. **Asymmetric Huber Loss**

Penalize large errors more heavily, and errors in one direction more than the other:

```python
def asymmetric_huber_loss(pred, target, delta=0.1, asymmetry_factor=2.0):
    """
    Asymmetric Huber loss:
    - Large errors penalized more (exponential)
    - Errors in "common → rare" direction penalized more
    """
    error = pred - target
    
    # Base Huber loss
    huber = F.smooth_l1_loss(pred, target, beta=delta)
    
    # Asymmetric penalty: common → rare is worse than rare → common
    if error > 0:  # Predicted more rare than actual
        # This is worse: we're filtering out common words
        asymmetric_penalty = asymmetry_factor * F.relu(error)
    else:  # Predicted more common than actual
        # Less bad: we're keeping rare words
        asymmetric_penalty = F.relu(-error)
    
    return huber + asymmetric_penalty
```

#### 2. **Magnitude-Weighted Ranking Loss**

Weight ranking errors by the magnitude of the ICF difference:

```python
def magnitude_weighted_ranking_loss(pred1, pred2, target1, target2, margin=0.1):
    """
    Ranking loss weighted by ICF difference magnitude.
    
    Large ICF differences (common vs rare) are more important to get right.
    """
    target_diff = abs(target1 - target2)  # How different are they?
    pred_diff = pred1 - pred2
    
    # Weight by target difference: larger differences = more important
    weight = 1.0 + target_diff  # At least 1.0, scales with difference
    
    # Standard margin loss
    violation = F.relu(margin - pred_diff * torch.sign(target_diff))
    
    return (weight * violation).mean()
```

#### 3. **Focal Ranking Loss**

Penalize hard examples (large errors) more heavily:

```python
def focal_ranking_loss(pred1, pred2, target1, target2, margin=0.1, gamma=2.0):
    """
    Focal ranking loss: focus on hard examples (large errors).
    
    Similar to focal loss for classification, but for ranking.
    """
    target_diff = target1 - target2
    pred_diff = pred1 - pred2
    
    # Standard margin loss
    base_loss = F.relu(margin - pred_diff * torch.sign(target_diff))
    
    # Focal weighting: large errors get exponentially more weight
    error_magnitude = abs(pred_diff - target_diff)
    focal_weight = (1.0 + error_magnitude) ** gamma
    
    return (focal_weight * base_loss).mean()
```

#### 4. **Direction-Aware Loss**

Different penalties for different error directions:

```python
def direction_aware_loss(pred, target, common_penalty=2.0, rare_penalty=1.0):
    """
    Direction-aware loss:
    - Predicting common word as rare: HIGH penalty (common_penalty)
    - Predicting rare word as common: LOW penalty (rare_penalty)
    """
    error = pred - target
    
    if error > 0:  # Predicted more rare (common → rare)
        # This is worse: filtering out important common words
        return common_penalty * F.smooth_l1_loss(pred, target)
    else:  # Predicted more common (rare → common)
        # Less bad: keeping rare words
        return rare_penalty * F.smooth_l1_loss(pred, target)
```

#### 5. **Combined Asymmetric Loss**

Combine all perspectives:

```python
class AsymmetricICFLoss(nn.Module):
    """
    Asymmetric loss that:
    1. Penalizes large errors exponentially
    2. Penalizes common→rare more than rare→common
    3. Weights ranking by ICF difference magnitude
    4. Uses focal weighting for hard examples
    """
    
    def __init__(
        self,
        huber_delta=0.1,
        asymmetry_factor=2.0,  # Common→rare penalty multiplier
        focal_gamma=2.0,  # Focal loss exponent
        magnitude_weight=True,  # Weight ranking by ICF difference
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.asymmetry_factor = asymmetry_factor
        self.focal_gamma = focal_gamma
        self.magnitude_weight = magnitude_weight
    
    def forward(self, predictions, targets, pairs=None):
        # 1. Asymmetric Huber loss
        error = predictions - targets
        huber_base = F.smooth_l1_loss(predictions, targets, beta=self.huber_delta)
        
        # Asymmetric penalty
        asymmetric_penalty = torch.where(
            error > 0,  # Common → rare
            self.asymmetry_factor * F.relu(error),
            F.relu(-error)  # Rare → common (less penalty)
        )
        huber_loss = huber_base + asymmetric_penalty.mean()
        
        # 2. Magnitude-weighted ranking loss
        if pairs is not None:
            idx1, idx2 = pairs[:, 0], pairs[:, 1]
            pred1, pred2 = predictions[idx1], predictions[idx2]
            target1, target2 = targets[idx1], targets[idx2]
            
            target_diff = target1 - target2
            pred_diff = pred1 - pred2
            
            # Base margin loss
            margin = 0.1
            base_rank_loss = F.relu(margin - pred_diff * torch.sign(target_diff))
            
            # Magnitude weighting
            if self.magnitude_weight:
                weight = 1.0 + abs(target_diff)  # Larger differences = more important
                rank_loss = (weight * base_rank_loss).mean()
            else:
                rank_loss = base_rank_loss.mean()
            
            # Focal weighting for hard examples
            error_magnitude = abs(pred_diff - target_diff)
            focal_weight = (1.0 + error_magnitude) ** self.focal_gamma
            rank_loss = (focal_weight * base_rank_loss).mean()
        else:
            rank_loss = torch.tensor(0.0, device=predictions.device)
        
        return huber_loss + rank_loss
```

### Why Current Approach Might Be Suboptimal

**Current ranking alignment**:
- ✅ Preserves relative ordering
- ✅ Works well for ranking quality
- ❌ Doesn't penalize large errors enough
- ❌ Doesn't distinguish error directions
- ❌ Doesn't weight by error magnitude

**Your insight**: Ranking alignment is good, but we need **asymmetric, magnitude-aware penalties** to handle:
1. **Polar opposites** (0.0 → 1.0) should be penalized MUCH more
2. **Large errors** (0.9 off) should be penalized exponentially more
3. **Error direction** (common→rare worse than rare→common)

### Recommendation

**Implement combined asymmetric loss**:
1. Keep ranking alignment (it's good!)
2. Add asymmetric penalties (common→rare worse)
3. Add magnitude weighting (large errors penalized more)
4. Add focal weighting (hard examples focused on)

**Expected impact**: Better handling of edge cases, improved performance on extreme values (very common/very rare words)

---

## Summary

### Parallel Filters ✅
- **Yes**, we have 3 different filters (kernel sizes 3, 5, 7)
- Each captures different character n-gram patterns
- Combined via multi-scale pooling
- Could add more views (dilated convs, more kernels) but trade-off with model size

### Loss Function Design ⚠️
- **Current**: Symmetric losses (all errors treated equally)
- **Problem**: Polar opposites and large errors not penalized enough
- **Solution**: Asymmetric, magnitude-weighted, focal loss
- **Recommendation**: Implement `AsymmetricICFLoss` that combines all perspectives

**Next steps**: Implement asymmetric loss and compare with current approach!

