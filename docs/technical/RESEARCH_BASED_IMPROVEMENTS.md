# Research-Based Improvements Applied

## Research Findings

Based on deep research into character-level neural networks for word frequency prediction and ranking tasks, the following key insights were identified:

### 1. **Listwise Losses Are Superior to Pairwise**

**Finding**: Pairwise ranking losses may not effectively optimize Spearman correlation because:
- Spearman requires sorting (non-differentiable)
- Pairwise losses optimize local pairs, not global ranking
- Listwise losses (LambdaRank, ApproxNDCG) directly optimize ranking metrics

**Implementation**: Created `src/tiny_icf/loss_listwise.py` with:
- `LambdaRank`: Computes gradients based on NDCG changes from swapping pairs
- `ApproxNDCG`: Differentiable approximation using softmax ranking
- `CombinedListwiseLoss`: Combines Huber + Listwise for absolute + ranking accuracy

### 2. **RBO (Rank-Biased Overlap) Addresses Spearman Limitations**

**Finding**: Spearman correlation can mask poor performance on top-ranked items. RBO:
- Emphasizes top-ranked results (position-biased)
- Reveals whether system excels at identifying best candidates
- More aligned with practical ranking quality

**Implementation**: Created `src/tiny_icf/eval_rbo.py` with:
- `rbo()`: Core RBO computation
- `compute_rbo_from_predictions()`: RBO for predicted vs target rankings
- `compute_rbo_metrics()`: RBO at multiple top-K values

### 3. **Collapse Detection Is Critical**

**Finding**: Models can collapse to predicting constant values (e.g., all 0.0), which:
- Makes Spearman = NaN (no variance)
- Indicates catastrophic training failure
- Needs early detection to prevent wasted training

**Implementation**: Added collapse detection to `train_epoch()`:
- Checks prediction std < 0.01
- Raises RuntimeError immediately
- Prevents wasted training time

### 4. **Loss Component Logging**

**Finding**: Need to verify ranking loss is actually contributing:
- Log Huber and ranking losses separately
- Monitor if ranking loss is decreasing
- Identify if loss scale mismatch exists

**Implementation**: Enhanced `train_epoch()` to:
- Return detailed metrics dictionary
- Log loss components separately (when enabled)
- Track prediction statistics (std, min, max, mean)

### 5. **Architecture Insights from Research**

**Key Findings**:
- Character-level CNNs with parallel filters (kernels 3, 5, 7) are effective
- Multi-scale pooling (max + mean + last) captures more information
- Clamp output (not sigmoid) avoids saturation
- He initialization + mean ICF bias helps prevent collapse

**Current Implementation**: Already follows these best practices.

## Implemented Improvements

### ✅ 1. Listwise Ranking Loss (`src/tiny_icf/loss_listwise.py`)

**LambdaRank Loss**:
- Computes DCG gain for each position
- Calculates lambda (gradient) based on NDCG change from swapping pairs
- Directly optimizes ranking quality

**ApproxNDCG Loss**:
- Uses softmax to approximate ranking (differentiable)
- More stable than LambdaRank for small batches
- Directly optimizes NDCG approximation

**CombinedListwiseLoss**:
- Combines Huber (absolute accuracy) + Listwise (ranking quality)
- Configurable weight for listwise component
- Supports both LambdaRank and ApproxNDCG methods

### ✅ 2. RBO Evaluation Metric (`src/tiny_icf/eval_rbo.py`)

**Features**:
- Position-biased overlap metric
- Emphasizes top-ranked items
- Computes RBO at multiple top-K values (10, 50, 100, full)
- Integrated into `compute_metrics()` in `eval.py`

### ✅ 3. Collapse Detection (`src/tiny_icf/train.py`)

**Features**:
- Checks prediction std < 0.01 each batch
- Raises RuntimeError immediately on collapse
- Prevents wasted training time
- Configurable (can disable for debugging)

### ✅ 4. Enhanced Loss Logging (`src/tiny_icf/train.py`)

**Features**:
- Returns metrics dictionary from `train_epoch()`
- Logs prediction statistics (std, min, max, mean)
- Optional detailed loss component logging
- Tracks training dynamics

### ✅ 5. Training Scripts

**`scripts/train_listwise.py`**:
- Training script using listwise losses
- Supports LambdaRank and ApproxNDCG
- Includes collapse detection
- Logs RBO metrics
- Early stopping support

**`scripts/ablation_loss_study.py`**:
- Systematic comparison of loss configurations:
  1. Huber only (rank_weight=0)
  2. Pairwise (rank_weight=2.0) - current
  3. Pairwise (rank_weight=10.0) - high
  4. Listwise LambdaRank
  5. Listwise ApproxNDCG
- Fair comparison (same seed, same data)
- Reports Spearman, RBO, MAE for each

## Next Steps

1. **Run Ablation Study**: Compare all loss configurations
   ```bash
   uv run scripts/ablation_loss_study.py --data data/word_frequency.csv --epochs 20
   ```

2. **Train with Listwise Loss**: Test if listwise improves Spearman
   ```bash
   uv run scripts/train_listwise.py --data data/word_frequency.csv --epochs 50 --listwise-method lambdarank
   ```

3. **Monitor RBO**: Track RBO alongside Spearman to detect top-K issues

4. **Fix Collapse Issues**: If collapse detected, investigate:
   - Output layer initialization
   - Loss function design
   - Learning rate schedule

## Expected Improvements

Based on research:

1. **Listwise losses should improve Spearman**:
   - LambdaRank directly optimizes NDCG (correlated with Spearman)
   - ApproxNDCG provides stable ranking signal
   - Both should outperform pairwise ranking

2. **RBO will reveal top-K issues**:
   - If Spearman is high but RBO is low, model fails on top items
   - Guides architecture/loss improvements

3. **Collapse detection prevents wasted time**:
   - Early detection saves training cycles
   - Helps identify root causes faster

4. **Loss logging enables diagnosis**:
   - Verify ranking loss is contributing
   - Identify loss scale mismatches
   - Monitor training dynamics

## Research Sources

Key papers and findings:
- Character-level CNNs for NLP (Zhang et al., 2015)
- LambdaRank for learning to rank (Burges et al., 2006)
- ApproxNDCG for differentiable ranking (Qin et al., 2010)
- RBO for position-biased evaluation (Webber et al., 2010)
- Frequency-aware training in NMT (Zhang et al., 2021)
- Long-tailed phenomena in neural models (Raunak et al., 2020)

