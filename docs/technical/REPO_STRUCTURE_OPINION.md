# Repository Structure Review - Opinionated Analysis

## Executive Summary

The repository has **good core structure** but suffers from **file fragmentation** and **lack of consolidation**. Too many similar files doing similar things makes it hard to know what to use.

## What Makes Sense ✅

### Core Structure
- `src/tiny_icf/` - Clean module structure
- `scripts/` - Training and utility scripts  
- `data/` - Data files
- `models/` - Model checkpoints
- `docs/` - Documentation (well organized after cleanup)

### Good Patterns
- PEP 723 inline dependencies (`/// script` blocks) - **Excellent**
- Unified training utilities (`training_utils.py`) - **Good consolidation**
- Centralized initialization (`initialization.py`) - **Good consolidation**
- Clear separation of concerns

## What Doesn't Make Sense ❌

### 1. Loss Files: TOO MANY (CONSOLIDATE!)

**Current State:**
- `loss.py` - Main loss (Huber + Ranking)
- `loss_research.py` - Research losses (NeuralNDCG, Softmax CE, Focal)
- `loss_listwise.py` - Listwise losses (LambdaRank, ApproxNDCG)
- `loss_multi.py` - Multi-objective losses
- `loss_diffsort.py` - DiffSort losses
- `temporal_loss.py` - Temporal losses

**Problem**: 
- Scattered across 6 files
- Hard to find what you need
- Unclear which to use
- NeuralNDCG is in `loss_research.py` but should be in main `loss.py`

**Solution**: 
- **Consolidate into `loss.py`** with clear sections:
  ```python
  # loss.py structure:
  # 1. Basic losses (Huber, Ranking)
  # 2. Research losses (NeuralNDCG, Softmax CE, Focal)
  # 3. Listwise losses (LambdaRank, ApproxNDCG)
  # 4. Multi-objective losses
  # 5. Temporal losses
  ```
- Keep other files as `_deprecated.py` or archive them
- Update all imports to use consolidated `loss.py`

**Priority**: HIGH - This is actively causing confusion

### 2. Model Files: NEEDS ORGANIZATION

**Current State:**
- `model.py` - UniversalICF (main)
- `model_residual.py` - ResidualICF
- `model_hierarchical.py` - HierarchicalICF, BoxEmbeddingICF
- `nano_model.py` - NanoICF

**Problem**: 
- Not terrible, but should be organized
- No clear `__init__.py` exports
- Hard to discover available models

**Solution**: 
- **Option A**: Keep as-is but add clear `__init__.py` exports
- **Option B**: Create `models/` package:
  ```
  models/
    __init__.py  # Exports all models
    universal.py
    residual.py
    hierarchical.py
    nano.py
  ```

**Priority**: MEDIUM - Works but could be cleaner

### 3. Eval Files: FRAGMENTED (CONSOLIDATE!)

**Current State:**
- `eval.py` - Main evaluation
- `eval_advanced.py` - Advanced evaluation
- `eval_rbo.py` - RBO metrics
- `eval_calibration.py` - Calibration (just added)
- `eval_stratified.py` - Stratified (just added)

**Problem**: 
- Just added 2 new files, making fragmentation worse
- Should consolidate into one file
- Hard to know which functions are where

**Solution**: 
- **Consolidate into `eval.py`** with clear sections:
  ```python
  # eval.py structure:
  # 1. Basic metrics (MAE, RMSE, Spearman)
  # 2. Ranking metrics (RBO, precision@k)
  # 3. Calibration metrics (ECE, MCE, Brier)
  # 4. Stratified evaluation (by rarity bins)
  # 5. Advanced evaluation (Jabberwocky, etc.)
  ```
- Move functions from other files into `eval.py`
- Archive old files

**Priority**: HIGH - Just made it worse by adding more files

### 4. Train Files: TOO MANY VARIATIONS (ARCHIVE!)

**Current State:**
- `train.py` - Main training
- `train_curriculum.py` - Curriculum learning
- `train_cv.py` - Cross-validation
- `train_lightning.py` - PyTorch Lightning
- `train_multi_loss.py` - Multi-loss
- `train_optimized.py` - Optimized
- `train_with_eval.py` - With evaluation
- Plus many `train_*.py` scripts in `scripts/`

**Problem**: 
- Too many variations
- Unclear which are current vs experimental
- Hard to know what to use

**Solution**: 
- **Archive old/unused ones** to `archive/training/`
- **Document which are current**:
  - `train_ephemeral_robust.py` - Current (ephemeral pods)
  - `train_residual.py` - Current (residual model)
  - `train_aggressive_regularization.py` - Current
  - `train_temporal_amoo.py` - Current (AMOO)
  - Others → Archive
- Add `CURRENT_TRAINING_SCRIPTS.md` documenting which to use

**Priority**: MEDIUM - Causes confusion but not breaking

### 5. Data Files: SCATTERED (ORGANIZE!)

**Current State:**
- `data.py` - Main data loading
- `data_multilingual.py` - Multilingual
- `data_temporal.py` - Temporal
- `data_universal.py` - Universal

**Problem**: 
- Should be organized
- Unclear relationships

**Solution**: 
- **Option A**: Consolidate into `data.py` with sections
- **Option B**: Create `data/` package:
  ```
  data/
    __init__.py  # Main data loading
    multilingual.py
    temporal.py
    universal.py
  ```

**Priority**: LOW - Works, just could be cleaner

### 6. Augmentation Files: SCATTERED (ORGANIZE!)

**Current State:**
- `augmentation.py` - Main augmentation
- `keyboard_augmentation.py` - Keyboard typos
- `symbol_augmentation.py` - Symbol handling
- `typo_augmentation.py` - Typo patterns

**Problem**: 
- Should be organized
- Unclear which to use

**Solution**: 
- **Create `augmentation/` package**:
  ```
  augmentation/
    __init__.py  # Main AdvancedAugmentation
    keyboard.py
    symbol.py
    typo.py
  ```

**Priority**: LOW - Works, just could be cleaner

### 7. Predict Files: MULTIPLE VERSIONS (CONSOLIDATE!)

**Current State:**
- `predict.py` - Basic prediction
- `predict_enhanced.py` - Enhanced
- `predict_advanced.py` - Advanced

**Problem**: 
- Unclear which to use
- Should be one interface with feature flags

**Solution**: 
- **Consolidate into `predict.py`** with feature flags:
  ```python
  def predict(word, model, enhanced=False, advanced=False):
      if advanced:
          return predict_advanced(word, model)
      elif enhanced:
          return predict_enhanced(word, model)
      else:
          return predict_basic(word, model)
  ```
- Archive old files

**Priority**: MEDIUM - Causes confusion

### 8. Redundant/Unclear Files

**Files to Review:**
- `text_reduction.py`, `text_reduction_real.py`, `text_reduction_isotonic.py`
- Multiple export files (`export_weights.py`, `export_nano_weights.py`)
- Multiple preprocessing files

**Problem**: 
- Unclear purpose
- May be unused

**Solution**: 
- **Audit and archive** if unused
- **Consolidate** if used
- **Document** purpose if keeping

**Priority**: LOW - May not be used

## Recommendations by Priority

### HIGH PRIORITY (Do Now)

1. **Consolidate loss files** → `loss.py` with sections
   - Move NeuralNDCG from `loss_research.py` to `loss.py` (already done)
   - Move other useful losses
   - Archive deprecated files

2. **Consolidate eval files** → `eval.py` with sections
   - Move calibration functions into `eval.py`
   - Move stratified functions into `eval.py`
   - Archive old files

3. **Document current training scripts**
   - Create `CURRENT_TRAINING_SCRIPTS.md`
   - List which scripts are current vs archived

### MEDIUM PRIORITY (Do Soon)

4. **Organize model files** → Better `__init__.py` exports
5. **Consolidate predict files** → One with feature flags
6. **Archive old train files** → Move unused to `archive/`

### LOW PRIORITY (Nice to Have)

7. **Organize data files** → `data/` package
8. **Organize augmentation** → `augmentation/` package
9. **Clean up redundant files** → Archive or delete
10. **Better `__init__.py` exports** → Clear public API

## Proposed Clean Structure

```
src/tiny_icf/
├── __init__.py              # Clear exports of public API
├── model.py                 # UniversalICF (main model)
├── model_residual.py        # ResidualICF
├── model_hierarchical.py   # HierarchicalICF, BoxEmbeddingICF
├── nano_model.py            # NanoICF
├── loss.py                  # ALL losses (consolidated)
│   ├── Basic losses (Huber, Ranking)
│   ├── Research losses (NeuralNDCG, Softmax CE, Focal)
│   ├── Listwise losses (LambdaRank, ApproxNDCG)
│   ├── Multi-objective losses
│   └── Temporal losses
├── data.py                  # Main data loading
├── data_multilingual.py     # Multilingual data
├── data_temporal.py         # Temporal data
├── eval.py                  # ALL evaluation (consolidated)
│   ├── Basic metrics
│   ├── Ranking metrics
│   ├── Calibration metrics
│   ├── Stratified evaluation
│   └── Advanced evaluation
├── augmentation.py          # Main augmentation
├── predict.py               # ONE prediction interface (with flags)
├── training_utils.py        # ✅ Good - unified training
├── initialization.py        # ✅ Good - unified init
├── baselines.py             # ✅ New - baseline comparisons
└── ... (other utilities)
```

## Files That Should Be Archived

### Loss Files (after consolidation)
- `loss_research.py` → Archive (move useful parts to `loss.py`)
- `loss_listwise.py` → Archive (move useful parts to `loss.py`)
- `loss_multi.py` → Archive (move useful parts to `loss.py`)
- `loss_diffsort.py` → Archive (if not used)

### Eval Files (after consolidation)
- `eval_advanced.py` → Archive (move to `eval.py`)
- `eval_rbo.py` → Keep as utility (used by `eval.py`)
- `eval_calibration.py` → Archive (move to `eval.py`)
- `eval_stratified.py` → Archive (move to `eval.py`)

### Train Files (unused)
- `train_curriculum.py` → Archive (if not used)
- `train_cv.py` → Archive (if not used)
- `train_lightning.py` → Archive (if not used)
- `train_multi_loss.py` → Archive (if not used)
- `train_optimized.py` → Archive (if not used)
- `train_with_eval.py` → Archive (if not used)

### Predict Files (after consolidation)
- `predict_enhanced.py` → Archive (move to `predict.py`)
- `predict_advanced.py` → Archive (move to `predict.py`)

## Action Plan

1. **Phase 1: Consolidate Losses** (HIGH)
   - Move NeuralNDCG to `loss.py` ✅ (done)
   - Move other useful losses
   - Archive deprecated files

2. **Phase 2: Consolidate Evaluation** (HIGH)
   - Move calibration to `eval.py`
   - Move stratified to `eval.py`
   - Archive old files

3. **Phase 3: Document Current Scripts** (HIGH)
   - Create `CURRENT_TRAINING_SCRIPTS.md`
   - Archive unused scripts

4. **Phase 4: Organize Models** (MEDIUM)
   - Better `__init__.py` exports
   - Clear documentation

5. **Phase 5: Consolidate Predict** (MEDIUM)
   - One interface with flags
   - Archive old files

## Conclusion

The repository has **good foundations** but suffers from **fragmentation**. The main issues are:

1. **Too many loss files** - Should be one file
2. **Too many eval files** - Should be one file  
3. **Too many train files** - Should archive unused ones
4. **Unclear which files are current** - Need documentation

**The good news**: Most of this is organizational, not functional. The code works, it's just hard to navigate.

**The fix**: Consolidate similar files, archive old ones, document what's current.

