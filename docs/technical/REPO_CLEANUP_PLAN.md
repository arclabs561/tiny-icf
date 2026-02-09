# Repository Cleanup Plan - Opinionated Recommendations

## Executive Summary

After reviewing the repository structure with `eza -T`, here's what **doesn't make sense** and what should be **consolidated or archived**.

## Critical Issues (Fix Now)

### 1. Eval Files: TOO FRAGMENTED ❌

**Current State:**
- `eval.py` - Main (now has consolidated functions)
- `eval_advanced.py` - Advanced evaluation
- `eval_rbo.py` - RBO metrics (utility, keep)
- `eval_calibration.py` - **ARCHIVE** (consolidated into eval.py)
- `eval_stratified.py` - **ARCHIVE** (consolidated into eval.py)

**Action**: 
- ✅ Already consolidated calibration/stratified into `eval.py`
- Archive `eval_calibration.py` and `eval_stratified.py`
- Review `eval_advanced.py` - move useful parts to `eval.py` or archive

### 2. Loss Files: SCATTERED ❌

**Current State:**
- `loss.py` - Main (now has NeuralNDCG)
- `loss_research.py` - Research losses
- `loss_listwise.py` - Listwise losses
- `loss_multi.py` - Multi-objective
- `loss_diffsort.py` - DiffSort
- `temporal_loss.py` - Temporal losses

**Problem**: 6 files for losses, hard to find what you need

**Action**: 
- Keep `loss.py` as main (has Huber, Ranking, NeuralNDCG)
- Keep `temporal_loss.py` (used by temporal_amoo, specialized)
- **Archive others** or move useful parts to `loss.py`:
  - `loss_research.py` → Archive (NeuralNDCG already in loss.py)
  - `loss_listwise.py` → Review, archive if not used
  - `loss_multi.py` → Review, archive if not used
  - `loss_diffsort.py` → Archive if not used

### 3. Train Files in src/: DEPRECATED ❌

**Current State:**
- `train.py` - Old main training
- `train_curriculum.py` - Curriculum
- `train_cv.py` - Cross-validation
- `train_lightning.py` - PyTorch Lightning
- `train_multi_loss.py` - Multi-loss
- `train_optimized.py` - Optimized
- `train_with_eval.py` - With evaluation

**Problem**: These are in `src/tiny_icf/` but we use `scripts/train_*.py` now

**Action**: **Archive all of these** to `archive/training/`

### 4. Predict Files: MULTIPLE VERSIONS ❌

**Current State:**
- `predict.py` - Basic
- `predict_enhanced.py` - Enhanced
- `predict_advanced.py` - Advanced

**Problem**: Unclear which to use

**Action**: 
- Consolidate into `predict.py` with feature flags
- Archive `predict_enhanced.py` and `predict_advanced.py`

### 5. Data Files: SCATTERED (Medium Priority)

**Current State:**
- `data.py` - Main
- `data_multilingual.py` - Multilingual
- `data_temporal.py` - Temporal
- `data_universal.py` - Universal

**Status**: Actually OK - these serve different purposes
**Action**: Keep as-is, but document relationships

### 6. Augmentation Files: SCATTERED (Medium Priority)

**Current State:**
- `augmentation.py` - Main
- `keyboard_augmentation.py` - Keyboard typos
- `symbol_augmentation.py` - Symbol handling
- `typo_augmentation.py` - Typo patterns

**Status**: Could be organized better
**Action**: Consider `augmentation/` package, but low priority

## What Makes Sense ✅

### Good Structure
- `training_utils.py` - ✅ Unified training logic
- `initialization.py` - ✅ Unified initialization
- `baselines.py` - ✅ New, well-organized
- `model.py`, `model_residual.py`, etc. - ✅ Clear model variants
- `scripts/` organization - ✅ Training scripts in one place

### Good Patterns
- PEP 723 inline dependencies - ✅ Excellent
- Clear module structure - ✅ Good
- Documentation organization - ✅ Good (after cleanup)

## Consolidation Plan

### Phase 1: Archive Deprecated (Do Now)

```bash
# Create archive structure
mkdir -p archive/$(date +%Y%m%d)/{eval_files,loss_files,train_files,predict_files}

# Archive eval files (after consolidation)
mv src/tiny_icf/eval_calibration.py archive/.../eval_files/
mv src/tiny_icf/eval_stratified.py archive/.../eval_files/

# Archive old train files from src/
mv src/tiny_icf/train.py archive/.../train_files/
mv src/tiny_icf/train_curriculum.py archive/.../train_files/
mv src/tiny_icf/train_cv.py archive/.../train_files/
mv src/tiny_icf/train_lightning.py archive/.../train_files/
mv src/tiny_icf/train_multi_loss.py archive/.../train_files/
mv src/tiny_icf/train_optimized.py archive/.../train_files/
mv src/tiny_icf/train_with_eval.py archive/.../train_files/

# Archive predict files (after consolidation)
mv src/tiny_icf/predict_enhanced.py archive/.../predict_files/
mv src/tiny_icf/predict_advanced.py archive/.../predict_files/
```

### Phase 2: Review and Consolidate Losses (Do Soon)

1. Review `loss_research.py`, `loss_listwise.py`, `loss_multi.py`, `loss_diffsort.py`
2. Move useful functions to `loss.py`
3. Archive unused files

### Phase 3: Consolidate Predict (Do Soon)

1. Merge `predict_enhanced.py` and `predict_advanced.py` into `predict.py`
2. Use feature flags for advanced features
3. Archive old files

## Files That Should Definitely Be Archived

### High Confidence (Archive Now)
- `src/tiny_icf/eval_calibration.py` - Consolidated
- `src/tiny_icf/eval_stratified.py` - Consolidated
- `src/tiny_icf/train.py` - Use scripts/ versions
- `src/tiny_icf/train_curriculum.py` - Experimental
- `src/tiny_icf/train_cv.py` - Experimental
- `src/tiny_icf/train_lightning.py` - Experimental
- `src/tiny_icf/train_multi_loss.py` - Experimental
- `src/tiny_icf/train_optimized.py` - Experimental
- `src/tiny_icf/train_with_eval.py` - Experimental
- `src/tiny_icf/predict_enhanced.py` - Should consolidate
- `src/tiny_icf/predict_advanced.py` - Should consolidate

### Medium Confidence (Review First)
- `src/tiny_icf/loss_research.py` - NeuralNDCG already in loss.py
- `src/tiny_icf/loss_listwise.py` - Review if used
- `src/tiny_icf/loss_multi.py` - Review if used
- `src/tiny_icf/loss_diffsort.py` - Review if used
- `src/tiny_icf/eval_advanced.py` - Review if used

## Proposed Clean Structure

```
src/tiny_icf/
├── __init__.py              # Clear exports
├── model.py                 # UniversalICF
├── model_residual.py        # ResidualICF
├── model_hierarchical.py    # HierarchicalICF, BoxEmbeddingICF
├── nano_model.py            # NanoICF
├── loss.py                  # ALL losses (Huber, Ranking, NeuralNDCG, Temporal)
├── data.py                  # Main data loading
├── data_multilingual.py     # Multilingual (keep)
├── data_temporal.py         # Temporal (keep)
├── eval.py                  # ALL evaluation (consolidated)
├── eval_rbo.py              # RBO utility (keep)
├── augmentation.py          # Main augmentation
├── predict.py               # ONE prediction interface
├── baselines.py             # ✅ New
├── training_utils.py        # ✅ Good
├── initialization.py        # ✅ Good
└── temporal_loss.py         # Temporal losses (specialized, keep)
```

## Scripts Organization

**Current scripts/ has 150+ files** - This is actually OK for a research repo, but:

### Good Organization ✅
- `train_*.py` - Clear training scripts
- `evaluate_*.py` - Clear evaluation scripts
- `monitor_*.sh` - Clear monitoring scripts
- `run_*.py` - Clear runner scripts

### Could Be Better
- Many `runpod_*.py` files - Could be `runpod/` package
- Many `download_*.py` files - Could be `data/download/` package
- Many `auto_start_*.py` files - Could be consolidated

**But**: For a research repo, this is acceptable. The main issue is `src/tiny_icf/` fragmentation.

## Action Items

### Immediate (Do Now)
1. ✅ Archive `eval_calibration.py` and `eval_stratified.py`
2. Archive old `train*.py` files from `src/tiny_icf/`
3. Document which files are current

### Short-term (Do Soon)
4. Review and consolidate loss files
5. Consolidate predict files
6. Review `eval_advanced.py`

### Long-term (Nice to Have)
7. Organize scripts/ into subdirectories
8. Better `__init__.py` exports
9. Create packages for related files

## Conclusion

**Main Problem**: `src/tiny_icf/` has too many fragmented files doing similar things.

**Main Solution**: Consolidate similar files, archive deprecated ones.

**Priority**: Fix `src/tiny_icf/` fragmentation first, then consider `scripts/` organization.

The repository structure is **good overall**, but **needs consolidation** to reduce confusion.

