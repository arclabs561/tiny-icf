# Code Refinement and Organization Status

## Summary

Comprehensive analysis and refinement of codebase organization while training runs. Identified key issues and created tools for systematic improvement.

## Issues Identified

### 1. Root Directory Clutter (HIGH PRIORITY) ✅ TOOL CREATED
- **Problem**: 123 markdown files in root directory
- **Impact**: Hard to navigate, unclear what's current
- **Solution**: Created `scripts/organize_root_files.sh` to organize into `docs/` subdirectories
- **Status**: Script ready, needs review before execution

### 2. Loss File Fragmentation (HIGH PRIORITY)
- **Problem**: Multiple loss files:
  - `loss.py` (498 lines) - Main losses
  - `loss_spearman.py` (372 lines) - Spearman loss
  - `loss_adaptive.py` - Adaptive weighting
  - `loss_monitoring.py` - Monitoring utilities
  - `temporal_loss.py` - Temporal losses
- **Total**: ~1,642 lines across 5 files
- **Impact**: Hard to find loss functions, unclear dependencies
- **Solution**: Consolidate into `loss.py` with clear sections
- **Status**: Needs consolidation

### 3. Eval File Fragmentation (HIGH PRIORITY)
- **Problem**: 7 eval files:
  - `eval.py` - Main evaluation
  - `eval_advanced.py` - Advanced evaluation
  - `eval_calibration.py` - Calibration metrics
  - `eval_rbo.py` - RBO metrics
  - `eval_stratified.py` - Stratified evaluation
  - `eval_robustness.py` - Robustness evaluation
  - `eval_uncertainty.py` - Uncertainty evaluation
- **Impact**: Hard to find evaluation functions, duplicate code
- **Solution**: Consolidate into `eval.py` with clear sections
- **Status**: Needs consolidation

### 4. Training Script Proliferation (MEDIUM PRIORITY)
- **Problem**: 50+ training scripts in `scripts/`
- **Impact**: Unclear which are current vs experimental
- **Solution**: Archive unused scripts, document current ones
- **Status**: Needs review and archiving

### 5. Duplicate Functions Found
- **Calibration metrics**: `expected_calibration_error`, `maximum_calibration_error`, `brier_score` in both `eval_calibration.py` and `eval.py`
- **Stratified evaluation**: `stratified_evaluation`, `evaluate_by_rarity_category` in both `eval_stratified.py` and `eval.py`
- **Text reduction**: Multiple `predict_icf`, `compute_embedding` functions
- **Impact**: Code duplication, maintenance burden
- **Solution**: Consolidate duplicates during file consolidation

## Tools Created

### 1. `scripts/organize_root_files.sh`
- Organizes 123 root markdown files into `docs/` subdirectories
- Categories: status, design, guides, results
- Safe: Creates archive directory, preserves files

### 2. `scripts/refine_code_organization.py`
- Analyzes codebase for duplicate functions/classes
- Identifies consolidation opportunities
- Reports import patterns

### 3. `docs/REFINEMENT_PLAN.md`
- Documents refinement strategy
- Prioritizes actions
- Tracks progress

## Best Practices Research

### PyTorch Lightning Best Practices Applied ✅
- Using Lightning for all training scenarios
- Proper callback organization (ModelCheckpoint, EarlyStopping, LearningRateMonitor)
- Automatic checkpointing and logging
- CosineAnnealingLR scheduler
- Mixed precision training (16-mixed)

### Code Organization Best Practices
- ✅ PEP 723 inline dependencies
- ✅ Unified training utilities (`training_utils.py`)
- ✅ Centralized initialization (`initialization.py`)
- ✅ Clear `src/` module structure
- ⏳ Better module exports (`__init__.py`)
- ⏳ Consolidated loss/eval files

### Project Structure Best Practices
- ✅ Clear separation: `src/`, `scripts/`, `docs/`, `data/`, `models/`
- ✅ Organized documentation in `docs/` subdirectories
- ⏳ Clean root directory (123 files → organized)

## Next Steps

### Phase 1: File Organization (IMMEDIATE)
1. ⏳ Review `scripts/organize_root_files.sh`
2. ⏳ Run organization script (creates archive, moves files)
3. ⏳ Update any broken references

### Phase 2: Code Consolidation (NEXT)
1. ⏳ Consolidate `loss_spearman.py` into `loss.py`
   - Move `SpearmanLoss` class and helper functions
   - Update imports in `loss.py` and `flexible_lightning_module.py`
   - Archive `loss_spearman.py`
2. ⏳ Consolidate eval files into `eval.py`
   - Move calibration functions (remove duplicates)
   - Move stratified functions (remove duplicates)
   - Move RBO, robustness, uncertainty functions
   - Archive old eval files
3. ⏳ Update all imports across codebase

### Phase 3: Script Documentation (ONGOING)
1. ⏳ Document current training scripts
2. ⏳ Archive unused/experimental scripts
3. ⏳ Create `CURRENT_TRAINING_SCRIPTS.md`

### Phase 4: Module Exports (MEDIUM)
1. ⏳ Improve `__init__.py` exports
2. ⏳ Add clear documentation strings
3. ⏳ Make modules more discoverable

## Current Training Status

- **Active**: Training running locally (CPU)
- **Experiments**: residual_optimal, residual_wide, residual_deep, residual_balanced
- **Progress**: Epoch 40/200 (~20% complete for first experiment)
- **Best Spearman**: 0.1785 (epochs 17, 24)
- **Refinements**: Enhanced loss logging, improved Lightning integration

## Metrics

- **Root markdown files**: 123 → Target: <10
- **Loss files**: 5 → Target: 1
- **Eval files**: 7 → Target: 1
- **Training scripts**: 50+ → Target: Document current, archive rest
- **Code duplication**: Found multiple duplicates → Target: Eliminate

## Notes

- Refinements are being applied while training runs (non-breaking)
- File organization can be done safely (creates archive)
- Code consolidation requires careful import updates
- All changes preserve functionality (organizational only)

