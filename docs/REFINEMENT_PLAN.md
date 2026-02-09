# Code Refinement and Organization Plan

## Current Issues Identified

### 1. Root Directory Clutter (HIGH PRIORITY)
- **Problem**: 50+ markdown files in root directory
- **Impact**: Hard to navigate, unclear what's current
- **Solution**: Organize into `docs/` subdirectories
- **Status**: Script created (`scripts/organize_root_files.sh`)

### 2. Loss File Fragmentation (HIGH PRIORITY)
- **Problem**: Multiple loss files (`loss.py`, `loss_spearman.py`, `loss_adaptive.py`, `loss_monitoring.py`, `temporal_loss.py`)
- **Impact**: Hard to find loss functions, unclear dependencies
- **Solution**: Consolidate into `loss.py` with clear sections
- **Status**: Needs review

### 3. Eval File Fragmentation (HIGH PRIORITY)
- **Problem**: Multiple eval files (`eval.py`, `eval_advanced.py`, `eval_calibration.py`, `eval_rbo.py`, `eval_stratified.py`, `eval_robustness.py`, `eval_uncertainty.py`)
- **Impact**: Hard to find evaluation functions
- **Solution**: Consolidate into `eval.py` with clear sections
- **Status**: Needs review

### 4. Training Script Proliferation (MEDIUM PRIORITY)
- **Problem**: 50+ training scripts in `scripts/`
- **Impact**: Unclear which are current vs experimental
- **Solution**: Archive unused scripts, document current ones
- **Status**: Needs review

### 5. Import Organization (MEDIUM PRIORITY)
- **Problem**: Inconsistent imports, unclear dependencies
- **Solution**: Better `__init__.py` exports, clear module structure
- **Status**: Needs review

## Refinement Actions

### Phase 1: File Organization (IMMEDIATE)
1. ✅ Create organization script
2. ⏳ Run organization script (after review)
3. ⏳ Update any broken references

### Phase 2: Code Consolidation (NEXT)
1. ⏳ Review loss files and consolidate
2. ⏳ Review eval files and consolidate
3. ⏳ Update all imports

### Phase 3: Documentation (ONGOING)
1. ⏳ Document current training scripts
2. ⏳ Create clear README structure
3. ⏳ Archive old documentation

## Best Practices Applied

### PyTorch Lightning
- ✅ Using Lightning for all training
- ✅ Proper callback organization
- ✅ Automatic checkpointing
- ✅ Learning rate monitoring

### Code Organization
- ✅ PEP 723 inline dependencies
- ✅ Unified training utilities
- ✅ Centralized initialization
- ⏳ Better module exports

### Project Structure
- ✅ Clear `src/` module structure
- ✅ Organized `scripts/` directory
- ✅ Documentation in `docs/`
- ⏳ Clean root directory

