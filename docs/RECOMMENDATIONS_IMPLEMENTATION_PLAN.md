# Recommendations Implementation Plan: All Tasks

## Overview

This document outlines the implementation plan for all recommendations, considering ALL tasks (ICF, text reduction, temporal, language, era, multi-task), not just ICF prediction.

## Current Status

### Completed
- ✅ Fundamental questions documented (`FUNDAMENTAL_QUESTIONS.md`)
- ✅ Information-theoretic constraints analyzed (`INFORMATION_THEORETIC_CONSTRAINTS.md`)
- ✅ All tasks structure analysis (`ALL_TASKS_STRUCTURE_ANALYSIS.md`)
- ✅ Unified loss framework created (`loss_unified.py`)
- ✅ Multi-task structure analysis script (`multi_task_structure_analysis.py`)

### In Progress
- ⏳ Structure analysis on actual data
- ⏳ Unified loss integration into training pipeline

### Pending
- ⏳ Model compression (quantization/pruning)
- ⏳ Hybrid system (dict + model)
- ⏳ OOV generalization testing
- ⏳ Multi-task data loading
- ⏳ Use case clarification

## Implementation Plan

### Phase 1: Structure Validation (Immediate)

**Goal**: Determine if structure exists and is strong enough for compression.

**Tasks**:
1. Run `multi_task_structure_analysis.py` on actual data
2. Measure structure strength for each task
3. Validate generalization (train vs OOV)
4. Decide: compression priority vs generalization priority

**Files**:
- `src/tiny_icf/multi_task_structure_analysis.py` (created)
- `src/tiny_icf/structure_analysis.py` (created)

**Next Steps**:
- Fix README.md (done)
- Run analysis with proper environment
- Document findings

### Phase 2: Unified Loss Integration (High Priority)

**Goal**: Integrate `UnifiedMultiTaskLoss` into training pipeline.

**Current State**:
- `flexible_lightning_module.py` uses `CombinedLoss` (ICF only)
- `loss_unified.py` has `UnifiedMultiTaskLoss` (all tasks)

**Tasks**:
1. Add option to use `UnifiedMultiTaskLoss` in `flexible_lightning_module.py`
2. Support multi-task data loading (ICF + language + temporal + era)
3. Configure AMOO for adaptive task weighting
4. Test with ICF-only first (backward compatible)
5. Gradually add other tasks

**Files to Modify**:
- `src/tiny_icf/flexible_lightning_module.py`
- `src/tiny_icf/lightning_data.py` (add multi-task data loading)
- `../trainctl/training/scripts/train_flexible_opportunistic.py` (add multi-task configs)

**Implementation**:
```python
# In flexible_lightning_module.py
if config.get('use_unified_loss', False):
    from tiny_icf.loss_unified import UnifiedMultiTaskLoss
    self.criterion = UnifiedMultiTaskLoss(
        icf_weight=config.get('icf_weight', 1.0),
        text_reduction_weight=config.get('text_reduction_weight', 0.5),
        temporal_weight=config.get('temporal_weight', 0.3),
        language_weight=config.get('language_weight', 0.2),
        era_weight=config.get('era_weight', 0.2),
        use_amoo=config.get('use_amoo', True),
    )
else:
    # Existing CombinedLoss (backward compatible)
    self.criterion = CombinedLoss(...)
```

### Phase 3: Multi-Task Data Loading (High Priority)

**Goal**: Load data for all tasks (ICF, language, temporal, era).

**Tasks**:
1. Extend `IDFDataModule` to support multi-task data
2. Load language labels (from patterns or external data)
3. Load temporal ICF (from historical data or simulate)
4. Load era labels (from patterns or external data)
5. Handle missing data gracefully (tasks are optional)

**Files to Modify**:
- `src/tiny_icf/lightning_data.py`
- `src/tiny_icf/data.py` (add multi-task data loading utilities)

**Implementation**:
```python
# In lightning_data.py
class MultiTaskIDFDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        language_data_path: Optional[Path] = None,
        temporal_data_path: Optional[Path] = None,
        era_data_path: Optional[Path] = None,
        ...
    ):
        # Load all task data
        # Handle missing data (tasks are optional)
```

### Phase 4: Model Architecture for Multi-Task (Medium Priority)

**Goal**: Extend model to output all task predictions.

**Current State**:
- Models (`UniversalICF`, `ResidualICF`) output only ICF
- Need: ICF + language + temporal + era outputs

**Tasks**:
1. Add task-specific heads to models
2. Share base architecture (character CNN)
3. Support single-task and multi-task modes
4. Test backward compatibility (ICF-only still works)

**Files to Modify**:
- `src/tiny_icf/model.py` (add multi-task heads)
- `src/tiny_icf/model_residual.py` (add multi-task heads)

**Implementation**:
```python
# In model.py
class UniversalICF(nn.Module):
    def __init__(
        self,
        output_tasks: List[str] = ['icf'],  # ['icf', 'language', 'temporal', 'era']
        ...
    ):
        # Base CNN (shared)
        self.base = ...
        
        # Task-specific heads
        if 'icf' in output_tasks:
            self.icf_head = nn.Linear(...)
        if 'language' in output_tasks:
            self.language_head = nn.Linear(..., num_languages)
        if 'temporal' in output_tasks:
            self.temporal_head = nn.Linear(..., num_decades)
        if 'era' in output_tasks:
            self.era_head = nn.Linear(..., num_eras)
```

### Phase 5: Model Compression (Medium Priority)

**Goal**: Reduce model size to 20-40 KB (beat sparse dictionaries).

**Tasks**:
1. Implement quantization (float32 → int8)
2. Implement pruning (remove low-magnitude weights)
3. Validate accuracy after compression
4. Compare: compressed model vs sparse dict

**Files to Create**:
- `src/tiny_icf/compression.py` (quantization, pruning utilities)

**Implementation**:
```python
# quantization
def quantize_model(model, bits=8):
    # Convert float32 → int8
    # Validate accuracy
    pass

# pruning
def prune_model(model, sparsity=0.5):
    # Remove low-magnitude weights
    # Retrain if needed
    pass
```

### Phase 6: Hybrid System (Low Priority)

**Goal**: Dictionary for seen words + model for OOV.

**Tasks**:
1. Build sparse dictionary (rare words only, ~90 KB)
2. Implement OOV detection
3. Implement hybrid lookup (dict → model fallback)
4. Measure: speed, accuracy, storage

**Files to Create**:
- `src/tiny_icf/hybrid_lookup.py`

### Phase 7: Generalization Testing (High Priority)

**Goal**: Validate that model learns structure (not memorization).

**Tasks**:
1. Create OOV test set (unseen words)
2. Measure: train accuracy vs OOV accuracy
3. Analyze: what patterns does model learn?
4. Test: can model generalize to new languages/eras?

**Files to Create**:
- `src/tiny_icf/test_generalization.py`

## Priority Ranking

### High Priority (Do First)
1. **Structure validation**: Know if compression is feasible
2. **Unified loss integration**: Enable multi-task learning
3. **Generalization testing**: Validate model learns structure

### Medium Priority (Do Next)
4. **Multi-task data loading**: Support all tasks
5. **Model architecture extension**: Output all task predictions
6. **Model compression**: Beat sparse dictionaries

### Low Priority (Do Later)
7. **Hybrid system**: Best of both worlds
8. **Use case clarification**: Document when to use what

## Next Immediate Actions

1. ✅ Fix README.md (done)
2. ⏳ Run structure analysis on actual data
3. ⏳ Integrate unified loss into `flexible_lightning_module.py`
4. ⏳ Add multi-task data loading support
5. ⏳ Test with ICF-only first (backward compatible)
6. ⏳ Gradually add other tasks

## Success Criteria

### Structure Validation
- [ ] Structure strength measured for all tasks
- [ ] Generalization validated (train vs OOV)
- [ ] Decision made: compression vs generalization priority

### Unified Loss Integration
- [ ] `UnifiedMultiTaskLoss` integrated into training
- [ ] Backward compatible (ICF-only still works)
- [ ] AMOO working (adaptive task weighting)

### Multi-Task Learning
- [ ] All tasks can be trained together
- [ ] Unified model is smaller than separate models
- [ ] All tasks show improvement vs single-task

### Model Compression
- [ ] Model compressed to 20-40 KB
- [ ] Accuracy maintained (>90% of original)
- [ ] Beats sparse dictionaries (90 KB)

### Generalization
- [ ] OOV accuracy > 0.5 × train accuracy
- [ ] Model learns patterns (not just memorizes)
- [ ] Generalizes to new languages/eras

