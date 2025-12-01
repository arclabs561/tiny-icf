# Implementation: Best Practice Approach (MCP Research-Based)

## Research Findings

From MCP deep research:
1. **Real typo corpora are highly effective** - "real typo augmentation based on empirically derived error models provides substantial benefits"
2. **Direct frequency expansion** - "vocabulary expansion using bilingual dictionaries" works, merge lists directly
3. **Hybrid approach** - Combine multiple strategies (subword + real augmentation + frequency expansion)
4. **Quality over quantity** - Smaller high-quality augmented data > large unvalidated data

## Implementation

### ✅ Step 1: Real Typo Corpus Integration

**File**: `src/tiny_icf/typo_augmentation.py`

**Features**:
- Loads real typo-correction pairs from GitHub corpus (82k pairs)
- Uses correction's frequency (correct approach!)
- Falls back to keyboard-aware augmentation
- Integrates with training pipeline

**Usage**:
```python
from tiny_icf.typo_augmentation import RealTypoAugmentation

aug = RealTypoAugmentation(
    typo_corpus_path=Path("data/typos/github_typos.csv"),
    use_prob=0.2,  # 20% real typos, 80% keyboard-aware
)
```

### ✅ Step 2: Typo-Aware Dataset

**File**: `src/tiny_icf/typo_augmentation.py`

**Key Insight**: When augmenting with typos, use correction's frequency (not mapped frequency).

**Example**:
- Word: `"computer"` (ICF=0.3)
- Typo: `"cmputer"` (from corpus)
- Training: Use `("cmputer", 0.3)` - correction's frequency ✅
- NOT: Map `"cmputer"` to nearest word ❌

### ✅ Step 3: Updated Training Script

**File**: `src/tiny_icf/train_curriculum.py`

**New Features**:
- `--typo-corpus`: Path to real typo corpus
- `--multilingual`: Use per-language ICF computation
- Automatically uses `TypoAwareDataset` if typo corpus provided

**Usage**:
```bash
python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --typo-corpus data/typos/github_typos.csv \
    --multilingual \
    --epochs 50 \
    --curriculum-stages 5 \
    --augment-prob 0.2
```

## Why This is Best Practice

### 1. Real Typo Data (Not Synthetic)
- **82k real typo pairs** from GitHub corpus
- **Empirically derived patterns** (char_drop: 44%, char_insert: 39%)
- **Research-backed**: "provides substantial benefits"

### 2. Correct Frequency Assignment
- **Use correction's frequency** (not mapped)
- **Model learns**: "typos have similar frequency to corrections"
- **Avoids**: Wrong frequency labels from mapping

### 3. Hybrid Approach
- **Real typos** (20%): High-quality, realistic
- **Keyboard-aware** (80%): Covers patterns not in corpus
- **Combined**: Best of both worlds

### 4. Multilingual Support
- **Per-language ICF**: Correct scores for each language
- **Language-balanced sampling**: Fair representation
- **No cross-language contamination**

## Expected Improvements

Based on research:
- **Better typo handling**: Real patterns > synthetic
- **Correct frequency learning**: Correction's frequency > mapped
- **Improved generalization**: Hybrid approach > single method
- **Multilingual robustness**: Per-language ICF > global

## Next Steps

1. **Train with real typo corpus**:
   ```bash
   python -m tiny_icf.train_curriculum \
       --data data/multilingual/multilingual_combined.csv \
       --typo-corpus data/typos/github_typos.csv \
       --multilingual \
       --epochs 50
   ```

2. **Compare to baseline**:
   - Train without typo corpus
   - Train with typo corpus
   - Measure improvement

3. **Validate on OOV words**:
   - Extract OOV from Common Crawl
   - Test model generalization
   - Measure zero-shot performance

## Files Created

- `src/tiny_icf/typo_augmentation.py`: Real typo corpus integration
- `IMPLEMENTATION_BEST_PRACTICE.md`: This file

## Status

✅ **Implemented**: Real typo corpus integration
✅ **Implemented**: Typo-aware dataset with correct frequencies
✅ **Implemented**: Updated training script
✅ **Ready**: Train with best practice approach

