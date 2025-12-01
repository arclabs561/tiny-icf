# Better Alternatives to Nearest-Word Mapping

## Your Idea (Nearest-Word Mapping)

1. Extract words from Common Crawl
2. Map to nearest words we have frequencies for
3. Union all languages
4. Random sample to impute as augmented data

## Why This Is Problematic

### Core Issue: Frequency ≠ Similarity

**Edit distance or embedding similarity does NOT correlate with word frequency.**

Examples:
- `"cat"` (edit distance 1 from `"bat"`) - very different frequencies
- `"the"` (most common) vs `"thy"` (rare, archaic) - similar form, opposite frequency
- `"computer"` vs `"computers"` - similar, but frequencies differ
- `"run"` vs `"fun"` - edit distance 1, but "run" is much more common

**Result**: We'd train the model with wrong frequency labels, teaching it incorrect patterns.

### Language Mixing Creates Noise

Union all languages means:
- English `"the"` might map to Spanish `"te"` (edit distance 1)
- But `"the"` (ICF ~0.0) vs `"te"` (ICF ~0.5) are very different
- Model learns: "words that look similar have similar frequency" (WRONG)

### Augmentation Purpose Mismatch

**Goal**: Make model robust to typos/misspellings.

**What We Want**:
- `"cmputer"` (typo) → should predict similar to `"computer"` (correct)
- Model learns: "typos of common words are still common"

**What Nearest-Word Mapping Does**:
- `"cmputer"` → maps to `"computer"` → uses `"computer"` frequency ✅
- BUT: `"xylophone"` (rare) → maps to `"telephone"` (common) → WRONG frequency ❌
- Model learns: "rare words that look like common words are common" (WRONG)

## Better Alternatives (Implemented)

### ✅ Option 1: Expand Frequency List Directly

**Instead of mapping**, extract word frequencies directly from Common Crawl:

```bash
# Merge frequency lists (no mapping)
python scripts/expand_frequency_list.py \
    --existing data/combined_frequencies.csv \
    --new common_crawl_frequencies.csv \
    --output expanded_frequencies.csv
```

**Benefits**:
- Real frequencies (not synthetic)
- More training data
- No frequency mismatch

### ✅ Option 2: Find Real Typos from Frequency Data

**Use frequency data itself to find typos**:

```bash
# Find low-frequency words close to high-frequency words
python scripts/find_real_typos_from_common_crawl.py \
    --frequency-list data/combined_frequencies.csv \
    --output data/typos/real_typos.csv
```

**Logic**:
- Low-frequency word (e.g., count=5) close to high-frequency word (e.g., count=1000)
- Likely a typo of the high-frequency word
- Use correction's frequency (correct!)

**Benefits**:
- Real typo data (not synthetic)
- Correct frequency labels
- Better augmentation

### ✅ Option 3: Use Real Typo Corpus (Already Done)

**We already downloaded GitHub Typo Corpus**:
- 82,347 real typo-correction pairs
- Real pattern frequencies analyzed
- Use for augmentation with correction's frequency

**Usage**:
```python
# Load real typos
typo_pairs = load_typo_corpus("data/typos/github_typos.csv")

# Augment: use typo form, but correction's frequency
for typo, correction in typo_pairs:
    # Train on (typo, correction_frequency)
    # Model learns: typos have similar frequency to corrections
```

### ✅ Option 4: OOV as Test Cases (Not Training)

**Use Common Crawl words for validation, not training**:

```python
# Extract OOV words from Common Crawl
# Don't train on them (no frequency labels)
# Use for zero-shot evaluation:
#   - Can model predict reasonable frequencies?
#   - Does it generalize to unseen words?
```

**Benefits**:
- No wrong labels
- Better evaluation
- Tests true generalization

## Research Validation

From MCP research (Perplexity):
- **Zipf's Law**: Word frequency follows long-tail distribution
- **Frequency ≠ Similarity**: Edit distance doesn't correlate with frequency
- **Sparse Signal**: Rare words have unreliable neighbors
- **Better Approach**: Use real frequencies, not synthetic mappings

## Recommended Workflow

### Step 1: Expand Frequency List
```bash
# Download larger frequency lists
# Merge with existing (no mapping)
python scripts/expand_frequency_list.py \
    --existing data/combined_frequencies.csv \
    --new common_crawl_freqs.csv \
    --output expanded.csv
```

### Step 2: Use Real Typos
```bash
# Already have: data/typos/github_typos.csv (82k pairs)
# Use for augmentation with correction's frequency
```

### Step 3: Train with Real Data
```bash
python -m tiny_icf.train_curriculum \
    --data expanded.csv \
    --typo-corpus data/typos/github_typos.csv \
    --augment-prob 0.2
```

### Step 4: Evaluate on OOV
```bash
# Extract OOV words from Common Crawl
# Test model generalization (don't train on them)
```

## Summary

**Nearest-word mapping is problematic** because:
- ❌ Frequency ≠ similarity
- ❌ Creates wrong labels
- ❌ Doesn't solve the real problem
- ❌ Language mixing adds noise

**Better alternatives**:
- ✅ Expand frequency list directly (real frequencies)
- ✅ Use real typo corpus (already downloaded)
- ✅ Find typos in frequency data itself
- ✅ Use OOV for evaluation, not training

**Status**: Better alternatives implemented and ready to use.

