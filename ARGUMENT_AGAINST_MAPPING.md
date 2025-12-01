# Argument Against Nearest-Word Mapping for Augmentation

## Your Proposal

1. Extract words from Common Crawl
2. Map to nearest words we have frequencies for (edit distance/embedding)
3. Union all languages
4. Random sample to impute as augmented data

## Problems with This Approach

### 1. ❌ Frequency ≠ Similarity

**Core Issue**: Edit distance or embedding similarity does NOT correlate with frequency.

**Examples**:
- `"cat"` (edit distance 1 from `"bat"`) - very different frequencies
- `"the"` (most common) vs `"thy"` (rare, archaic) - similar form, opposite frequency
- `"computer"` vs `"computers"` - similar, but plural might be less common
- `"run"` vs `"fun"` - edit distance 1, but "run" is much more common

**Result**: We'd train the model with wrong frequency labels, teaching it incorrect patterns.

### 2. ❌ Language Mixing Creates Noise

**Problem**: Union all languages means:
- English `"the"` might map to Spanish `"te"` (edit distance 1)
- But `"the"` (ICF ~0.0) vs `"te"` (ICF ~0.5) are very different
- Model learns: "words that look similar have similar frequency" (WRONG)

**Better**: Keep languages separate, or use language-aware mapping.

### 3. ❌ Augmentation Purpose Mismatch

**Goal of Augmentation**: Make model robust to typos/misspellings.

**What We Want**:
- `"cmputer"` (typo) → should predict similar to `"computer"` (correct)
- Model learns: "typos of common words are still common"

**What Nearest-Word Mapping Does**:
- `"cmputer"` → maps to `"computer"` → uses `"computer"` frequency ✅
- BUT: `"xylophone"` (rare) → maps to `"telephone"` (common) → WRONG frequency ❌
- Model learns: "rare words that look like common words are common" (WRONG)

### 4. ❌ Data Leakage / Synthetic Labels

**Problem**: We're creating synthetic frequency labels by mapping.

**Example**:
- Common Crawl word: `"biotechnology"` (not in our frequency list)
- Nearest word: `"technology"` (in our list, ICF=0.3)
- We assign: `"biotechnology"` → ICF=0.3
- **Reality**: `"biotechnology"` is much rarer (ICF ~0.8)

**Result**: Model learns wrong frequency patterns.

### 5. ❌ Doesn't Solve the Real Problem

**Real Problem**: We need more training data with correct frequencies.

**Your Solution**: Creates more data with potentially wrong frequencies.

**Better Solution**: 
- Expand frequency list (download larger corpus)
- Use Common Crawl to find MORE words with frequencies
- Don't map - just add words that have real frequency data

## Better Alternatives

### Option 1: Expand Frequency List from Common Crawl ✅

**Instead of mapping**, extract word frequencies directly:

```python
# Extract word counts from Common Crawl
# Use words that appear frequently enough to have reliable counts
# Add to frequency list (no mapping needed)
```

**Benefits**:
- Real frequencies (not synthetic)
- More training data
- No frequency mismatch

**Implementation**:
- Download Common Crawl word counts (already available)
- Merge with existing frequency list
- Use stratified sampling as before

### Option 2: Common Crawl for Real Typo Discovery ✅

**Use Common Crawl to find real typos**:

```python
# Find words in Common Crawl that:
# 1. Are close to words in our frequency list (edit distance 1-2)
# 2. Have low frequency (likely typos)
# 3. Map to high-frequency words (typo → correction)
# Result: Real typo-correction pairs with frequencies
```

**Benefits**:
- Real typo data (not synthetic)
- Correct frequency labels (use correction's frequency)
- Better augmentation

### Option 3: OOV as Test Cases (Not Training) ✅

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

### Option 4: Language-Aware Expansion ✅

**If we must map, do it language-aware**:

```python
# For each language:
#   1. Extract Common Crawl words in that language
#   2. Map only within same language
#   3. Use morphological similarity (not just edit distance)
#   4. Weight by frequency similarity (not just form similarity)
```

**Still problematic**, but better than cross-language mapping.

## Recommended Approach

### Phase 1: Expand Frequency List (No Mapping)
1. Download larger frequency lists (Google 1T, Common Crawl counts)
2. Merge with existing data
3. Train on real frequencies

### Phase 2: Real Typo Augmentation
1. Use GitHub Typo Corpus (already downloaded: 82k pairs)
2. Use Common Crawl to find more typo patterns
3. Augment with real typos, keeping original word's frequency

### Phase 3: OOV Evaluation
1. Extract OOV words from Common Crawl
2. Evaluate model generalization (don't train on them)
3. Measure zero-shot performance

## Conclusion

**Nearest-word mapping for augmentation is problematic** because:
- Frequency ≠ similarity
- Creates wrong labels
- Doesn't solve the real problem
- Better alternatives exist

**Better**: Expand frequency list directly, use real typos, evaluate on OOV.

