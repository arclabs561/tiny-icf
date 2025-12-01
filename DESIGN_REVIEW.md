# Design Review: Potential Issues and Improvements

## Issues Identified

### 1. ❌ Augmentation Doesn't Match Real Typo Distributions

**Problem**: We're using synthetic misspelling patterns, not real-world distributions.

**Real Typo Patterns** (from research):
- **Keyboard proximity**: Most typos involve adjacent keys (QWERTY layout)
  - `e` → `w`, `r`, `d`, `s` (adjacent keys)
  - `i` → `o`, `k`, `l`, `u` (adjacent keys)
- **Position-dependent**: Errors more common at word boundaries
- **Frequency-dependent**: Common words have more observed typos
- **Language-dependent**: Different languages have different typo patterns

**Current Implementation**: Random patterns, not keyboard-aware.

**Fix Needed**: 
- Use QWERTY keyboard distance
- Weight by key proximity
- Use real typo corpus if available

### 2. ⚠️ ICF Normalization May Be Incorrect

**Current Formula**:
```
ICF = (log(Total_Tokens) - log(Count)) / log(Total_Tokens)
```

**Potential Issue**: This assumes all words are in the same corpus. For multilingual data with language prefixes (`en:word`, `es:palabra`), we're mixing corpora.

**Better Approach**: 
- Compute ICF per language separately
- Or use global corpus (all languages combined)
- Current approach may give wrong scores for multilingual words

### 3. ⚠️ Stratified Sampling May Not Work for Multilingual

**Current**: Samples by ICF rank globally.

**Problem**: With language prefixes, `en:the` and `es:el` are treated as different words. But they're both "the" in their languages. The sampling might not balance languages properly.

**Fix**: 
- Sample per-language first, then combine
- Or remove language prefixes for sampling, add back for training

### 4. ❌ No Validation of Augmentation Quality

**Problem**: We generate typos but don't verify they're realistic.

**Missing**:
- No check if typo is valid (could create impossible strings)
- No verification that typo frequency matches real distributions
- No validation that model learns from typos correctly

### 5. ⚠️ Curriculum Learning Difficulty Scoring

**Current**: ICF (40%) + Length (20%) + Diversity (20%) + Morphology (20%)

**Potential Issues**:
- Weights are arbitrary (no validation)
- Morphology detection is simplistic (just counts affixes)
- Doesn't account for cross-lingual difficulty

### 6. ❌ Missing: Real Typo Dataset Integration

**Problem**: We generate synthetic typos but don't use real typo data.

**Available Datasets**:
- GitHub Typo Corpus (mentioned in search)
- Search engine query typos
- Spell-checker correction logs

**Action**: Download and integrate real typo pairs.

## Design Improvements Needed

### 1. Keyboard-Aware Augmentation

```python
QWERTY_ADJACENT = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's', 'a'], 'e': ['w', 'r', 'd', 's'],
    # ... full QWERTY adjacency map
}

def keyboard_typo(word: str) -> str:
    """Generate typo based on QWERTY keyboard proximity."""
    # Most typos are adjacent key swaps
    # Weight by distance on keyboard
```

### 2. Per-Language ICF Computation

```python
def compute_icf_per_language(word_counts: Dict[str, int]) -> Dict[str, float]:
    """Compute ICF separately for each language."""
    languages = {}
    for word, count in word_counts.items():
        if ':' in word:
            lang, base_word = word.split(':', 1)
            languages.setdefault(lang, {})[base_word] = count
    
    # Compute ICF per language
    icf_scores = {}
    for lang, lang_counts in languages.items():
        total = sum(lang_counts.values())
        for word, count in lang_counts.items():
            key = f"{lang}:{word}"
            icf_scores[key] = compute_normalized_icf_single(count, total)
    
    return icf_scores
```

### 3. Real Typo Dataset Integration

```python
def load_typo_corpus(path: Path) -> Dict[str, str]:
    """Load real typo -> correction pairs."""
    # Format: typo,correction
    # Use these as augmentation targets
```

### 4. Typo Distribution Matching

```python
def match_typo_distribution(typo_corpus: Dict[str, str]) -> Dict[str, float]:
    """Compute frequency of each typo pattern."""
    # Count: adjacent_swap, char_drop, etc.
    # Use these frequencies to weight augmentation
```

## Validation Checklist

- [ ] Verify ICF computation is correct for multilingual data
- [ ] Test augmentation produces realistic typos
- [ ] Validate curriculum difficulty scoring
- [ ] Check stratified sampling balances languages
- [ ] Download and integrate real typo corpus
- [ ] Match augmentation to real typo distributions

