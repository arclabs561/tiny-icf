# Features and Additional Scores Analysis

## Question 1: More Features or Learn Them?

### Current Architecture
The model **learns features automatically** via:
- **Byte embeddings**: Learns character-level patterns
- **CNN kernels (3, 5, 7)**: Extract morphological n-grams
  - Kernel 3: Trigrams (prefixes/suffixes like "ing", "pre")
  - Kernel 5: Roots (like "graph", "comput")
  - Kernel 7: Complex affixes
- **Multi-scale pooling**: Max + Mean + Last position
- **Implicit length**: Via padding/truncation

### What It Learns Automatically
1. **Morphological patterns**: Prefixes, suffixes, roots
2. **Character sequences**: Byte-level n-grams
3. **Structural validity**: Phonotactics (what sequences are valid)
4. **Length patterns**: Implicitly via padding

### Should We Add Explicit Features?

**Recommendation: NO**

**Why:**
- CNNs are designed to extract these patterns automatically
- Adding explicit features (word length, character counts, etc.) would:
  - Increase model size
  - Add preprocessing complexity
  - Likely not improve performance (CNNs already learn these)
  - Violate the "learn from bytes" principle

**Exception:** If model struggles with specific patterns, we could add:
- Explicit length encoding (but padding already handles this)
- Character class features (but byte embeddings learn this)

**Current Status:** Model learns features automatically ✓

---

## Question 2: Return ERA or Other Scores?

### Current Output
- **Single ICF score** (0.0 to 1.0)
- No confidence/uncertainty
- No error estimates

### Potential Additional Scores

#### 1. **Confidence/Uncertainty**
**What:** How certain is the model about its prediction?

**Implementation Options:**
- **Monte Carlo Dropout**: Run inference multiple times with dropout enabled, compute std
- **Uncertainty Head**: Add second output head predicting uncertainty
- **Ensemble**: Train multiple models, use variance

**Use Cases:**
- Filter low-confidence predictions
- Identify edge cases
- Quality control

**Trade-offs:**
- Adds complexity
- Slower inference (if Monte Carlo)
- More parameters (if uncertainty head)

#### 2. **Error Rate Approximation (ERA)**
**What:** Expected error range for this prediction?

**Implementation Options:**
- **Validation-based**: Use validation set to estimate per-word error
- **Model-based**: Train model to predict its own error
- **Heuristic**: Based on word length, pattern rarity, etc.

**Use Cases:**
- Error bounds for downstream tasks
- Quality assessment
- Filtering unreliable predictions

**Trade-offs:**
- Requires validation data
- Adds complexity
- May not generalize well

#### 3. **Feature Attribution**
**What:** Which patterns/characters contributed most?

**Implementation Options:**
- **Gradient-based**: Grad-CAM, Integrated Gradients
- **Attention**: Add attention mechanism
- **SHAP values**: Feature importance scores

**Use Cases:**
- Debugging model behavior
- Understanding predictions
- Feature analysis

**Trade-offs:**
- Significant complexity
- Slower inference
- May not be needed for production

### Recommendation

**For Now: Single ICF Score is Sufficient**

**Why:**
1. **Use case doesn't require it**: Primary use is filtering/weighting tokens
2. **Simplicity**: Single score is easier to use and reason about
3. **Performance**: Model is already small and fast
4. **Can add later**: If needed, can add uncertainty without breaking changes

**When to Add:**
- **Uncertainty**: If users need to filter low-confidence predictions
- **ERA**: If downstream tasks need error bounds
- **Attribution**: If debugging/analysis is needed

**Implementation Priority:**
1. **Uncertainty** (if needed): Monte Carlo Dropout (easiest, no model changes)
2. **ERA** (if needed): Validation-based lookup table
3. **Attribution** (if needed): Gradient-based methods

---

## Summary

### Features
- ✅ **Model learns features automatically** via CNNs
- ❌ **No explicit features needed** (would add complexity without benefit)

### Additional Scores
- ✅ **Current: Single ICF score** (sufficient for use case)
- ⚠️ **Can add later if needed:**
  - Uncertainty (Monte Carlo Dropout)
  - ERA (validation-based)
  - Attribution (gradient-based)

**Recommendation:** Keep it simple. Add uncertainty/ERA only if use cases require it.

