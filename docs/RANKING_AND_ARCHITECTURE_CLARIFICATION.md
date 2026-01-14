# Ranking and Architecture Clarification

## What Ranking Are We Aligning With?

### The Core Task: Ranking Words by ICF

**ICF (Inverse Collection Frequency)** = How rare/common a word is
- **0.0** = Very common (e.g., "the", "a", "is")
- **1.0** = Very rare (e.g., "xylophone", "qzxbjk" (gibberish))

**We're ranking words by their ICF scores:**
```
"the" (0.0) < "apple" (0.3) < "xylophone" (0.95) < "qzxbjk" (1.0)
```

### Why We Use Ranking Losses

**The problem**: We're predicting continuous ICF values, but what we actually care about is **relative ordering**.

**Example**:
- If model predicts: "the" → 0.5, "xylophone" → 0.4
- MSE loss might be low (both close to some average)
- **But ordering is WRONG!** "the" should be < "xylophone"

**Solution**: Ranking losses enforce correct relative ordering:
- If "the" is more common than "xylophone" in ground truth
- Then model must predict: icf("the") < icf("xylophone")
- Even if exact values are off, ordering is preserved

### What We're Aligning With During Distillation

**During knowledge distillation**, we align with the **teacher model's ranking**:

1. **Teacher model** (language model like `all-MiniLM-L6-v2`):
   - Processes words → embeddings → ICF predictions
   - Has better semantic understanding
   - Produces more accurate ICF rankings

2. **Student model** (our character-level CNN):
   - Processes words → character patterns → ICF predictions
   - Limited to character-level patterns
   - Less accurate initially

3. **Distillation alignment**:
   - We want student to **rank words the same way teacher does**
   - If teacher says: "the" < "xylophone" < "qzxbjk"
   - Student should learn: "the" < "xylophone" < "qzxbjk"
   - Even if exact ICF values differ, the **ranking order** should match

**Why this matters**:
- Teacher has semantic knowledge (word meanings, context)
- Student only has character patterns
- By aligning rankings, we transfer semantic understanding → character patterns

---

## Architecture: We're STILL Using 1D CNN!

### Yes, We're Still Doing 1D CNN

**The core architecture has NOT changed** - we're still using **1D convolutions**:

```python
# From src/tiny_icf/model.py
self.conv3 = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)
self.conv5 = nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2)
self.conv7 = nn.Conv1d(emb_dim, conv_channels, kernel_size=7, padding=3)
```

**`Conv1d` = 1D Convolution** (operates on sequences, not 2D images)

### Architecture Flow

```
Input: Word as UTF-8 bytes [batch, seq_len]
  ↓
Byte Embedding [batch, seq_len, emb_dim]
  ↓
Parallel 1D CNNs:
  - Conv1d(kernel=3) → trigrams like "ing", "pre"
  - Conv1d(kernel=5) → roots like "graph"
  - Conv1d(kernel=7) → complex affixes
  ↓
Multi-scale Pooling (max, mean, last)
  ↓
[Optional] Multi-head Self-Attention (NEW - for long-range dependencies)
  ↓
MLP Head → ICF score [batch, 1]
```

### What Changed: Attention Added (Optional)

**We added attention as an enhancement**, but it's **optional**:

```python
# NEW: Optional attention mechanism
if use_attention:
    self.attention = nn.MultiheadAttention(...)
    # Applies attention to concatenated conv outputs
    # Helps with long-range dependencies
else:
    self.attention = None  # Original 1D CNN behavior
```

**Why attention?**
- 1D CNNs have limited receptive field
- Attention can model long-range dependencies
- But core architecture is still 1D CNN-based

### Why 1D CNN?

**1D CNN is perfect for this task**:
- **Sequential data**: Words are character sequences
- **Local patterns**: Character n-grams (prefixes, suffixes, roots)
- **Efficient**: Fast inference, small model size
- **Universal**: Works with any UTF-8 language

**We're NOT doing 2D CNN** (that would be for images):
- 2D CNN: `Conv2d` (height × width)
- 1D CNN: `Conv1d` (sequence length) ← **This is what we use**

---

## Summary

### What Ranking?
- **Ranking words by ICF** (common → rare)
- During distillation: **Align with teacher's ranking**
- Goal: Preserve relative ordering, not exact values

### What Architecture?
- **1D CNN** (Conv1d) - still the core architecture
- **Attention added** as optional enhancement
- **Not 2D CNN** - we're processing sequences, not images

### Why This Matters
- **Ranking alignment** transfers semantic knowledge from teacher
- **1D CNN** is the right architecture for character sequences
- **Attention** helps with long-range dependencies but doesn't replace CNN

---

## Code References

**1D CNN Architecture**:
- `src/tiny_icf/model.py`: `UniversalICF` class
- Uses `nn.Conv1d` for parallel convolutions
- Kernel sizes: 3, 5, 7 (character n-grams)

**Ranking Losses**:
- `src/tiny_icf/loss.py`: `CombinedLoss` with ranking component
- `src/tiny_icf/loss_unified.py`: `ICFPredictionLoss` with Spearman loss
- Enforces: if word1 < word2 in ground truth, then pred1 < pred2

**Distillation Alignment**:
- `src/tiny_icf/distillation.py`: `DistillationLoss`
- Aligns student predictions with teacher predictions
- Preserves teacher's ranking order

