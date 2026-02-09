# MCP Research Summary: Best Practices for Word Frequency Estimation

## Key Findings from Research

### 1. ✅ Subword Tokenization is Foundation
- **BPE/WordPiece/SentencePiece** are standard in SOTA models
- Enables handling OOV by decomposing into known subwords
- We already use byte-level (even more granular) ✅

### 2. ✅ Real Typo Corpora are Highly Effective
- Research shows: "real typo augmentation based on empirically derived error models provides substantial benefits"
- Context-aware typo correction > simple spelling correction
- We already have: **82k real typo pairs from GitHub** ✅

### 3. ✅ Direct Frequency Expansion (Not Mapping)
- "Vocabulary expansion using bilingual dictionaries" works well
- "Merge frequency lists directly" - no mapping needed
- Better than nearest-neighbor mapping (frequency ≠ similarity)

### 4. ✅ Hybrid Approach is Best
- SOTA models combine: subword + contextual embeddings + real augmentation
- Multiple complementary strategies > single approach
- Quality over quantity for augmentation

## Recommended Implementation

**Best Practice**: Use real typo corpus + expand frequency list directly

1. **Real Typo Augmentation** (already have 82k pairs)
   - Use GitHub Typo Corpus
   - Augment with real typos, keep correction's frequency
   - Keyboard-aware patterns (already implemented)

2. **Direct Frequency Expansion**
   - Merge frequency lists (no mapping)
   - Use words with real frequency counts
   - Expand vocabulary with actual data

3. **Hybrid Training**
   - Combine real typos + expanded frequencies
   - Use keyboard-aware augmentation
   - Quality over quantity

## Implementation Plan

1. Integrate real typo corpus into training
2. Expand frequency list (merge, don't map)
3. Update training to use both
4. Validate improvements

