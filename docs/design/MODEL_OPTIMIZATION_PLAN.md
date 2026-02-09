# Model Optimization Plan: Tiny but Powerful

## Goal
**Tiny end model (< 20k parameters) that is powerful for ICF estimation.**

## Current Status

### UniversalICF
- **Parameters**: ~40k
- **Size**: 161 KB (float32) / 81 KB (float16)
- **Architecture**: Byte-level CNN with parallel kernels

### Proposed Alternatives

#### 1. HierarchicalICF (~12k params)
**Structure:**
- Character level → N-gram level → Word level
- Smaller embeddings (32 vs 48)
- Fewer conv channels (16 vs 24)
- Hierarchical composition

**Advantages:**
- 70% smaller than UniversalICF
- More structured representation
- Better for compositional words

**Trade-offs:**
- May need more training
- Architecture more complex

#### 2. BoxEmbeddingICF (~8k params)
**Structure:**
- Box embeddings (hyperrectangles) for frequency ranges
- Center + width representation
- Better for ranking tasks

**Advantages:**
- 80% smaller than UniversalICF
- Naturally handles ranges/uncertainty
- Good for ranking (ICF is about relative ordering)

**Trade-offs:**
- Novel architecture (less tested)
- May need custom loss functions

## Training Data Improvements

### 1. Historical Ngrams
**Source**: Google Books Ngram (2000-2019)
- **Size**: Very large (terabytes)
- **Approach**: Download sample or use pre-processed
- **Alternative**: Common Crawl processed frequencies

**Benefits:**
- Historical word usage patterns
- Better for rare words
- Temporal trends

### 2. Recent Blog/Web Data
**Sources:**
- Common Crawl (recent crawls)
- Reddit/Twitter word frequencies
- Wikipedia word frequencies
- News article frequencies

**Benefits:**
- Modern neologisms
- Current usage patterns
- Domain-specific terms

**Implementation:**
- Use existing `word_frequency_modern.csv`
- Add more via `scripts/add_modern_words.py`
- Future: Scrape recent blog/news frequencies

## Optimization Strategy

### Phase 1: Data (Current)
1. ✅ Added modern words (`word_frequency_modern.csv`)
2. ⏳ Download historical ngram sample
3. ⏳ Add blog/web frequencies

### Phase 2: Architecture (Next)
1. Test HierarchicalICF
2. Test BoxEmbeddingICF
3. Compare performance vs size
4. Choose best architecture

### Phase 3: Compression (Future)
1. Quantization (float16, int8)
2. Pruning (remove low-importance weights)
3. Knowledge distillation (small model from large)

## Next Steps

1. **Continue current training** (let it finish)
2. **Test hierarchical/box models** on small dataset
3. **Download ngram sample** or use Common Crawl
4. **Add more modern words** from blogs/web
5. **Compare all models** (size vs performance)
6. **Choose best architecture** and retrain

## Target Metrics

- **Size**: < 20k parameters (< 80 KB float32, < 40 KB float16)
- **Performance**: 
  - MAE < 0.1
  - Spearman correlation > 0.8
  - Jabberwocky Protocol: 4/5 tests pass
- **Speed**: < 1ms inference per word (CPU)

