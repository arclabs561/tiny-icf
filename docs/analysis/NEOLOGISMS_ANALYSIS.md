# Neologisms and Portmanteaus: How the Model Handles Them

## Current Behavior

### Portmanteaus (Blended Words)

**In Training Data:**
- `brunch` (breakfast + lunch) → ICF: 0.6170 (count: 2,413)
- `smog` (smoke + fog) → ICF: 0.6818 (count: 645)
- `motel` (motor + hotel) → ICF: 0.5510 (count: 9,233)
- `infomercial` (information + commercial) → ICF: 0.7287 (count: 248)

**NOT in Training Data (Model Generalizes):**
- `frenemy` (friend + enemy) → Model predicts: 0.5874
- `hangry` (hungry + angry) → Model predicts: 0.5651
- `webinar` (web + seminar) → Model predicts: 0.6356
- `spork` (spoon + fork) → Model predicts: 0.5675

**Analysis:** Model generalizes well to unseen portmanteaus by learning:
- Component morphological patterns (prefixes/suffixes)
- Character sequences that look "English-like"
- Structural validity (phonotactics)

### Neologisms (New Words)

**In Training Data:**
- `selfie` → ICF: 0.6504 (count: 1,222)
- `blog` → ICF: 0.6029 (count: 3,217)
- `tweet` → ICF: 0.6253 (count: 2,038)
- `google` → ICF: 0.5994 (count: 3,453)

**NOT in Training Data (Model Generalizes):**
- `uberize` → Model predicts: 0.6191
- `cryptocurrency` → Model predicts: 0.6711

**Analysis:** Model handles modern neologisms reasonably well, predicting moderate-to-high ICF (0.6-0.7), which is appropriate for new/rare words.

## How It Works

### 1. **If Word is in Training Data**
- Model learns ICF directly from frequency counts
- Best case: Accurate prediction

### 2. **If Word is NOT in Training Data (Zero-Shot)**
Model generalizes from learned patterns:

**Morphological Patterns:**
- CNNs learn common prefixes (`pre-`, `un-`, `re-`)
- CNNs learn common suffixes (`-ing`, `-tion`, `-ly`)
- Portmanteaus often contain recognizable parts

**Character Sequences:**
- Byte-level embeddings learn valid character combinations
- Portmanteaus follow English phonotactics (sound patterns)
- Model recognizes "word-like" sequences

**Structural Validity:**
- Model learns what sequences are plausible
- Portmanteaus are structurally valid (unlike gibberish)
- Predicts moderate ICF (not extreme)

## Strengths

1. **Generalizes to Unseen Portmanteaus**: Model predicts reasonable ICF for words it hasn't seen
2. **Learns Morphological Patterns**: Recognizes component parts
3. **Handles Modern Neologisms**: Works for new words like "uberize", "cryptocurrency"

## Limitations

1. **May Overestimate Rare Words**: New portmanteaus might get higher ICF than they deserve (if they become common)
2. **Training Data Recency**: Older frequency lists may miss recent neologisms
3. **Domain-Specific**: Very domain-specific neologisms might be misclassified

## Recommendations

### 1. **Add More Recent Portmanteaus/Neologisms to Training Data**

**Priority Words to Add:**
- `frenemy`, `hangry`, `webinar` (common modern portmanteaus)
- `uberize`, `cryptocurrency`, `blockchain` (tech neologisms)
- `mansplain`, `woke`, `cancel` (social media terms)
- `podcast`, `vlog`, `meme` (internet terms)

**How to Add:**
- Download recent frequency lists (2020-2024)
- Use Google Books Ngram or Common Crawl recent data
- Manually curate high-frequency neologisms

### 2. **Data Augmentation for Portmanteaus**

**Synthetic Portmanteau Generation:**
- Combine common word pairs: `breakfast + lunch` → `brunch`
- Train model to recognize blended patterns
- Helps generalization to unseen portmanteaus

### 3. **Regular Data Updates**

- Update training data annually with new high-frequency words
- Track emerging neologisms
- Retrain model periodically

## Current Training Data Status

- **Total words**: ~50k (word_frequency.csv)
- **Multilingual**: 400k words (multilingual_combined.csv)
- **Coverage**: Good for established words, moderate for recent neologisms

## Next Steps

1. ✅ Document current behavior (this file)
2. ⏳ Download recent frequency lists (2020-2024)
3. ⏳ Add high-frequency neologisms to training data
4. ⏳ Test model on portmanteau/neologism test set
5. ⏳ Consider portmanteau augmentation during training

