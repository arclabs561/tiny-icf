# Product Thinking: What Really Matters?

## The Core Question

**What is the actual use case for this frequency estimator?**

Let's think through the product requirements:

## Use Case Analysis

### Primary Use Case: Zero-Shot Text Processing

The model predicts **normalized ICF** (0.0 = common, 1.0 = rare) for arbitrary text.

**Key Applications**:
1. **Token Filtering in RAG/Retrieval**
   - Filter out stopwords before expensive embedding computation
   - Keep only informative tokens (high ICF)
   - Reduce compute cost by 30-50%

2. **Zero-Shot Classification/Retrieval**
   - Weight tokens by informativeness
   - Down-weight common words, up-weight rare content words
   - Works without domain-specific training

3. **Text Quality Assessment**
   - Detect gibberish (very high ICF for random strings)
   - Identify domain-specific jargon (moderate ICF)
   - Filter low-quality content

## What Really Matters?

### ✅ What We NEED to Handle

1. **Real Words (All Languages)**
   - English, Spanish, French, German, Russian, etc.
   - Common words: `"the"`, `"el"`, `"le"` → ICF ~0.0
   - Rare words: `"xylophone"`, `"café"` → ICF ~0.7-0.9

2. **Typos/Misspellings**
   - `"cmputer"` → should predict similar to `"computer"`
   - `"recieve"` → should predict similar to `"receive"`
   - **Why**: Users make typos, we need robustness

3. **Morphological Variants**
   - `"run"`, `"runs"`, `"running"` → similar ICF
   - `"friend"`, `"friendly"`, `"unfriendliness"` → related ICF
   - **Why**: Same semantic content, should have similar informativeness

4. **Domain-Specific Terms**
   - `"logits"` (rare in general, common in ML) → moderate ICF
   - `"patient"` (common in medical) → low ICF in medical domain
   - **Why**: Need to handle domain shift gracefully

### ❌ What We DON'T Need to Handle

1. **HTML Entities**
   - `&nbsp;`, `&amp;`, `&lt;` → These are **not words**
   - **Action**: Filter out before processing
   - **Why**: Not linguistic content, just markup

2. **URLs**
   - `https://example.com/path` → Not a word
   - **Action**: Extract words from URLs, not process URLs as words
   - **Why**: URLs are structured data, not language

3. **Code Snippets**
   - `function() { return x; }` → Not natural language
   - **Action**: Filter out code blocks
   - **Why**: Code has different frequency patterns than language

4. **Encoding Errors (Mojibake)**
   - `â€™` (corrupted apostrophe) → Not valid text
   - **Action**: Detect and filter encoding errors
   - **Why**: Corrupted data, not real language

5. **Random Gibberish**
   - `"qkzjx"` → Should predict very high ICF (~0.99)
   - **Action**: Model should naturally assign high ICF
   - **Why**: This is actually GOOD - we want to detect gibberish

6. **Mixed Content**
   - `"hello123"` → Part word, part number
   - **Action**: Could split or treat as single token
   - **Why**: Depends on tokenization strategy

## The Real Product Requirements

### Core Functionality

**Input**: Arbitrary word/token string
**Output**: Normalized ICF (0.0 = common, 1.0 = rare)

**Must Work For**:
- ✅ Real words (all languages)
- ✅ Typos/misspellings
- ✅ Morphological variants
- ✅ Domain-specific terms
- ✅ Gibberish (should return high ICF)

**Doesn't Need To Work For**:
- ❌ HTML entities (filter first)
- ❌ URLs (extract words first)
- ❌ Code (filter first)
- ❌ Encoding errors (detect and filter)

### The Preprocessing Pipeline

**Before Frequency Estimation**:
1. **Extract text from HTML** (remove tags, entities)
2. **Detect language** (optional, for per-language ICF)
3. **Tokenize** (split into words)
4. **Filter non-linguistic**:
   - URLs → extract domain/path words
   - Code blocks → skip
   - Encoding errors → detect and skip
   - Pure numbers → skip (or treat separately)

**Then Frequency Estimation**:
- Process each token through the model
- Get ICF score
- Use for filtering/weighting

## What About Common Crawl "Gunk"?

### The Reality

Common Crawl contains:
- **70% duplication** (paragraph level)
- **HTML boilerplate** (menus, headers, footers)
- **Repetitive n-grams** (corrupted text)
- **Encoding errors** (mojibake)
- **Code snippets** (embedded scripts)
- **URLs and structured data**

### Should We Train On This?

**NO** - Here's why:

1. **Not Representative of Use Case**
   - We're estimating frequency for **clean text**
   - Not for HTML entities or code snippets
   - Training on noise teaches wrong patterns

2. **Frequency Mismatch**
   - `&nbsp;` appears frequently in Common Crawl (HTML)
   - But it's not a word, shouldn't be in our model
   - Training on it would pollute the model

3. **Quality Over Quantity**
   - Better to train on 100k clean words
   - Than 1M words with 70% noise
   - Model learns wrong patterns from noise

### What We SHOULD Do

1. **Use Clean Frequency Lists**
   - Google 1T Word Corpus (already cleaned)
   - Wikipedia-derived lists (high quality)
   - Domain-specific lists (medical, legal, etc.)

2. **Filter Training Data**
   - Only real words (alphanumeric + accents)
   - Minimum length (e.g., 2 characters)
   - Maximum length (e.g., 50 characters)
   - Language detection (optional)

3. **Augment with Real Typos**
   - Use real typo corpus (82k pairs)
   - Keyboard-aware typos
   - Morphological variants

4. **Test on Edge Cases**
   - Gibberish → should return high ICF
   - Typos → should return similar to correct word
   - Mixed content → depends on tokenization

## The Product Decision

### What We're Building

A **word frequency estimator** for **clean, tokenized text**.

**Not**:
- HTML parser
- Code detector
- URL extractor
- Encoding error fixer

**Is**:
- Frequency predictor for words
- Robust to typos
- Handles all languages
- Detects gibberish (high ICF)

### The Preprocessing Boundary

**Before Our Model**:
- Extract text from HTML
- Tokenize into words
- Filter non-linguistic content
- Detect encoding errors

**Our Model**:
- Takes clean words
- Predicts ICF
- Handles typos/morphology
- Works for all languages

**After Our Model**:
- Use ICF for filtering/weighting
- Apply to downstream tasks

## Recommendations

### 1. Focus on Clean Data

**Training Data**:
- ✅ Clean frequency lists (Google 1T, Wikipedia)
- ✅ Real typo corpus (for robustness)
- ✅ Multilingual word lists (for coverage)
- ❌ Raw Common Crawl (too noisy)

### 2. Handle Edge Cases in Preprocessing

**Before Model**:
- Filter HTML entities
- Extract words from URLs
- Skip code blocks
- Detect encoding errors

**Model Should**:
- Handle typos (robustness)
- Handle gibberish (high ICF)
- Handle all languages (UTF-8)

### 3. Test on Real Use Cases

**Validation**:
- Real words (all languages) → correct ICF
- Typos → similar to correct word
- Gibberish → very high ICF
- Domain terms → reasonable ICF

**Don't Test On**:
- HTML entities (filter first)
- URLs (extract words first)
- Code (filter first)

## Conclusion

**What Really Matters**:
1. ✅ Real words (all languages)
2. ✅ Typos/misspellings (robustness)
3. ✅ Morphological variants (generalization)
4. ✅ Gibberish detection (high ICF)

**What Doesn't Matter**:
1. ❌ HTML entities (preprocessing concern)
2. ❌ URLs (preprocessing concern)
3. ❌ Code snippets (preprocessing concern)
4. ❌ Encoding errors (preprocessing concern)

**The Product**: A frequency estimator for **clean text**, not a text cleaner.

