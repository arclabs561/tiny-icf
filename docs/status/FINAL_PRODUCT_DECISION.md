# Final Product Decision: What Really Matters

## The Core Insight

**We're building a frequency estimator for CLEAN TEXT, not a text cleaner.**

## Use Case Analysis

### Primary Use Case: Zero-Shot Text Processing

The model predicts **normalized ICF** (0.0 = common, 1.0 = rare) for arbitrary text.

**Applications**:
1. **Token Filtering in RAG/Retrieval**: Filter stopwords before expensive embedding computation
2. **Zero-Shot Classification**: Weight tokens by informativeness
3. **Text Quality Assessment**: Detect gibberish (high ICF)

## What Really Matters?

### ✅ What We NEED to Handle

1. **Real Words (All Languages)**
   - English, Spanish, French, German, Russian, etc.
   - Common: `"the"`, `"el"`, `"le"` → ICF ~0.0
   - Rare: `"xylophone"`, `"café"` → ICF ~0.7-0.9

2. **Typos/Misspellings**
   - `"cmputer"` → similar to `"computer"`
   - **Why**: Users make typos, need robustness

3. **Morphological Variants**
   - `"run"`, `"runs"`, `"running"` → similar ICF
   - **Why**: Same semantic content

4. **Gibberish Detection**
   - `"qkzjx"` → very high ICF (~0.99)
   - **Why**: This is GOOD - we want to detect gibberish

### ❌ What We DON'T Need to Handle (Preprocessing)

1. **HTML Entities** (`&nbsp;`, `&amp;`) → Filter out
2. **URLs** (`https://example.com`) → Extract words or skip
3. **Code** (`function() { }`) → Skip
4. **Encoding Errors** (`â€™`) → Detect and skip
5. **Emails** (`test@email.com`) → Skip
6. **Pure Numbers** (`12345`) → Skip

## The Product Boundary

### What We're Building

A **word frequency estimator** for **clean, tokenized text**.

**Not**: HTML parser, code detector, URL extractor
**Is**: Frequency predictor for words, robust to typos, handles all languages

### The Pipeline

**Before Model** (Preprocessing):
1. Extract text from HTML
2. Tokenize into words
3. Filter non-linguistic content

**Model**:
- Takes clean words
- Predicts ICF
- Handles typos/morphology

**After Model**:
- Use ICF for filtering/weighting

## Common Crawl "Gunk"

### Should We Train On Raw Common Crawl?

**NO** - Here's why:

1. **Not Representative**: We estimate frequency for clean text, not HTML entities
2. **Frequency Mismatch**: `&nbsp;` is frequent in HTML but not a word
3. **Quality Over Quantity**: Better 100k clean words than 1M with 70% noise

### What We SHOULD Do

1. **Use Clean Frequency Lists**
   - Google 1T Word Corpus (already cleaned)
   - Wikipedia-derived lists (high quality)

2. **Filter Training Data**
   - Only real words (alphanumeric + accents)
   - Length bounds (2-50 chars)
   - Remove HTML entities, URLs, code, etc.

3. **Augment with Real Typos**
   - Real typo corpus (82k pairs)
   - Keyboard-aware typos

## Implementation

### Preprocessing Module (`src/tiny_icf/preprocessing.py`)

**Functions**:
- `is_html_entity()`: Detect HTML entities
- `is_url()`: Detect URLs
- `is_email()`: Detect emails
- `is_code_like()`: Detect code patterns
- `has_encoding_errors()`: Detect mojibake
- `is_pure_number()`: Detect numbers
- `is_gibberish()`: Detect gibberish
- `is_valid_word()`: Combined check
- `clean_text_for_frequency()`: Extract valid words
- `filter_frequency_list()`: Filter frequency list

### Integration

**Data Loading** (`src/tiny_icf/data.py`):
- `load_frequency_list()` now has `filter_noise=True` by default
- Automatically filters HTML entities, URLs, code, etc.
- Keeps only valid words for training

## Testing

### Valid Words ✅
- `"hello"` → Valid
- `"café"` → Valid
- `"qkzjx"` → Valid (gibberish, but still a word - model will assign high ICF)

### Invalid Words ❌
- `"&nbsp;"` → Invalid (HTML entity)
- `"https://example.com"` → Invalid (URL)
- `"function()"` → Invalid (code)
- `"â€™"` → Invalid (encoding error)
- `"12345"` → Invalid (pure number)
- `"test@email.com"` → Invalid (email)

## Conclusion

**What Really Matters**:
1. ✅ Real words (all languages)
2. ✅ Typos/misspellings (robustness)
3. ✅ Morphological variants (generalization)
4. ✅ Gibberish detection (high ICF)

**What Doesn't Matter**:
1. ❌ HTML entities (preprocessing)
2. ❌ URLs (preprocessing)
3. ❌ Code snippets (preprocessing)
4. ❌ Encoding errors (preprocessing)

**The Solution**: Preprocessing module filters noise before frequency estimation.

**The Product**: A frequency estimator for clean text, not a text cleaner.

