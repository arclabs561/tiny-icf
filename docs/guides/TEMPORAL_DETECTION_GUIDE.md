# Temporal/Era Detection: Predicting Word Usage Over Time

## Overview

The model now predicts when words were used, their historical era, and usage span. This helps understand:
- **Historical context**: When was this word commonly used?
- **Era classification**: What historical period does it belong to?
- **Usage span**: How long has this word been in use?
- **Temporal category**: Very old, old, classic, recent, very recent

## Features

### 1. Era Detection
Detects historical era based on:
- **Morphological patterns**: Suffixes (-eth, -est = archaic)
- **Prefixes**: (ye, thou, thy = archaic)
- **Word structure**: Compound words, neologisms
- **Examples**: Known archaic/contemporary words

### 2. Usage Span Estimation
Estimates when words were most commonly used:
- **Archaic**: pre-1800s (thou, thee, hast)
- **Early Modern**: 1800s-early 1900s
- **Modern**: 1900s-2000s (computer, technology)
- **Contemporary**: 2000s-present (selfie, tweet)
- **Neologism**: 2010s-present (iPhone, YouTube)

### 3. Temporal Category
Classifies words by temporal usage:
- **Very Old**: Archaic or very old usage (pre-1800s)
- **Old**: Older usage (1800s-early 1900s)
- **Classic**: Classic/modern usage (1900s-2000s)
- **Recent**: Recent usage (2000s-2010s)
- **Very Recent**: Very recent usage (2010s-present)

## Usage

### Basic Prediction with Temporal Analysis

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "thou computer selfie" \
    --data data/word_frequency.csv
```

### JSON Output

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "thou selfie tweet" \
    --data data/word_frequency.csv \
    --json
```

## Example Output

```json
{
  "word": "thou",
  "icf_score": 0.95,
  "temporal": {
    "primary_era": "archaic",
    "era_name": "Archaic (pre-1800s)",
    "era_confidence": 0.85,
    "usage_span": "pre-1800s",
    "approximate_era": "pre-1800s",
    "temporal_category": "very_old",
    "is_archaic": true,
    "is_neologism": false,
    "is_contemporary": false
  }
}
```

```json
{
  "word": "selfie",
  "icf_score": 0.8,
  "temporal": {
    "primary_era": "contemporary",
    "era_name": "Contemporary (2000s-present)",
    "era_confidence": 0.9,
    "usage_span": "2000s-present",
    "approximate_era": "2000s-present",
    "temporal_category": "very_recent",
    "is_archaic": false,
    "is_neologism": false,
    "is_contemporary": true
  }
}
```

## Detection Methods

### 1. Pattern-Based Detection
- **Suffixes**: `-eth`, `-est` → archaic
- **Prefixes**: `ye`, `thou` → archaic
- **Regex patterns**: Match known era patterns
- **Examples**: Known words from each era

### 2. ICF Score Integration
- High ICF + archaic patterns → very old
- Low ICF + contemporary patterns → very recent
- Combines frequency with structural patterns

### 3. Technology/Social Media Detection
- Tech terms: `app`, `web`, `net`, `tech` → contemporary
- Social terms: `tweet`, `post`, `share` → very recent
- Compound words: Often modern/contemporary

## Era Classifications

### Archaic (pre-1800s)
- Examples: thou, thee, thy, hast, doth, hath
- Patterns: `-eth`, `-est` suffixes, `ye` prefix
- Usage: Pre-1800s English

### Early Modern (1800s)
- Examples: wherefore, hence, whence
- Patterns: Classical English structure
- Usage: 1800s-early 1900s

### Modern (1900s-2000s)
- Examples: computer, internet, technology
- Patterns: Standard modern English
- Usage: 1900s-2000s

### Contemporary (2000s-present)
- Examples: selfie, tweet, blog, app
- Patterns: Modern technology/social terms
- Usage: 2000s-present

### Neologism (2010s-present)
- Examples: iPhone, YouTube, WiFi, eBay
- Patterns: Compound words, brand names, tech terms
- Usage: 2010s-present

## Use Cases

### 1. Historical Text Analysis
```python
# Identify archaic words in text
for word in text:
    temporal = estimate_usage_period(word)
    if temporal['is_archaic']:
        print(f"Archaic word: {word}")
```

### 2. Modern Language Detection
```python
# Find contemporary/neologisms
for word in text:
    temporal = estimate_usage_period(word)
    if temporal['is_contemporary']:
        print(f"Modern word: {word} ({temporal['approximate_era']})")
```

### 3. Temporal Filtering
```python
# Filter by era
archaic_words = [w for w in words if estimate_usage_period(w)['is_archaic']]
contemporary_words = [w for w in words if estimate_usage_period(w)['is_contemporary']]
```

## Integration

Temporal detection is automatically included in:
- `predict.py` (enhanced predictions)
- `predict_advanced.py` (advanced analysis)

All predictions now include:
- ICF score
- Language detection
- **Temporal/era detection** (NEW)
- Percentile rank
- Similar words
- Feature importance

## Limitations

Current implementation uses:
- Pattern matching (suffixes, prefixes, examples)
- ICF score integration
- Heuristics for tech/social terms

**Future improvements**:
- Train on historical frequency data
- Use model's internal features for temporal patterns
- Improve accuracy with labeled temporal datasets
- Add decade-level granularity

## Examples

| Word | Era | Usage Span | Category |
|------|-----|------------|----------|
| thou | Archaic | pre-1800s | very_old |
| computer | Modern | 1900s-2000s | classic |
| selfie | Contemporary | 2000s-present | very_recent |
| groovy | Modern | 1900s-2000s | classic |
| tweet | Contemporary | 2000s-present | very_recent |
| iPhone | Neologism | 2010s-present | very_recent |

