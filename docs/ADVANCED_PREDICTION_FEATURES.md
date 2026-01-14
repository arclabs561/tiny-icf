# Advanced Prediction Features: Error Analysis, Percentile Rank, and More

## Overview

Beyond basic ICF scores, the model can now return:
- **Percentile Rank**: Where does this word rank among all words? (0% = most common, 100% = most rare)
- **Prediction Intervals**: Confidence bounds based on historical error distribution
- **Similar Words**: Words with similar ICF scores
- **Feature Importance**: Which parts of the word contribute most to the prediction
- **Error Analysis**: Expected error magnitude and calibration

## Usage

### Basic Advanced Prediction

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone qzxbjk" \
    --data data/word_frequency.csv
```

Output:
```
Word                 ICF      Percentile    Interval            Category        
--------------------------------------------------------------------------------
the                  0.0000   0.0%          [0.00, 0.20]        very_common     
xylophone            1.0000   99.5%         [0.80, 1.00]        very_rare       
qzxbjk               1.0000   99.8%         [0.80, 1.00]        very_rare       
```

### With Feature Analysis

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "xylophone" \
    --data data/word_frequency.csv \
    --features \
    --json
```

Output includes:
- Percentile rank
- Prediction intervals (95% confidence)
- Similar words
- Feature importance analysis
- Character-level importance

### JSON Output

```bash
uv run src/tiny_icf/predict_advanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone" \
    --data data/word_frequency.csv \
    --json
```

## Features Explained

### 1. Percentile Rank

**What it is**: Where this word ranks among all words in the reference dataset.

- **0%**: Most common word (lowest ICF)
- **50%**: Median word
- **100%**: Most rare word (highest ICF)

**Example**:
```json
{
  "word": "the",
  "icf_score": 0.0,
  "percentile_rank": 0.1  // Rarer than only 0.1% of words (very common)
}
```

### 2. Prediction Intervals

**What it is**: Confidence bounds based on historical prediction errors.

- **Lower bound**: Minimum likely ICF score (95% confidence)
- **Upper bound**: Maximum likely ICF score (95% confidence)
- **Error std**: Standard deviation of prediction errors

**Example**:
```json
{
  "prediction_interval": {
    "lower": 0.00,
    "upper": 0.20,
    "error_std": 0.31,
    "confidence_level": 0.95
  }
}
```

### 3. Similar Words

**What it is**: Words with similar ICF scores from the reference dataset.

**Example**:
```json
{
  "similar_words": [
    {"word": "xylophone", "icf_score": 0.95, "score_diff": 0.05},
    {"word": "zephyr", "icf_score": 0.92, "score_diff": 0.08}
  ]
}
```

### 4. Feature Importance

**What it is**: Analysis of which parts of the word contribute most to the prediction.

- **Top features**: Indices of most important hidden layer features
- **Top characters**: Characters that contribute most to the prediction
- **Feature statistics**: Mean, std of feature activations

**Example**:
```json
{
  "feature_analysis": {
    "top_features": [12, 34, 7, 45, 23],
    "feature_magnitudes": {
      "12": 0.8234,
      "34": 0.7123
    },
    "top_characters": [
      {"char": "x", "importance": 0.45},
      {"char": "y", "importance": 0.38}
    ],
    "feature_std": 0.12,
    "feature_mean": 0.05
  }
}
```

## Python API

```python
from tiny_icf.model import UniversalICF
from tiny_icf.predict_advanced import predict_with_analysis
from tiny_icf.data import WordICFDataset
import torch

# Load model
model = UniversalICF()
model.load_state_dict(torch.load('models/model_diagnostic_rank5.pt'))
model.eval()

# Load reference dataset
reference_dataset = WordICFDataset('data/word_frequency.csv')

# Predict with analysis
result = predict_with_analysis(
    model,
    "xylophone",
    torch.device('cpu'),
    reference_dataset=reference_dataset,
    include_features=True,
)

print(f"ICF Score: {result['icf_score']:.4f}")
print(f"Percentile Rank: {result['percentile_rank']:.1f}%")
print(f"Prediction Interval: [{result['prediction_interval']['lower']:.2f}, {result['prediction_interval']['upper']:.2f}]")
print(f"Similar Words: {[w['word'] for w in result['similar_words']]}")
```

## Use Cases

### 1. Understanding Word Rarity

Percentile rank tells you how rare a word is relative to all words:
- "the" at 0.1% = extremely common
- "xylophone" at 99.5% = extremely rare
- "unfriendliness" at 85% = quite rare

### 2. Confidence in Predictions

Prediction intervals show uncertainty:
- Narrow interval = confident prediction
- Wide interval = uncertain prediction

### 3. Finding Similar Words

Similar words help understand:
- What other words have similar frequency?
- Are there patterns in rare/common words?

### 4. Debugging Predictions

Feature importance helps understand:
- Which characters matter most?
- What patterns is the model detecting?

## Performance Notes

- **Reference dataset**: Optional but recommended for percentile/similarity
- **Sampling**: Uses up to 10,000 samples from reference dataset for efficiency
- **Feature analysis**: Adds ~10-20ms per word (computes gradients)

## Examples

### Example 1: Common Word
```json
{
  "word": "the",
  "icf_score": 0.0,
  "percentile_rank": 0.1,
  "prediction_interval": {"lower": 0.0, "upper": 0.2},
  "similar_words": [
    {"word": "a", "icf_score": 0.0, "score_diff": 0.0},
    {"word": "an", "icf_score": 0.0, "score_diff": 0.0}
  ]
}
```

### Example 2: Rare Word
```json
{
  "word": "xylophone",
  "icf_score": 1.0,
  "percentile_rank": 99.5,
  "prediction_interval": {"lower": 0.8, "upper": 1.0},
  "similar_words": [
    {"word": "zephyr", "icf_score": 0.95, "score_diff": 0.05},
    {"word": "quixotic", "icf_score": 0.92, "score_diff": 0.08}
  ]
}
```

## Benefits

1. **Context**: Percentile rank provides context for raw scores
2. **Uncertainty**: Prediction intervals show confidence
3. **Discovery**: Similar words help find patterns
4. **Debugging**: Feature importance explains predictions
5. **Analysis**: Error analysis helps understand model behavior

