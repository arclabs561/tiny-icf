# Enhanced Prediction: Returning More Than Just Score

## Overview

The model now returns rich information beyond just the ICF score, including:
- **ICF Score**: The predicted frequency score (0.0=common, 1.0=rare)
- **Interpretation**: Human-readable category (Very Common, Common, Rare, Very Rare)
- **Category**: Machine-readable category string
- **Confidence**: Confidence estimate based on feature activations (0.0-1.0)
- **Raw Output**: Raw model output before clamping

## Usage

### Basic Prediction (Score Only)

```python
from tiny_icf.model import UniversalICF
from tiny_icf.predict import predict_icf
import torch

model = UniversalICF()
model.load_state_dict(torch.load('models/model_diagnostic_rank5.pt'))
model.eval()

# Simple prediction (returns float)
score = predict_icf(model, "the", torch.device('cpu'))
print(f"ICF Score: {score:.4f}")
```

### Enhanced Prediction (Rich Output)

```python
# Detailed prediction (returns dict)
result = predict_icf(model, "the", torch.device('cpu'), return_details=True)
print(result)
# {
#     'word': 'the',
#     'icf_score': 0.0000,
#     'interpretation': 'Very Common (stopword-like)',
#     'category': 'very_common',
#     'confidence': 0.8234,
#     'raw_output': -0.0123
# }
```

### Command Line Usage

#### Basic Output
```bash
uv run src/tiny_icf/predict.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone qzxbjk"
```

Output:
```
Word                 ICF Score    Interpretation                    
--------------------------------------------------------------------------------
the                  0.0000       Very Common (stopword-like)        
xylophone            1.0000       Very Rare/Unique                   
qzxbjk               1.0000       Very Rare/Unique                   
```

#### Detailed Output (with Confidence)
```bash
uv run src/tiny_icf/predict.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone qzxbjk" \
    --detailed
```

Output:
```
Word                 ICF Score    Confidence    Interpretation                    
--------------------------------------------------------------------------------
the                  0.0000       0.8234        Very Common (stopword-like)        
xylophone            1.0000       0.9123        Very Rare/Unique                   
qzxbjk               1.0000       0.9456        Very Rare/Unique                   
```

#### JSON Output
```bash
uv run src/tiny_icf/predict.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone qzxbjk" \
    --json
```

Output:
```json
[
  {
    "word": "the",
    "icf_score": 0.0,
    "interpretation": "Very Common (stopword-like)",
    "category": "very_common",
    "confidence": 0.8234,
    "raw_output": -0.0123
  },
  {
    "word": "xylophone",
    "icf_score": 1.0,
    "interpretation": "Very Rare/Unique",
    "category": "very_rare",
    "confidence": 0.9123,
    "raw_output": 1.0234
  },
  {
    "word": "qzxbjk",
    "icf_score": 1.0,
    "interpretation": "Very Rare/Unique",
    "category": "very_rare",
    "confidence": 0.9456,
    "raw_output": 1.0567
  }
]
```

### Enhanced Prediction Script

For batch processing and file output:

```bash
uv run src/tiny_icf/predict_enhanced.py \
    --model models/model_diagnostic_rank5.pt \
    --words "the xylophone qzxbjk unfriendliness flimjam" \
    --output predictions.json
```

## Model Forward Method

The model's `forward()` method now supports returning features:

```python
# Basic usage (returns score only)
score = model(byte_tensor)  # [Batch, 1]

# Enhanced usage (returns score + features)
score, features = model(byte_tensor, return_features=True)
# features contains:
#   - 'icf_score': [Batch, 1] clamped score
#   - 'raw_output': [Batch, 1] raw output before clamping
#   - 'feature_activations': [Batch, hidden_dim] hidden layer activations
#   - 'confidence': [Batch, 1] confidence estimate
```

## Return Value Structure

### Simple Prediction
- **Type**: `float`
- **Value**: ICF score (0.0-1.0)

### Detailed Prediction
- **Type**: `dict`
- **Keys**:
  - `'word'`: str - The input word
  - `'icf_score'`: float - ICF score (0.0=common, 1.0=rare)
  - `'interpretation'`: str - Human-readable category
  - `'category'`: str - One of: 'very_common', 'common', 'rare', 'very_rare'
  - `'confidence'`: float - Confidence estimate (0.0-1.0)
  - `'raw_output'`: float - Raw model output before clamping

## Category Thresholds

- **Very Common**: ICF < 0.2 (stopword-like)
- **Common**: 0.2 ≤ ICF < 0.5
- **Rare**: 0.5 ≤ ICF < 0.8
- **Very Rare**: ICF ≥ 0.8 (unique/gibberish)

## Confidence Estimate

The confidence is computed from the magnitude of feature activations:
- Higher activation magnitude → Higher confidence
- Normalized to [0, 1] range using sigmoid
- Rough estimate based on how "strong" the model's internal features are

## Examples

### Python API

```python
from tiny_icf.model import UniversalICF
from tiny_icf.predict import predict_icf
import torch

# Load model
model = UniversalICF()
model.load_state_dict(torch.load('models/model_diagnostic_rank5.pt'))
model.eval()

# Predict with details
words = ["the", "xylophone", "qzxbjk"]
for word in words:
    result = predict_icf(model, word, torch.device('cpu'), return_details=True)
    print(f"{result['word']}: {result['icf_score']:.4f} "
          f"({result['category']}, confidence={result['confidence']:.2f})")
```

### Batch Processing

```python
from tiny_icf.predict_enhanced import predict_batch

results = predict_batch(model, words, device, return_details=True)
for result in results:
    print(f"{result['word']}: {result['icf_score']:.4f} - {result['interpretation']}")
```

## Benefits

1. **Rich Information**: Get more than just a score
2. **Confidence Estimates**: Know how confident the model is
3. **Interpretation**: Automatic categorization
4. **Raw Output**: Access to unclamped values for debugging
5. **Feature Activations**: Access to internal representations

## Backward Compatibility

The basic API still works:
- `predict_icf(model, word, device)` returns `float` (backward compatible)
- `model(byte_tensor)` returns `Tensor` (backward compatible)
- Enhanced features are opt-in via `return_details=True` or `return_features=True`

