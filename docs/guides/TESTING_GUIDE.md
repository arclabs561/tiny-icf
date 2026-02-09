# Testing and Evaluation Guide

## Test Structure

### Unit Tests
- `tests/test_model.py` - Model architecture and basic functionality
- `tests/test_eval.py` - Evaluation metrics and utilities
- `tests/test_properties.py` - Property-based tests
- `tests/test_jabberwocky.py` - Jabberwocky Protocol (generalization)
- `tests/test_regression.py` - Regression tests to prevent backsliding

### Shared Fixtures
- `tests/conftest.py` - Pytest fixtures (device, models, etc.)

## Running Tests

### All Tests
```bash
uv run pytest
```

### Specific Test File
```bash
uv run pytest tests/test_eval.py
```

### With Coverage
```bash
uv run pytest --cov=src/tiny_icf --cov-report=html
```

### Fast Tests Only (Skip Slow)
```bash
uv run pytest -m "not slow"
```

### Property Tests Only
```bash
uv run pytest -m property
```

## Evaluation Scripts

### Comprehensive Model Evaluation
```bash
python scripts/evaluate_model.py \
  --model models/model_local_v3.pt \
  --data data/word_frequency.csv \
  --output evaluation_results.json
```

### Jabberwocky Protocol Only
```bash
python scripts/evaluate_model.py \
  --model models/model_local_v3.pt \
  --jabberwocky-only
```

## Metrics Explained

### Absolute Errors
- **MAE** (Mean Absolute Error): Average absolute difference
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **Median AE**: Robust to outliers
- **Max AE**: Worst-case error
- **P95 AE**: 95th percentile error

### Correlation Metrics
- **Spearman**: Rank correlation (measures ordering)
- **Pearson**: Linear correlation
- **Kendall**: Rank correlation (alternative to Spearman)

### Ranking Metrics
- **Precision@K**: Fraction of top-K items correctly identified
- **Rank Error**: Position difference in rankings

### Calibration
- **Calibration Error**: How well predictions match targets in bins

## Property-Based Testing

Properties that should always hold:
1. **Output Range**: All predictions in [0, 1]
2. **Deterministic**: Same input → same output
3. **Batch Consistency**: Batch matches single-item processing
4. **Unicode Handling**: Handles all UTF-8 characters
5. **Length Invariance**: Padding doesn't affect predictions

## Jabberwocky Protocol

Tests generalization to pseudo-words:
- `"the"` → ~0.0 (common)
- `"xylophone"` → ~0.7-0.95 (rare but valid)
- `"flimjam"` → ~0.6-0.85 (rare, looks English)
- `"qzxbjk"` → ~0.95-1.0 (impossible structure)
- `"unfriendliness"` → ~0.4-0.7 (composed of common parts)

**Target**: 4/5 or 5/5 tests pass after training

## Regression Testing

Prevents performance backsliding:
- Model parameter count < 50k
- Output range [0, 1]
- Batch consistency
- Minimum performance thresholds

## Continuous Integration

Recommended CI setup:
```yaml
# .github/workflows/test.yml
- Run: uv run pytest
- Run: uv run pytest -m "not slow"  # Fast tests
- Run: python scripts/evaluate_model.py --model <model> --jabberwocky-only
```

## Test Coverage Goals

- **Unit Tests**: > 80% coverage
- **Property Tests**: All critical properties
- **Integration Tests**: End-to-end workflows
- **Regression Tests**: Prevent backsliding

## Adding New Tests

1. **Unit Test**: Add to appropriate `test_*.py` file
2. **Property Test**: Add to `test_properties.py`
3. **Regression Test**: Add to `test_regression.py`
4. **Integration Test**: Create `test_integration.py` if needed

## Evaluation Best Practices

1. **Always evaluate on held-out test set**
2. **Report multiple metrics** (not just MAE)
3. **Check calibration** (predictions match targets in bins)
4. **Test ranking quality** (ordering matters for ICF)
5. **Run Jabberwocky Protocol** (generalization test)

