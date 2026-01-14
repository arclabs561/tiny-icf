# Isotonic Regret Text Reduction

## The Concept

**Remove words one at a time, tracking regret at each step, ensuring regret increases monotonically (isotonic property).**

### Why "Isotonic"?

**Isotonic** means "monotonically increasing" - in this context, regret should only increase (or stay the same) as we remove words. This ensures:
- Smooth progression (no oscillations)
- Predictable behavior (regret never decreases)
- Better optimization (we're making consistent progress)

## How It Works

### Algorithm

1. **Start with full sentence**
   - Compute original embedding
   - Predict ICF for all words

2. **Iteratively remove words**
   - Try dropping each word
   - Compute embedding regret for each candidate
   - Select word that causes **least regret** (weighted by ICF)
   - **Enforce isotonic property**: Only accept if regret ≥ previous regret

3. **Track progression**
   - Record: step, words remaining, regret, word removed, ICF removed
   - Build regret curve showing how regret increases

4. **Stop at target length**
   - Return reduced text, final regret, and full progression

### The Isotonic Property

```python
# Regret should only increase (or stay the same)
regret_curve = [0.0, 0.05, 0.12, 0.18, 0.25, ...]
# Each step: regret[i] >= regret[i-1] ✓

# Violation (not isotonic):
regret_curve = [0.0, 0.05, 0.12, 0.10, 0.25, ...]
# Step 3: regret decreased (0.12 → 0.10) ✗
```

### Why This Matters

**Without isotonic enforcement:**
- Regret can oscillate (increase, decrease, increase)
- Hard to track progress
- Unpredictable behavior

**With isotonic enforcement:**
- Regret only increases
- Smooth progression curve
- Predictable, consistent behavior
- Better for optimization

## Usage

### Basic Usage

```python
from tiny_icf.text_reduction_isotonic import reduce_text_isotonic

text = "the quick brown fox jumps over the lazy dog"
reduced, regret, stats = reduce_text_isotonic(
    text=text,
    icf_model=icf_model,
    target_ratio=0.5,  # Keep 50% of words
    enforce_isotonic=True,  # Ensure monotonic regret
    verbose=True,  # Print progress
)

print(f"Original: {text}")
print(f"Reduced: {reduced}")
print(f"Final regret: {regret:.4f}")
print(f"Isotonic: {stats['is_isotonic']}")
```

### Command Line

```bash
# Basic reduction
uv run scripts/demo_isotonic_reduction.py \
    --model models/model.pt \
    --text "the quick brown fox jumps over the lazy dog" \
    --target-ratio 0.5

# With verbose output (see each step)
uv run scripts/demo_isotonic_reduction.py \
    --model models/model.pt \
    --text "your text here" \
    --target-ratio 0.5 \
    --verbose

# Save results to JSON
uv run scripts/demo_isotonic_reduction.py \
    --model models/model.pt \
    --text "your text here" \
    --output results.json
```

## Output

### Progression Table

```
Step | Words | Regret | Δ Regret | Word Removed | ICF
-----|-------|--------|----------|--------------|-----
   0 |     9 | 0.0000 |   +0.0000 | (start)      | -
   1 |     8 | 0.0123 |   +0.0123 | the          | 0.05
   2 |     7 | 0.0234 |   +0.0111 | the          | 0.05
   3 |     6 | 0.0456 |   +0.0222 | over         | 0.30
   4 |     5 | 0.0678 |   +0.0222 | dog          | 0.30
```

### Regret Curve

```python
regret_curve = [0.0, 0.0123, 0.0234, 0.0456, 0.0678, ...]
# Monotonically increasing ✓
```

### Statistics

```python
stats = {
    'original_length': 9,
    'reduced_length': 5,
    'reduction_ratio': 0.44,  # 44% reduction
    'regret': 0.0678,
    'is_isotonic': True,  # Regret increased monotonically
    'steps': 4,
    'progression': [...],  # Full step-by-step history
    'regret_curve': [0.0, 0.0123, ...],
    'max_regret_increase': 0.0222,
    'min_regret_increase': 0.0111,
}
```

## Comparison with Other Methods

### Greedy ICF (Simple)
- **Method**: Sort by ICF, drop lowest first
- **Regret tracking**: Only final regret
- **Isotonic**: Not enforced
- **Speed**: Fast (O(N log N))

### Optimal Regret (Current)
- **Method**: Try each word, pick least regret
- **Regret tracking**: Only final regret
- **Isotonic**: Not enforced
- **Speed**: Slow (O(N²) embeddings)

### Isotonic Regret (New)
- **Method**: Try each word, pick least regret, enforce isotonic
- **Regret tracking**: Full progression curve
- **Isotonic**: Enforced (monotonic increase)
- **Speed**: Slow (O(N²) embeddings) but same as optimal

## Use Cases

### 1. **Text Summarization**
- Reduce length while preserving meaning
- Track how meaning degrades as words are removed
- Ensure smooth degradation (isotonic property)

### 2. **Token Filtering**
- Filter before expensive embedding computation
- See exactly which words to drop first
- Understand regret progression

### 3. **Compression Analysis**
- Understand trade-off between length and meaning
- Visualize regret curve
- Find optimal compression ratio

### 4. **RAG Optimization**
- Keep only informative tokens
- Track semantic preservation
- Ensure consistent behavior (isotonic)

## Advantages

1. **Full Progression Tracking**: See exactly what happens at each step
2. **Isotonic Property**: Ensures smooth, predictable progression
3. **Better Optimization**: Monotonic regret makes optimization easier
4. **Debugging**: Can see which words cause most regret
5. **Visualization**: Regret curve can be plotted

## Limitations

1. **Slower**: O(N²) embedding computations (same as optimal regret)
2. **Requires Embeddings**: Needs sentence-transformers
3. **Isotonic Enforcement**: May skip some good candidates if they violate isotonic property

## Future Enhancements

1. **Visualization**: Plot regret curve
2. **Batch Processing**: Process multiple texts
3. **Caching**: Cache embeddings for faster iteration
4. **Adaptive Isotonic**: Allow small decreases with penalty
5. **Multi-Objective**: Balance regret, ICF, and other factors

