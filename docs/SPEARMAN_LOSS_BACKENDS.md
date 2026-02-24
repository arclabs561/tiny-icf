# Spearman Loss Backend Options

## Overview

The Spearman loss implementation now supports multiple backends for differentiable sorting/ranking, automatically selecting the best available option.

## Available Backends

### 1. **torchsort** (Recommended) ⭐
- **Source**: PyTorch implementation of [fast-soft-sort](https://github.com/google-research/fast-soft-sort) (Google Research)
- **Complexity**: O(n log n) - fastest option
- **Performance**: Compiled C++/CUDA kernels, minimal overhead
- **Installation**: `pip install torchsort` or `uv pip install torchsort`
- **Best for**: Production training, large batches, maximum performance

### 2. **diffsort**
- **Source**: [Differentiable Sorting Networks](https://github.com/Felix-Petersen/diffsort) (Felix Petersen, ICLR 2022). Implements Petersen et al.–style differentiable sorting networks (relaxed pairwise comparators); see arXiv 2105.04019, 2203.09630.
- **Complexity**: O(n²(log n)²) - slower but more structured
- **Performance**: Python-based, more interpretable
- **Installation**: `pip install diffsort` (default dependency when torchsort unavailable)
- **Best for**: Research, or when torchsort is not installed (e.g. ABI issues)

### 3. **built-in** (Fallback)
- **Source**: Custom implementation using sigmoid-based soft ranking
- **Complexity**: O(n²) - simple but slower
- **Performance**: Pure PyTorch, no dependencies
- **Installation**: No installation needed (always available)
- **Best for**: Environments where external libraries can't be installed

## Automatic Backend Selection

The `SpearmanLoss` class automatically selects the best available backend:

```python
from tiny_icf.loss import SpearmanLoss

# Auto-selects: torchsort -> diffsort -> built-in
loss = SpearmanLoss(regularization_strength=1.0, backend='auto')

# Check which backend was selected
info = loss.get_backend_info()
print(info['current'])  # 'torchsort', 'diffsort', or 'built-in'
```

## Manual Backend Selection

You can also explicitly choose a backend:

```python
# Force torchsort (will error if not installed)
loss = SpearmanLoss(backend='torchsort', regularization_strength=1.0)

# Force diffsort
loss = SpearmanLoss(backend='diffsort', steepness=5.0)

# Force built-in (always works)
loss = SpearmanLoss(backend='built-in', regularization_strength=0.1)
```

## Performance Comparison

Based on research and benchmarks:

| Backend | Complexity | Speed | GPU Support | Dependencies |
|---------|-----------|-------|-------------|--------------|
| **torchsort** | O(n log n) | ⭐⭐⭐⭐⭐ | ✅ Excellent | `torchsort` |
| **diffsort** | O(n²(log n)²) | ⭐⭐⭐ | ✅ Good | `diffsort` |
| **built-in** | O(n²) | ⭐⭐ | ✅ Good | None |

## Integration

The backend selection is transparent - all backends use the same interface:

```python
from tiny_icf.loss import CombinedLoss

# CombinedLoss automatically uses best available backend
loss_fn = CombinedLoss(
    use_spearman=True,
    spearman_weight=10.0,
    spearman_reg_strength=1.0,
)
# Will print: "✅ Spearman loss using torchsort (O(n log n), best performance)"
```

## Installation

### For Best Performance (Recommended)
```bash
pip install torchsort
# or
uv pip install torchsort
```

### Alternative
```bash
pip install diffsort
# or
uv pip install diffsort
```

Both are also available as optional dependencies:
```bash
uv pip install -e ".[sorting]"
```

## References

1. **fast-soft-sort**: Blondel et al. "Fast Differentiable Sorting and Ranking" (ICML 2020)
   - [GitHub](https://github.com/google-research/fast-soft-sort)
   - [torchsort (PyTorch implementation)](https://github.com/teddykoker/torchsort)

2. **diffsort**: Petersen et al. "Monotonic Differentiable Sorting Networks" (ICLR 2022)
   - [GitHub](https://github.com/Felix-Petersen/diffsort)

3. **difftopk**: Differentiable top-k operations
   - [GitHub](https://github.com/Felix-Petersen/difftopk)

## Migration Notes

- **No breaking changes**: Existing code continues to work
- **Automatic upgrade**: If `torchsort` is installed, it will be used automatically
- **Backward compatible**: Falls back to built-in if no libraries are available
- **Performance boost**: Installing `torchsort` provides ~10-100x speedup for large batches

