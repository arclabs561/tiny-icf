# rank-fast vs rank-relax: Understanding the Difference

## The Key Distinction

- **`rank-fast`**: Non-differentiable, fast ranking for **inference**
- **`rank-relax`**: Differentiable ranking for **training** in Rust ML frameworks

## rank-fast: Inference-Time Fast Ranking

### What It Does

Fast, non-differentiable ranking and sorting operations for use **after model inference**.

### Concrete Example in Our ICF Project

```rust
use rank_fast::{rank, sort_by_score, top_k};

// After model inference - we have ICF scores
let words = vec!["the", "apple", "xylophone", "qzxbjk", "café"];
let scores = vec![0.05, 0.45, 0.85, 0.95, 0.60];  // From model inference

// Sort words by ICF score (common to rare)
let sorted = sort_by_score(&words, &scores);
// Result: ["the", "apple", "café", "xylophone", "qzxbjk"]

// Get top-k rarest words
let top_rare = top_k(&words, &scores, 2);
// Result: ["qzxbjk", "xylophone"] (top 2 rarest)

// Get ranks (1 = most common, 5 = rarest)
let ranks = rank(&scores);
// Result: [1, 2, 4, 5, 3] (ranks for each word)
```

### Use Cases

1. **Sorting predictions by score** - Order results for display
2. **Top-k selection** - Get top-k highest/lowest scoring items
3. **Ranking for APIs** - Return ranked results to users
4. **Comparison operations** - Compare scores without gradients
5. **Filtering** - Filter by rank thresholds

### Characteristics

- ❌ **No gradients needed** - Just regular sorting/ranking
- ❌ **No differentiability needed** - Not part of training
- ✅ **Fast** - Can use SIMD, optimized algorithms (quicksort, etc.)
- ✅ **Simple** - Just sorting/ranking, no ML complexity
- ✅ **Zero dependencies** - Pure Rust standard library

### When We'd Use It

- After model inference
- When displaying results to users
- When filtering/sorting predictions
- Any time we need ranking but don't need gradients

**Key point**: This is just regular sorting/ranking - no ML involved, just fast operations.

## rank-relax: Training-Time Differentiable Ranking

### What It Does

Differentiable ranking operations for use **during training** in Rust ML frameworks (candle/burn).

### Concrete Example in Candle

```rust
use rank_relax::spearman_loss;
use candle_core::{Tensor, Device, Var};

// During training loop
fn training_step(
    model: &Model,
    batch: &Batch,
    optimizer: &mut Optimizer,
) -> Result<f32> {
    // Forward pass
    let predictions: Tensor = model.forward(&batch.inputs)?;  // [batch_size]
    let targets: Tensor = batch.targets.clone();              // [batch_size]
    
    // Compute Spearman correlation loss (differentiable!)
    let loss = spearman_loss(&predictions, &targets, 1.0)?;
    
    // Backprop - gradients flow through ranking operation
    loss.backward()?;
    optimizer.step()?;
    
    Ok(loss.to_scalar::<f32>()?)
}
```

### Use Cases

1. **Loss functions** - Spearman correlation loss during training
2. **Gradient flow** - Backprop through ranking operations
3. **Training objectives** - Optimize ranking metrics directly
4. **Custom losses** - Any loss that depends on ordering

### Characteristics

- ✅ **Gradients needed** - Must integrate with autograd
- ✅ **Differentiability required** - Smooth relaxation of discrete operations
- ✅ **Integrates with autograd** - Works with candle/burn's autograd
- ✅ **Used during training** - Part of the training loop, not inference

### When We'd Use It

- During training in Rust ML frameworks (candle/burn)
- When optimizing ranking metrics (Spearman, NDCG)
- When we need gradients through ranking operations
- In loss functions that depend on ordering

**Key point**: This enables training with ranking objectives in Rust ML, just like we do in Python/PyTorch.

## Comparison Table

| Feature | rank-fast | rank-relax |
|---------|-----------|------------|
| **Purpose** | Inference-time ranking | Training-time ranking |
| **Differentiable** | ❌ No | ✅ Yes |
| **Gradients** | ❌ No | ✅ Yes |
| **Speed** | ⚡ Very fast | 🐢 Slower (smooth operations) |
| **Complexity** | Simple (regular sort) | Complex (smooth relaxation) |
| **Dependencies** | Zero (std only) | May need tensor libs |
| **Use case** | After inference | During training |
| **Example** | Sort results for display | Spearman loss in training |

## In Our Project

### Current State

- **Training**: Python/PyTorch (`loss_spearman.py`)
- **Inference**: Rust (`rust/src/main.rs`)

### With rank-relax

- **Training in Rust**: Use `rank-relax` with candle/burn for training
- **Inference in Rust**: Use `rank-fast` (or just std::sort) for ranking results

### Workflow

```
Training (Rust with candle/burn):
  Model → rank-relax::spearman_loss → Backprop → Optimize

Inference (Rust):
  Model → Predictions → rank-fast::sort_by_score → Display
```

## Conclusion

- **`rank-fast`**: Simple, fast ranking for inference (like `std::sort` but with ranking utilities)
- **`rank-relax`**: Complex, differentiable ranking for training (like `torchsort` but for Rust ML)

Both have their place, but they solve completely different problems:
- `rank-fast`: "Sort these results fast"
- `rank-relax`: "Train a model to optimize ranking"

