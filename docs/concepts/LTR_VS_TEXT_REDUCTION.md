# LTR for Training vs Text Reduction: The Confusion

## The Two Different Things

### 1. **LTR for Training the ICF Model** (What we're doing)
- **Purpose**: Train the model to predict word frequencies correctly
- **Goal**: Get relative ordering right (common < rare)
- **Metric**: Spearman correlation (ranking quality)
- **Why LTR**: The task is fundamentally about ranking words by frequency

### 2. **Text Reduction Application** (Downstream use case)
- **Purpose**: Use trained ICF model to reduce text length
- **Goal**: Drop words that minimize embedding regret
- **Metric**: Embedding regret, compression ratio
- **Why ICF**: Common words (low ICF) contribute less to semantics

## The Confusion

You might be thinking: "LTR is for text reduction, where we compare how much dropping words changes the ranking of similar words, and it should be p@1, right?"

**Actually, these are two separate things:**

### LTR is NOT for Text Reduction

**LTR (Learning-to-Rank) is used during TRAINING:**
- We use ranking losses to train the ICF model
- Goal: Make the model predict that "the" < "xylophone" in ICF space
- This ensures the model learns correct relative ordering

**Text Reduction is a DOWNSTREAM APPLICATION:**
- Uses the already-trained ICF model
- Drops words based on ICF scores
- Measures embedding regret (not ranking)

## What is p@1?

**p@1 (Precision at 1)** is a ranking evaluation metric:
- Measures: Is the top-ranked item correct?
- Used for: Evaluating ranking quality
- Example: In search, is the #1 result relevant?

**We do use p@1 (precision_at_k) in our evaluation:**
```python
# From src/tiny_icf/eval.py
precision_at_k = overlap / top_k
```

But this is for **evaluating the ICF model's ranking quality**, not for text reduction.

## The Actual Flow

### Step 1: Train ICF Model (Uses LTR)
```python
# Training with ranking loss
loss = huber_loss(pred, target) + ranking_loss(pred1, pred2)
# Goal: Learn that "the" < "xylophone" in ICF space
```

### Step 2: Use ICF Model for Text Reduction
```python
# Text reduction (doesn't use LTR)
icf_scores = [model.predict(word) for word in words]
# Sort by ICF, drop lowest ICF words
# Measure embedding regret
```

## Why the Confusion?

You might be thinking about:
1. **Text reduction evaluation**: "Does dropping words change the ranking of similar words?"
2. **p@1 metric**: "Is the top-ranked word correct?"

But these are different from:
- **LTR for training**: "Does the model learn correct word frequency ranking?"

## The Real Question

**Why do we use LTR for training the ICF model?**

Answer: Because the task is fundamentally about **relative ordering** (ranking), not exact values.

**Why do we use ICF for text reduction?**

Answer: Because common words (low ICF) contribute less to semantics, so dropping them minimizes embedding regret.

**These are two separate things that happen at different stages!**

## Summary

| Stage | What | Why LTR? | Why p@1? |
|-------|------|----------|----------|
| **Training** | Train ICF model | Task is about ranking words by frequency | Evaluate ranking quality |
| **Text Reduction** | Use ICF to drop words | Not using LTR - using ICF scores | Not using p@1 - using embedding regret |

**LTR is for training, not for text reduction!**

