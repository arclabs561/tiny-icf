# Base Models and Transfer Learning Opportunities

## Available Base Models

### 1. Model Architectures

**UniversalICF** (`model.py`):
- Standard character-level CNN
- Byte-level embeddings (256 vocab)
- Parallel CNNs (kernels 3, 5, 7)
- Multi-scale pooling (max, mean, last)
- MLP head
- Optional attention mechanism
- ~50k parameters

**ResidualICF** (`model_residual.py`):
- Enhanced version with residual connections
- Pre-activation residuals in MLP head
- Kaiming initialization for ReLU
- Better gradient flow
- ~50k parameters (similar size)

**MultiTaskICF** (`model_multi_task.py`):
- Multi-task learning support
- Shared feature extraction
- Task-specific heads
- Can learn ICF + other tasks simultaneously

### 2. Knowledge Distillation Teachers

**ModernBERT** (`allenai/modernbert-base`):
- Modernized BERT architecture
- Better efficiency than original BERT
- Used as teacher for distillation

**Sentence-Transformers** (`all-MiniLM-L6-v2`):
- Optimized for semantic similarity
- Lightweight and efficient
- Good for feature distillation

### 3. Best Model Checkpoints

From our experiments, we have checkpoints from:

1. **residual_balanced** (0.1864 Spearman)
   - Best overall performer
   - ResidualICF architecture
   - Could be fine-tuned with new loss/config

2. **iter4_residual_distillation** (0.1875 Spearman)
   - Best distillation result
   - ResidualICF + ModernBERT distillation
   - Could be fine-tuned for longer training

3. **loss_ablation_balanced_hybrid** (0.1891 Spearman)
   - Best loss ablation result
   - Optimal loss configuration
   - Could be used as initialization

4. **residual_high_spearman** (0.1855 Spearman)
   - High-performing residual model
   - Good candidate for transfer

## Transfer Learning Opportunities

### 1. Fine-tuning from Best Checkpoints

**Current Problem**: We always train from scratch, wasting learned features.

**Solution**: Fine-tune from best checkpoints:
- Load weights from `residual_balanced` or `iter4_residual_distillation`
- Continue training with new loss/config
- Lower learning rate for fine-tuning (e.g., 1e-4 instead of 1e-3)
- Can break through ceiling by building on learned features

**Implementation**:
```python
# Load checkpoint
checkpoint = torch.load("models/residual_balanced/best.ckpt")
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune with lower LR
optimizer = AdamW(model.parameters(), lr=1e-4)  # Lower LR
```

### 2. Transfer Learning Between Architectures

**UniversalICF → ResidualICF**:
- Transfer convolutional layers (shared architecture)
- Initialize new residual connections
- Faster convergence than training from scratch

**ResidualICF → UniversalICF**:
- Transfer learned features
- Drop residual connections
- Useful if we want to simplify architecture

### 3. Knowledge Distillation

**Current**: Using ModernBERT as teacher
- Good for semantic understanding
- Helps break character-level limitations

**Additional Opportunities**:
- **Best model → Smaller model**: Compress best model
- **Ensemble → Single model**: Distill ensemble knowledge
- **Multi-teacher**: Combine ModernBERT + best model

### 4. Ensemble Methods

**Combine Multiple Best Models**:
- Average predictions from top 3-5 models
- Weighted ensemble based on validation performance
- Can exceed individual model performance

## Why This Matters for Breaking the Ceiling

The ~0.18-0.19 ceiling might be breakable with:

1. **Fine-tuning from best models**: 
   - Best models learned useful patterns
   - Fine-tuning can refine these patterns
   - Lower learning rate prevents catastrophic forgetting

2. **Transfer learning**:
   - Learned features transfer across architectures
   - Faster convergence = more experiments
   - Can test more configurations

3. **Ensemble**:
   - Multiple models capture different aspects
   - Ensemble can exceed individual limits
   - Simple averaging often works well

4. **Progressive training**:
   - Train base model → fine-tune with new loss
   - Train with distillation → fine-tune without teacher
   - Build on previous successes

## Implementation Plan

1. **Add checkpoint loading to training script**:
   - `init_from_checkpoint` parameter
   - Load weights before training
   - Support for partial weight transfer

2. **Create fine-tuning experiments**:
   - Fine-tune `residual_balanced` with `ResearchAlignedICFLoss`
   - Fine-tune `iter4_residual_distillation` for longer training
   - Fine-tune best model with new hyperparameters

3. **Transfer learning utilities**:
   - Function to transfer weights between architectures
   - Handle architecture mismatches gracefully
   - Support partial transfer (e.g., only conv layers)

4. **Ensemble experiments**:
   - Combine top 3-5 models
   - Weighted averaging
   - Stacking with meta-learner

