# Knowledge Distillation from Language Models

## Overview

We're implementing knowledge distillation to transfer linguistic knowledge from pre-trained language models (LMs) into our character-level CNN model. This should significantly improve ICF prediction by leveraging the semantic understanding that LMs have learned from massive text corpora.

## Why Distillation?

1. **Semantic Knowledge Transfer**: LMs understand word relationships, morphology, and linguistic patterns that our character-level model must learn from scratch
2. **Better Generalization**: Teacher model's predictions provide richer training signal than just ground-truth labels
3. **Maintain Efficiency**: Student model stays small (<50k params) while benefiting from teacher's knowledge
4. **Multi-Granularity Learning**: Bridge word-level (teacher) and character-level (student) representations

## Architecture

### Teacher Models

We support two types of teacher models:

1. **Sentence-Transformers** (Recommended for speed)
   - `all-MiniLM-L6-v2`: 22.7M params, 384-dim embeddings, ~5ms/word
   - Lightweight, fast inference
   - Good for feature-based distillation

2. **HuggingFace Transformers** (More powerful)
   - `bert-base-uncased`: 110M params, 768-dim embeddings
   - `roberta-base`: 125M params, 768-dim embeddings
   - Better semantic understanding, slower inference
   - Good for output-based distillation

### Student Model

Our existing character-level CNN (`UniversalICF`):
- Processes UTF-8 bytes directly
- ~33k parameters
- Fast inference (<1ms/word)

### Distillation Framework

```
Teacher (LM)                    Student (CNN)
    |                               |
    |  Word: "the"                  |  Bytes: [116, 104, 101, ...]
    |                               |
    |  Embedding: [0.1, 0.2, ...]  |  Features: [0.3, 0.1, ...]
    |                               |
    |  ICF Prediction: 0.05         |  ICF Prediction: 0.12
    |                               |
    └─────────── Distillation Loss ─┘
```

## Distillation Methods

### 1. Output-Based Distillation (Primary)

**Soft Targets**: Teacher's ICF predictions (softened with temperature) guide student learning.

**Loss Function**:
```
L_total = (1-α) * L_supervised + α * L_distillation
```

Where:
- `L_supervised`: MSE(student, ground_truth)
- `L_distillation`: MSE(student_soft, teacher_soft) * T²
- `α`: Weight for distillation (typically 0.3-0.7)
- `T`: Temperature (typically 3.0-5.0)

**Benefits**:
- Simple to implement
- Works with any teacher model
- Provides rich training signal

### 2. Feature-Based Distillation (Optional)

**Feature Alignment**: Align student's intermediate features with teacher's embeddings.

**Loss Function**:
```
L_feature = 1 - cosine_similarity(student_features, teacher_features)
```

**Benefits**:
- Transfers internal representations
- Helps student learn similar feature spaces
- Can improve generalization

### 3. Similarity-Preserving Distillation (Future)

**Relational Knowledge**: Preserve word-to-word relationships learned by teacher.

**Implementation**: Ensure that if two words are similar in teacher space, they're similar in student space.

## Implementation Details

### Teacher Model Selection

For ICF prediction, we recommend:

1. **Start with sentence-transformers** (`all-MiniLM-L6-v2`):
   - Fast inference
   - Good embeddings for feature distillation
   - Can add ICF prediction head if needed

2. **Upgrade to BERT/RoBERTa** if needed:
   - Better semantic understanding
   - More accurate soft targets
   - Slower but more powerful

### Training Strategy

**Phase 1: Warm-up (Epochs 1-10)**
- Train student on ground-truth only
- Establish baseline performance

**Phase 2: Distillation (Epochs 11-50)**
- Gradually increase distillation weight (α: 0.1 → 0.5)
- Use teacher's soft targets
- Monitor both supervised and distillation losses

**Phase 3: Fine-tuning (Epochs 51+)**
- Reduce distillation weight (α: 0.5 → 0.1)
- Focus on ground-truth alignment
- Ensure student doesn't overfit to teacher

### Hyperparameters

**Temperature**: 3.0-5.0
- Higher = softer targets = more information
- Lower = sharper targets = less information

**Distillation Weight (α)**: 0.3-0.7
- Higher = more reliance on teacher
- Lower = more reliance on ground-truth

**Feature Alignment Weight (β)**: 0.1-0.3
- Only if using feature-based distillation
- Balances output vs feature alignment

## Expected Benefits

1. **Improved Spearman Correlation**: 0.17 → 0.25-0.30 (target)
2. **Better Generalization**: Student learns linguistic patterns from teacher
3. **Faster Convergence**: Teacher provides better initialization signal
4. **Maintained Efficiency**: Student stays small and fast

## Integration with Existing Training

The distillation framework integrates seamlessly with:
- ✅ Existing `FlexibleIDFLightningModule`
- ✅ Multi-task learning (can distill from multiple teachers)
- ✅ Unified loss framework
- ✅ Experiment tracking (Aim, TensorBoard)

## Usage Example

```python
from tiny_icf.distillation import LanguageModelTeacher, DistillationLoss, DistilledICFModel
from tiny_icf.model import UniversalICF

# Create teacher model
teacher = LanguageModelTeacher(
    model_name="all-MiniLM-L6-v2",
    model_type="sentence-transformers",
    use_word_frequency_head=False,  # Use embeddings as ICF proxy
)

# Create student model
student = UniversalICF()

# Wrap with distillation
distilled_model = DistilledICFModel(
    student_model=student,
    teacher_model=teacher,
)

# Create distillation loss
distillation_loss = DistillationLoss(
    temperature=3.0,
    alpha=0.5,  # 50% distillation, 50% supervised
    beta=0.1,   # 10% feature alignment
    use_feature_distillation=True,
)

# Training loop
for batch in dataloader:
    byte_tensors, icf_targets, words = batch
    
    # Forward pass
    outputs = distilled_model(
        byte_tensors=byte_tensors,
        words=words,
        return_features=True,
    )
    
    # Compute loss
    loss, components = distillation_loss(
        student_predictions=outputs['student_predictions'],
        teacher_predictions=outputs['teacher_predictions'],
        ground_truth=icf_targets,
        student_features=outputs.get('student_features'),
        teacher_features=outputs.get('teacher_features'),
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

## Next Steps

1. ✅ Implement distillation framework (`distillation.py`)
2. ⏳ Integrate into `FlexibleIDFLightningModule`
3. ⏳ Add distillation experiment configs
4. ⏳ Test with `all-MiniLM-L6-v2` teacher
5. ⏳ Compare with baseline (no distillation)
6. ⏳ Experiment with different teachers (BERT, RoBERTa)
7. ⏳ Tune hyperparameters (temperature, α, β)

## Research References

- **Knowledge Distillation**: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
- **BERTtoCNN**: Similar approach distilling BERT to CNN
- **Feature Distillation**: FitNets (Romero et al., 2015)
- **Similarity-Preserving**: Similarity-Preserving Knowledge Distillation (Tung & Mori, 2019)

