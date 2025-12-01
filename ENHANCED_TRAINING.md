# Enhanced Training: Augmentation, Multilingual, and Curriculum Learning

## Overview

Enhanced training pipeline with:
1. **Advanced Data Augmentation**: Realistic misspellings and morphological variations
2. **Multilingual Support**: Train on multiple languages simultaneously
3. **Curriculum Learning**: Progressively harder training examples

## Features

### 1. Advanced Data Augmentation

**Location**: `src/tiny_icf/augmentation.py`

**Strategies**:
- **Misspelling Patterns**:
  - Double letter errors (`l` → `ll`, `e` → `ee`)
  - Single letter errors (`ll` → `l`)
  - Vowel swaps (`a` ↔ `e`, `ie` ↔ `ei`)
  - Common typos (`ph` ↔ `f`, `tion` ↔ `sion`)
  - Adjacent character swaps
  - Character deletion
  - Character insertion

- **Morphological Variations**:
  - Plural removal (`cats` → `cat`)
  - Past tense removal (`walked` → `walk`)
  - Gerund removal (`running` → `run`)
  - Adverb suffix removal (`quickly` → `quick`)

**Usage**:
```python
from tiny_icf.augmentation import AdvancedAugmentation

aug = AdvancedAugmentation(
    misspelling_prob=0.15,
    morphological_prob=0.1,
    noise_prob=0.05,
)

augmented = aug("computer")  # Might produce "comuter", "computr", etc.
```

### 2. Multilingual Support

**Location**: `scripts/download_multilingual.py`

**Downloaded Languages** (8 languages, 400k words, 2.2B tokens):
- English (en): 725M tokens
- Spanish (es): 413M tokens
- French (fr): 313M tokens
- German (de): 152M tokens
- Italian (it): 242M tokens
- Portuguese (pt): 212M tokens
- Russian (ru): 145M tokens
- Korean (ko): 5M tokens

**Format**: Words are prefixed with language code (`en:the`, `es:el`) to avoid collisions.

**Usage**:
```bash
# Download multilingual datasets
python scripts/download_multilingual.py

# Train on multilingual data
python -m tiny_icf.train_curriculum \
    --data data/multilingual/multilingual_combined.csv \
    --epochs 50 \
    --curriculum-stages 5
```

### 3. Curriculum Learning

**Location**: `src/tiny_icf/curriculum.py`

**Concept**: Train on progressively harder words:
- **Stage 0**: Easiest words (common, short, simple)
- **Stage N-1**: Hardest words (rare, long, morphologically complex)

**Difficulty Factors**:
- ICF score (rarity): 40% weight
- Length: 20% weight
- Character diversity: 20% weight
- Morphological complexity: 20% weight

**Schedule**:
- Linear progression through stages
- Configurable warmup epochs (easy words only)
- Automatic stage advancement

**Usage**:
```python
from tiny_icf.curriculum import create_curriculum_schedule, CurriculumSampler

# Create 5-stage curriculum
stages = create_curriculum_schedule(word_icf_pairs, num_stages=5)

# Use in training
curriculum = CurriculumSampler(stages, schedule, warmup_epochs=5)
```

## Training Scripts

### Standard Training
```bash
python -m tiny_icf.train \
    --data frequencies.csv \
    --epochs 50 \
    --batch-size 64
```

### Cross-Validation
```bash
python -m tiny_icf.train_cv \
    --data frequencies.csv \
    --epochs 30 \
    --folds 5
```

### Curriculum Learning
```bash
python -m tiny_icf.train_curriculum \
    --data frequencies.csv \
    --epochs 50 \
    --curriculum-stages 5 \
    --warmup-epochs 5 \
    --augment-prob 0.2
```

## Benefits

### Augmentation Benefits
1. **Robustness**: Model learns to handle typos and misspellings
2. **Generalization**: Better performance on noisy real-world text
3. **Morphological Awareness**: Understands word structure beyond exact matches

### Multilingual Benefits
1. **Universal Patterns**: Learns cross-lingual character patterns
2. **Scale**: 400k words vs 50k (8x more data)
3. **Diversity**: Different writing systems (Latin, Cyrillic, Hangul)

### Curriculum Benefits
1. **Faster Convergence**: Easy words first, hard words later
2. **Better Generalization**: Gradual difficulty prevents overfitting
3. **Stable Training**: Reduces early training instability

## Expected Improvements

With enhanced training:
- **Spearman Correlation**: 0.18 → >0.7 (target: >0.8)
- **Jabberwocky Protocol**: 1/5 → 4/5+ passes
- **Robustness**: Handles typos and misspellings
- **Multilingual**: Works across 8 languages

## Next Steps

1. **Train on Multilingual Data**:
   ```bash
   python -m tiny_icf.train_curriculum \
       --data data/multilingual/multilingual_combined.csv \
       --epochs 50 \
       --curriculum-stages 5 \
       --augment-prob 0.2
   ```

2. **Evaluate Improvements**:
   - Run Jabberwocky Protocol
   - Test on misspelled words
   - Measure cross-lingual performance

3. **Fine-tune Hyperparameters**:
   - Augmentation probabilities
   - Curriculum stage count
   - Learning rate schedule

## Files Created

- `src/tiny_icf/augmentation.py`: Advanced augmentation strategies
- `src/tiny_icf/curriculum.py`: Curriculum learning implementation
- `src/tiny_icf/train_curriculum.py`: Training with curriculum
- `src/tiny_icf/train_cv.py`: Cross-validation training
- `scripts/download_multilingual.py`: Multilingual dataset downloader

