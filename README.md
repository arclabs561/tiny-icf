# Tiny ICF: Compressed Character-Level Word Frequency Estimation

A tiny neural network (<50k parameters) that predicts word frequency (ICF) from character patterns, enabling zero-shot token filtering and weighting without massive dictionaries.

## Quick Start

```bash
# Install dependencies
uv sync

# Train a model
uv run python -m tiny_icf.train --data data/word_frequency.csv

# Predict ICF scores
uv run python -m tiny_icf.predict --model models/model.pt --words "the xylophone qzxbjk"
```

## Features

- **Tiny**: <50k parameters (~160 KB)
- **Fast**: <1ms inference per word
- **Universal**: Works with any UTF-8 language
- **Generalizes**: Handles unseen words, typos, neologisms
- **Multi-task**: ICF prediction, language detection, temporal analysis, text reduction

## Documentation

See `docs/` for detailed documentation:
- `PROJECT_OVERVIEW.md` - What we're building and why
- `TECHNICAL_MATHEMATICAL_DESCRIPTION.md` - Mathematical formulation
- `INFORMATION_THEORETIC_CONSTRAINTS.md` - Kolmogorov complexity analysis
- `ALL_TASKS_STRUCTURE_ANALYSIS.md` - Analysis of all tasks

## Tasks

1. **ICF Prediction**: word → ICF score (0.0=common, 1.0=rare)
2. **Text Reduction**: Optimize word dropping to minimize embedding regret
3. **Temporal ICF**: Predict ICF across decades (1800s, 1900s, 2000s)
4. **Language Detection**: Detect language from character patterns
5. **Era Classification**: Classify historical era (archaic, modern, contemporary)
6. **Multi-Task**: Unified model for all tasks with AMOO

## License

MIT

