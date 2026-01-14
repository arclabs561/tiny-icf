# PEP 723 Inline Script Metadata

## What is PEP 723?

PEP 723 allows embedding dependencies and Python version requirements directly in Python scripts using inline metadata blocks. This makes scripts self-contained and easier to run with `uv run`.

## Format

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
# ]
# ///
```

## Scripts with PEP 723 Metadata

The following scripts now include PEP 723 metadata:

### Training Scripts
- `scripts/train_research_loss.py` - Research-based loss training (NeuralNDCG, Softmax CE)
- `scripts/train_best_practices.py` - Unified best practices training
- `scripts/ablation_loss_study.py` - Loss function ablation study

### Evaluation Scripts
- `scripts/comprehensive_eval.py` - Comprehensive model evaluation

### Demo Scripts
- `scripts/demo_isotonic_reduction.py` - Isotonic regret text reduction demo

### Data Scripts
- `scripts/download_datasets_enhanced.py` - Enhanced dataset downloader

## Usage

With PEP 723 metadata, scripts can be run directly with `uv run`:

```bash
# In workspace: uv automatically reads inline metadata and installs dependencies
# The tiny_icf package is installed from the workspace
uv run scripts/train_research_loss.py --data data/word_frequency.csv --epochs 100

# Standalone: For scripts to work outside the workspace, install the package first
# uv pip install -e .  # Install tiny_icf in editable mode
uv run scripts/demo_isotonic_reduction.py --model models/model.pt --text "sample text"
```

**Note**: These scripts are part of the `tiny-icf` workspace. When run with `uv run` in the workspace, the `tiny_icf` package is automatically available. For standalone use, install the package first with `uv pip install -e .`.

## Benefits

1. **Self-contained scripts**: Dependencies are embedded, no separate config needed
2. **Easy sharing**: Scripts can be shared and run without project setup
3. **Version control**: Dependencies are version-controlled with the script
4. **uv integration**: Works seamlessly with `uv run`

## Dependencies Included

### Core Dependencies (Most Scripts)
- `torch>=2.0.0` - PyTorch
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `tqdm>=4.65.0` - Progress bars
- `scipy>=1.10.0` - Scientific computing (correlation metrics)

### Optional Dependencies
- `aim>=3.29.0` - Experiment tracking (training scripts)
- `sentence-transformers>=2.2.0` - Embeddings (text reduction scripts)
- `requests>=2.31.0` - HTTP requests (download scripts)

## Python Version

All scripts require Python >= 3.11 (matching project requirements).

## Notes

- Scripts still work with the project's `pyproject.toml` dependencies
- PEP 723 metadata is optional - scripts work without it too
- `uv run` automatically handles dependency resolution from inline metadata
- Other tools (PDM, Hatch, pipx) also support PEP 723

