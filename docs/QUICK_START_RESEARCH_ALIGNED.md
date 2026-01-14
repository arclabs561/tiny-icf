# Quick Start: Research-Aligned Experiments

## Overview

Research-aligned experiments use `ResearchAlignedICFLoss` which incorporates:
- ✅ Adaptive regularization (matches data scale)
- ✅ Focal loss (hard example mining)
- ✅ Asymmetric penalties (common→rare worse)
- ✅ Multiple ranking methods (sigmoid, neural_sort, etc.)

## Quick Commands

### 1. Test Integration
```bash
uv run python scripts/quick_test_research_aligned.py
```

### 2. Launch All Experiments
```bash
./scripts/launch_research_aligned_experiments.sh [data_file.csv]
```

### 3. Launch Single Experiment
```bash
uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --experiments research_aligned_standard
```

### 4. Monitor Experiments
```bash
./scripts/monitor_research_aligned_experiments.sh
```

### 5. Compare with Baselines
```bash
uv run python scripts/compare_baseline_vs_research_aligned.py
```

### 6. Update Experiment Registry
```bash
uv run python scripts/create_experiment_registry.py
```

## Experiments

| Name | Migrated From | Key Features |
|------|---------------|--------------|
| `research_aligned_standard` | `standard_improved` | Main baseline comparison |
| `research_aligned_neural_sort` | - | NeuralSort ranking method |
| `research_aligned_high_spearman` | `multitask_icf_high_spearman_plateau` | High Spearman focus (15.0) |
| `research_aligned_strong_reg` | `multitask_icf_strong_reg` | Strong regularization |
| `research_aligned_residual` | `residual_listwise` | ResidualICF model |

## Expected Results

After training, compare:
- `research_aligned_standard` vs `standard_improved`
- `research_aligned_high_spearman` vs `multitask_icf_high_spearman_plateau`
- `research_aligned_strong_reg` vs `multitask_icf_strong_reg`
- `research_aligned_residual` vs `residual_listwise`

## Monitoring

### Check Logs
```bash
tail -f models/research_aligned_*/training.log
```

### Check Metrics
```bash
cat models/research_aligned_*/lightning_logs/version_0/metrics.csv
```

### Aim UI
```bash
aim up
# Navigate to: icf-training experiment
# Filter: research_aligned_*
```

## Troubleshooting

### Import Errors
If you see "rank-relax not available", the loss will fall back to built-in implementations. This is fine for testing, but install rank-relax for best performance:
```bash
cd ../rank-relax && maturin develop
```

### Training Fails
1. Check data file exists: `ls data/word_frequency.csv`
2. Check logs: `tail -20 models/research_aligned_*/training.log`
3. Verify config: `uv run python scripts/quick_test_research_aligned.py`

## Next Steps

1. ✅ Launch experiments
2. ✅ Monitor progress
3. ✅ Compare with baselines
4. ⏳ Tune hyperparameters if needed
5. ⏳ Try advanced features (monotonicity, quantile regression)

