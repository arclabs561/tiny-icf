# Loss Ablation Experiments

## Purpose

Systematic data science experiments to determine the optimal loss function configuration by varying individual components.

## Experimental Design

### Variables Tested

1. **Spearman Loss Weight**: 0, 1, 5, 10, 20, 50
2. **Ranking Loss Weight**: 0, 0.1, 0.5, 1.0, 2.0
3. **Focal Loss**: Enabled vs Disabled
4. **Monotonicity Constraints**: Enabled vs Disabled

### Experiments

| # | Name | Spearman | Ranking | Focal | Monotonicity | Description |
|---|------|----------|---------|-------|--------------|-------------|
| 1 | `loss_ablation_pure_spearman` | 10.0× | 0.0× | ✅ | ❌ | Pure Spearman optimization |
| 2 | `loss_ablation_pure_ranking` | ❌ | 1.0× | ✅ | ❌ | Pure ranking loss |
| 3 | `loss_ablation_balanced_hybrid` | 10.0× | 0.5× | ✅ | ❌ | Current baseline |
| 4 | `loss_ablation_high_spearman` | 20.0× | 0.5× | ✅ | ❌ | High Spearman focus |
| 5 | `loss_ablation_very_high_spearman` | 50.0× | 0.1× | ✅ | ❌ | Very high Spearman |
| 6 | `loss_ablation_high_ranking` | 5.0× | 2.0× | ✅ | ❌ | High ranking focus |
| 7 | `loss_ablation_no_focal` | 10.0× | 0.5× | ❌ | ❌ | No focal weighting |
| 8 | `loss_ablation_with_monotonicity` | 10.0× | 0.5× | ✅ | ✅ | + Monotonicity |
| 9 | `loss_ablation_low_spearman` | 5.0× | 1.0× | ✅ | ❌ | Low Spearman |
| 10 | `loss_ablation_equal_weights` | 1.0× | 1.0× | ✅ | ❌ | Equal weights |

## Launching Experiments

```bash
# Launch all experiments
./scripts/launch_loss_ablation_experiments.sh data/word_frequency.csv

# Or launch individually
uv run python ../trainctl/training/scripts/train_flexible_opportunistic.py \
    --data data/word_frequency.csv \
    --experiments loss_ablation_pure_spearman
```

## Analyzing Results

```bash
# Analyze all results
uv run python scripts/analyze_loss_ablation_results.py

# Compare with existing experiments
uv run python scripts/compare_all_experiments.py
```

## Expected Insights

1. **Is Spearman loss necessary?** Compare pure_spearman vs pure_ranking
2. **Optimal Spearman weight?** Compare 5×, 10×, 20×, 50×
3. **Optimal ranking weight?** Compare 0.5×, 1.0×, 2.0×
4. **Focal loss impact?** Compare with_focal vs no_focal
5. **Monotonicity impact?** Compare with_monotonicity vs baseline

## Success Criteria

- Best Spearman correlation > 0.20 (moderate correlation)
- Consistent improvement over baseline (balanced_hybrid)
- Clear winner among configurations

## Timeline

- **Launch**: All 10 experiments in parallel
- **Duration**: ~100 epochs each (shorter for faster iteration)
- **Analysis**: After all experiments complete or reach 50+ epochs

