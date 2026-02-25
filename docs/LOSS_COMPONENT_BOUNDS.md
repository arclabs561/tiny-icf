# Loss Component Bounds

Theoretical and empirically refined bounds for loss components used in `analyze_loss_bounds.py`.

## Bounds (refined from top performers)

| Component | Good | Poor | Best |
|-----------|------|------|------|
| huber | 0.08 | 0.20 | 0.05 |
| rank | 0.12 | 0.30 | 0.05 |
| spearman | 0.82 | 0.90 | 0.81 |
| asymmetric_penalty | 0.05 | 0.10 | 0.02 |
| monotonicity | 0.01 | 0.05 | 0.0 |
| quantile | 0.20 | 0.30 | 0.10 |

Values below "good" indicate healthy optimization; above "poor" suggests issues.
