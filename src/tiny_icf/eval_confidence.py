"""Bootstrap confidence intervals for evaluation metrics.

This module provides statistical rigor to evaluation by computing
confidence intervals for all metrics using bootstrap resampling.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import spearmanr, pearsonr, kendalltau
import warnings


def compute_metrics_with_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary mapping metric name to {'value': float, 'ci_lower': float, 'ci_upper': float}
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    n = len(predictions)

    if n < 10:
        warnings.warn(f"Sample size {n} too small for bootstrap CI")
        return {}

    # Bootstrap sampling
    bootstrap_metrics: Dict[str, List[float]] = {
        "spearman_corr": [],
        "pearson_corr": [],
        "kendall_corr": [],
        "mae": [],
        "rmse": [],
    }

    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        pred_sample = predictions[indices]
        target_sample = targets[indices]

        # Compute metrics on bootstrap sample
        if np.std(pred_sample) > 0 and np.std(target_sample) > 0:
            try:
                spearman, _ = spearmanr(pred_sample, target_sample)
                if not np.isnan(spearman):
                    bootstrap_metrics["spearman_corr"].append(spearman)
            except Exception:
                pass

            try:
                pearson, _ = pearsonr(pred_sample, target_sample)
                if not np.isnan(pearson):
                    bootstrap_metrics["pearson_corr"].append(pearson)
            except Exception:
                pass

            try:
                kendall, _ = kendalltau(pred_sample, target_sample)
                if not np.isnan(kendall):
                    bootstrap_metrics["kendall_corr"].append(kendall)
            except Exception:
                pass

        # MAE and RMSE
        mae = np.mean(np.abs(pred_sample - target_sample))
        rmse = np.sqrt(np.mean((pred_sample - target_sample) ** 2))
        bootstrap_metrics["mae"].append(mae)
        bootstrap_metrics["rmse"].append(rmse)

    # Compute confidence intervals
    results = {}
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for metric_name, values in bootstrap_metrics.items():
        if len(values) == 0:
            continue

        values_array = np.array(values)
        results[metric_name] = {
            "value": float(np.mean(values_array)),
            "ci_lower": float(np.percentile(values_array, lower_percentile)),
            "ci_upper": float(np.percentile(values_array, upper_percentile)),
        }

    return results


def format_metric_with_ci(
    metric_dict: Dict[str, float],
    metric_name: str = "metric",
    decimals: int = 4,
) -> str:
    """
    Format a metric with confidence interval for display.

    Args:
        metric_dict: Dictionary with 'value', 'ci_lower', 'ci_upper'
        metric_name: Name of the metric
        decimals: Number of decimal places

    Returns:
        Formatted string like "0.1234 [0.1200, 0.1268]"
    """
    value = metric_dict.get("value", 0.0)
    ci_lower = metric_dict.get("ci_lower", value)
    ci_upper = metric_dict.get("ci_upper", value)

    return f"{value:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
