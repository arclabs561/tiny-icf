"""Ranking evaluation metrics using rank-eval (NDCG, MAP, MRR).

This module integrates rank-eval for industry-standard ranking metrics.
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings

try:
    import rank_eval

    HAS_RANK_EVAL = True
except ImportError:
    HAS_RANK_EVAL = False
    warnings.warn(
        "rank-eval not available. Install with: pip install rank-eval or build from ../rank-eval/rank-eval-python"
    )


def compute_ranking_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
    use_graded: bool = True,
) -> Dict[str, float]:
    """
    Compute NDCG, MAP, MRR ranking metrics using rank-eval.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N] (higher = rarer)
        k_values: List of k values for NDCG@k
        use_graded: If True, use graded relevance (targets as relevance scores)
                    If False, use binary relevance (top-k targets as relevant)

    Returns:
        Dictionary of ranking metrics
    """
    if not HAS_RANK_EVAL:
        return {}

    predictions = predictions.flatten()
    targets = targets.flatten()

    metrics = {}

    if use_graded:
        # Graded relevance: use targets as relevance scores
        # For ICF: higher ICF = rarer = more relevant for ranking
        # Convert ICF to relevance: normalize to 0-3 scale for graded relevance
        # Common words (low ICF) = low relevance, rare words (high ICF) = high relevance

        # Normalize targets to 0-3 relevance scale
        min_target = targets.min()
        max_target = targets.max()
        if max_target > min_target:
            relevance_scores = ((targets - min_target) / (max_target - min_target) * 3.0).astype(
                int
            )
        else:
            relevance_scores = np.zeros_like(targets, dtype=int)

        # Create ranked list: (index, prediction_score)
        indices = np.arange(len(predictions))
        ranked: List[Tuple[str, float]] = [
            (str(i), float(pred)) for i, pred in zip(indices, predictions)
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)  # Sort by prediction descending

        # Create qrels: index -> relevance
        qrels = {str(i): int(rel) for i, rel in zip(indices, relevance_scores)}

        # Compute NDCG@k for each k
        for k in k_values:
            try:
                ndcg = rank_eval.compute_ndcg(ranked, qrels, k=k)
                metrics[f"ndcg@{k}"] = float(ndcg)
            except Exception as e:
                warnings.warn(f"Failed to compute NDCG@{k}: {e}")
                metrics[f"ndcg@{k}"] = 0.0

        # Compute MAP
        try:
            map_score = rank_eval.compute_map(ranked, qrels)
            metrics["map"] = float(map_score)
        except Exception as e:
            warnings.warn(f"Failed to compute MAP: {e}")
            metrics["map"] = 0.0

    else:
        # Binary relevance: top-k targets are considered relevant
        # For ICF: top-k rarest words (highest targets) are relevant

        # Get top-k indices by target (rarest words)
        top_k = max(k_values)
        top_k_indices = np.argsort(targets)[-top_k:][::-1]  # Descending
        relevant_set = {str(i) for i in top_k_indices}

        # Create ranked list by predictions
        indices = np.arange(len(predictions))
        ranked = [str(i) for i in np.argsort(predictions)[::-1]]  # Descending by prediction

        # Compute NDCG@k for each k
        for k in k_values:
            try:
                ndcg = rank_eval.ndcg_at_k(ranked, relevant_set, k=k)
                metrics[f"ndcg@{k}"] = float(ndcg)
            except Exception as e:
                warnings.warn(f"Failed to compute NDCG@{k}: {e}")
                metrics[f"ndcg@{k}"] = 0.0

        # Compute MAP (average precision)
        try:
            map_score = rank_eval.average_precision(ranked, relevant_set)
            metrics["map"] = float(map_score)
        except Exception as e:
            warnings.warn(f"Failed to compute MAP: {e}")
            metrics["map"] = 0.0

        # Compute MRR
        try:
            mrr = rank_eval.mrr(ranked, relevant_set)
            metrics["mrr"] = float(mrr)
        except Exception as e:
            warnings.warn(f"Failed to compute MRR: {e}")
            metrics["mrr"] = 0.0

    return metrics


def compute_ranking_metrics_with_confidence(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    use_graded: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ranking metrics with bootstrap confidence intervals.

    Args:
        predictions: Model predictions [N]
        targets: Ground truth ICF scores [N]
        k_values: List of k values for NDCG@k
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        use_graded: Use graded or binary relevance

    Returns:
        Dictionary mapping metric name to {'value': float, 'ci_lower': float, 'ci_upper': float}
    """
    if not HAS_RANK_EVAL:
        return {}

    predictions = predictions.flatten()
    targets = targets.flatten()
    n = len(predictions)

    # Bootstrap sampling
    bootstrap_metrics: Dict[str, List[float]] = {f"ndcg@{k}": [] for k in k_values}
    bootstrap_metrics["map"] = []
    if not use_graded:
        bootstrap_metrics["mrr"] = []

    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        pred_sample = predictions[indices]
        target_sample = targets[indices]

        # Compute metrics on bootstrap sample
        sample_metrics = compute_ranking_metrics(
            pred_sample, target_sample, k_values=k_values, use_graded=use_graded
        )

        for metric_name, value in sample_metrics.items():
            if metric_name in bootstrap_metrics:
                bootstrap_metrics[metric_name].append(value)

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
