"""Comprehensive evaluation metrics and utilities for tiny-icf."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from scipy.stats import spearmanr, pearsonr, kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    assert len(predictions) == len(targets), "Predictions and targets must have same length"
    
    metrics = {}
    
    # Absolute errors
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    metrics['mae'] = float(np.mean(abs_errors))
    metrics['rmse'] = float(np.sqrt(np.mean(errors ** 2)))
    metrics['median_ae'] = float(np.median(abs_errors))
    metrics['max_ae'] = float(np.max(abs_errors))
    metrics['p95_ae'] = float(np.percentile(abs_errors, 95))
    
    # Relative errors (for non-zero targets)
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        rel_errors = abs_errors[non_zero_mask] / (targets[non_zero_mask] + 1e-8)
        metrics['mre'] = float(np.mean(rel_errors))  # Mean Relative Error
        metrics['median_re'] = float(np.median(rel_errors))
    else:
        metrics['mre'] = 0.0
        metrics['median_re'] = 0.0
    
    # Correlation metrics
    if HAS_SCIPY and len(predictions) > 1:
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        kendall_corr, kendall_p = kendalltau(predictions, targets)
        
        metrics['spearman_corr'] = float(spearman_corr)
        metrics['spearman_p'] = float(spearman_p)
        metrics['pearson_corr'] = float(pearson_corr)
        metrics['pearson_p'] = float(pearson_p)
        metrics['kendall_corr'] = float(kendall_corr)
        metrics['kendall_p'] = float(kendall_p)
    else:
        metrics['spearman_corr'] = 0.0
        metrics['spearman_p'] = 1.0
        metrics['pearson_corr'] = 0.0
        metrics['pearson_p'] = 1.0
        metrics['kendall_corr'] = 0.0
        metrics['kendall_p'] = 1.0
    
    # Prediction statistics
    metrics['pred_min'] = float(predictions.min())
    metrics['pred_max'] = float(predictions.max())
    metrics['pred_mean'] = float(predictions.mean())
    metrics['pred_std'] = float(predictions.std())
    metrics['pred_median'] = float(np.median(predictions))
    
    # Target statistics
    metrics['target_min'] = float(targets.min())
    metrics['target_max'] = float(targets.max())
    metrics['target_mean'] = float(targets.mean())
    metrics['target_std'] = float(targets.std())
    metrics['target_median'] = float(np.median(targets))
    
    # Distribution similarity
    metrics['mean_diff'] = float(abs(predictions.mean() - targets.mean()))
    metrics['std_ratio'] = float(predictions.std() / (targets.std() + 1e-8))
    
    # Range coverage
    metrics['range_coverage'] = float(
        (predictions.max() - predictions.min()) / (targets.max() - targets.min() + 1e-8)
    )
    
    # Calibration: Check if predictions are well-calibrated
    # Bin predictions and targets, compare means
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_errors = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_mean = predictions[mask].mean()
            target_mean = targets[mask].mean()
            calibration_errors.append(abs(pred_mean - target_mean))
    
    metrics['calibration_error'] = float(np.mean(calibration_errors)) if calibration_errors else 0.0
    
    return metrics


def evaluate_ranking(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate ranking quality (top-K accuracy, etc.).
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth values [N]
        top_k: Number of top items to consider
    
    Returns:
        Dictionary of ranking metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Sort by target (descending: highest ICF first)
    sorted_indices = np.argsort(targets)[::-1]
    top_k_indices_gt = sorted_indices[:top_k]
    
    # Sort by predictions (descending)
    sorted_indices_pred = np.argsort(predictions)[::-1]
    top_k_indices_pred = sorted_indices_pred[:top_k]
    
    # Overlap
    overlap = len(set(top_k_indices_gt) & set(top_k_indices_pred))
    precision_at_k = overlap / top_k
    
    # Ranking position errors
    position_errors = []
    for idx in top_k_indices_gt:
        if idx in sorted_indices_pred:
            pred_rank = np.where(sorted_indices_pred == idx)[0][0]
            gt_rank = np.where(sorted_indices == idx)[0][0]
            position_errors.append(abs(pred_rank - gt_rank))
    
    metrics = {
        'precision_at_k': float(precision_at_k),
        'top_k_overlap': int(overlap),
        'mean_rank_error': float(np.mean(position_errors)) if position_errors else 0.0,
        'median_rank_error': float(np.median(position_errors)) if position_errors else 0.0,
    }
    
    return metrics


def evaluate_jabberwocky(
    model: torch.nn.Module,
    device: torch.device,
    test_cases: Optional[List[Tuple[str, float, float, str]]] = None,
) -> Dict[str, any]:
    """
    Evaluate model on Jabberwocky Protocol (pseudo-words).
    
    Args:
        model: Trained model
        device: Device for computation
        test_cases: List of (word, min_icf, max_icf, description) tuples
    
    Returns:
        Dictionary with results and pass rate
    """
    if test_cases is None:
        test_cases = [
            ("the", 0.0, 0.1, "Common stopword"),
            ("xylophone", 0.7, 0.95, "Rare but valid word"),
            ("flimjam", 0.6, 0.85, "Rare, looks English"),
            ("qzxbjk", 0.95, 1.0, "Impossible structure"),
            ("unfriendliness", 0.4, 0.7, "Composed of common parts"),
        ]
    
    model.eval()
    results = []
    
    for word, min_icf, max_icf, description in test_cases:
        # Convert word to bytes
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            icf = model(byte_tensor).item()
        
        passed = min_icf <= icf <= max_icf
        results.append({
            'word': word,
            'predicted': icf,
            'min_icf': min_icf,
            'max_icf': max_icf,
            'passed': passed,
            'description': description,
        })
    
    passed_count = sum(1 for r in results if r['passed'])
    pass_rate = passed_count / len(results)
    
    return {
        'pass_rate': pass_rate,
        'passed_count': passed_count,
        'total_count': len(results),
        'results': results,
    }


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    batch_size: int = 64,
) -> Dict[str, any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device for computation
        max_samples: Maximum number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    all_predictions = []
    all_targets = []
    all_words = []
    
    sample_count = 0
    with torch.no_grad():
        for byte_tensors, icf_targets in dataloader:
            byte_tensors = byte_tensors.to(device)
            icf_targets = icf_targets.to(device)
            
            predictions = model(byte_tensors)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(icf_targets.cpu())
            
            # Decode words (approximate)
            for byte_tensor in byte_tensors.cpu():
                word_bytes = byte_tensor.numpy()
                word = bytes(word_bytes[word_bytes > 0]).decode('utf-8', errors='ignore')
                all_words.append(word)
            
            sample_count += len(byte_tensors)
            if max_samples and sample_count >= max_samples:
                break
    
    # Concatenate
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    ranking_metrics = evaluate_ranking(predictions, targets, top_k=10)
    
    return {
        'metrics': metrics,
        'ranking_metrics': ranking_metrics,
        'predictions': predictions,
        'targets': targets,
        'words': all_words[:len(predictions)],
        'n_samples': len(predictions),
    }

