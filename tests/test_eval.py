"""Tests for evaluation metrics and utilities."""

import numpy as np
import pytest
import torch

from tiny_icf.eval import (
    compute_metrics,
    evaluate_ranking,
    evaluate_jabberwocky,
    evaluate_on_dataset,
)
from tiny_icf.model import UniversalICF
from tiny_icf.data import WordICFDataset


def test_compute_metrics_perfect():
    """Test metrics computation with perfect predictions."""
    predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    targets = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    metrics = compute_metrics(predictions, targets)
    
    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['spearman_corr'] == 1.0
    assert metrics['pearson_corr'] == 1.0


def test_compute_metrics_errors():
    """Test metrics computation with errors."""
    predictions = np.array([0.1, 0.4, 0.6, 0.8, 1.0])
    targets = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
    
    metrics = compute_metrics(predictions, targets)
    
    assert metrics['mae'] > 0.0
    assert metrics['rmse'] > 0.0
    assert metrics['mae'] < metrics['rmse']  # RMSE penalizes large errors more
    assert 0.0 <= metrics['spearman_corr'] <= 1.0


def test_compute_metrics_torch_tensors():
    """Test metrics computation with torch tensors."""
    predictions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    targets = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    
    metrics = compute_metrics(predictions, targets)
    
    assert metrics['mae'] == 0.0
    assert isinstance(metrics['mae'], float)


def test_evaluate_ranking():
    """Test ranking evaluation."""
    # Perfect ranking
    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    targets = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    
    metrics = evaluate_ranking(predictions, targets, top_k=5)
    
    assert metrics['precision_at_k'] == 1.0
    assert metrics['top_k_overlap'] == 5
    assert metrics['mean_rank_error'] == 0.0


def test_evaluate_ranking_imperfect():
    """Test ranking evaluation with imperfect ranking."""
    predictions = np.array([0.5, 0.9, 0.3, 0.7, 0.1, 0.8, 0.2, 0.6, 0.4, 0.0])
    targets = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    
    metrics = evaluate_ranking(predictions, targets, top_k=5)
    
    assert 0.0 <= metrics['precision_at_k'] <= 1.0
    assert metrics['top_k_overlap'] <= 5


def test_evaluate_jabberwocky():
    """Test Jabberwocky Protocol evaluation."""
    model = UniversalICF()
    device = torch.device("cpu")
    
    # Use custom test cases
    test_cases = [
        ("the", 0.0, 0.5, "Common word"),  # Relaxed for untrained model
        ("xylophone", 0.0, 1.0, "Rare word"),  # Relaxed
    ]
    
    results = evaluate_jabberwocky(model, device, test_cases)
    
    assert 'pass_rate' in results
    assert 'results' in results
    assert len(results['results']) == 2
    assert 0.0 <= results['pass_rate'] <= 1.0


def test_evaluate_on_dataset():
    """Test dataset evaluation."""
    # Create small dummy dataset
    pairs = [
        ("the", 0.1),
        ("apple", 0.3),
        ("xylophone", 0.8),
        ("qzxbjk", 0.99),
    ]
    
    dataset = WordICFDataset(pairs, max_length=20)
    model = UniversalICF()
    device = torch.device("cpu")
    
    results = evaluate_on_dataset(model, dataset, device, max_samples=4)
    
    assert 'metrics' in results
    assert 'ranking_metrics' in results
    assert 'predictions' in results
    assert 'targets' in results
    assert len(results['predictions']) == 4
    assert 'mae' in results['metrics']
    assert 'spearman_corr' in results['metrics']


def test_metrics_range_validation():
    """Test that metrics handle edge cases."""
    # All zeros
    predictions = np.zeros(10)
    targets = np.zeros(10)
    metrics = compute_metrics(predictions, targets)
    assert metrics['mae'] == 0.0
    
    # All ones
    predictions = np.ones(10)
    targets = np.ones(10)
    metrics = compute_metrics(predictions, targets)
    assert metrics['mae'] == 0.0
    
    # Constant predictions
    predictions = np.full(10, 0.5)
    targets = np.linspace(0, 1, 10)
    metrics = compute_metrics(predictions, targets)
    assert metrics['mae'] > 0.0
    assert metrics['pred_std'] == 0.0  # Constant predictions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

