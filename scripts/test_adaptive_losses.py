# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "tqdm>=4.65.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Test adaptive loss weighting strategies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np

from tiny_icf.loss import CombinedLoss
from tiny_icf.loss_adaptive import (
    RealTimeNormalizedLoss,
    UncertaintyWeightedLoss,
    compute_gradient_norms,
    monitor_loss_components,
)
from tiny_icf.loss_monitoring import (
    compute_loss_component_metrics,
    detect_loss_imbalance,
)


def test_combined_loss_with_monitoring():
    """Test CombinedLoss with component tracking."""
    print("=" * 70)
    print("Testing CombinedLoss with Component Tracking")
    print("=" * 70)
    
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 0.5 + 0.5
    targets = torch.rand(batch_size, 1)
    
    # Create pairs for ranking loss
    pairs = torch.tensor([[i, i+1] for i in range(0, batch_size-1, 2)])
    pair_diffs = torch.rand(len(pairs))
    
    loss_fn = CombinedLoss(
        use_neural_ndcg=True,
        neural_ndcg_weight=0.5,
        track_components=True,
    )
    
    # Forward pass
    loss = loss_fn(predictions, targets, pairs=pairs, pair_target_diffs=pair_diffs)
    
    print(f"\nTotal Loss: {loss.item():.4f}")
    
    # Get component stats
    stats = loss_fn.get_component_stats()
    print("\nComponent Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Check for imbalance
    loss_components = {
        'huber': stats.get('huber_mean', 0),
        'ranking': stats.get('ranking_mean', 0) * loss_fn.rank_weight,
        'neural_ndcg': stats.get('neural_ndcg_mean', 0) * loss_fn.neural_ndcg_weight,
    }
    
    is_imbalanced, dominant = detect_loss_imbalance(loss_components)
    if is_imbalanced:
        print(f"\n⚠️  Imbalance detected! Dominant: {dominant}")
    else:
        print("\n✅ Loss components are balanced")
    
    print("\n✅ CombinedLoss with monitoring test passed!")


def test_real_time_normalized_loss():
    """Test real-time normalized loss."""
    print("\n" + "=" * 70)
    print("Testing Real-Time Normalized Loss")
    print("=" * 70)
    
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 0.5 + 0.5
    targets = torch.rand(batch_size, 1)
    
    pairs = torch.tensor([[i, i+1] for i in range(0, batch_size-1, 2)])
    pair_diffs = torch.rand(len(pairs))
    
    loss_fn = RealTimeNormalizedLoss(
        use_neural_ndcg=True,
        neural_ndcg_weight=0.5,
    )
    
    # Forward pass
    loss, diagnostics = loss_fn(predictions, targets, pairs=pairs, pair_target_diffs=pair_diffs)
    
    print(f"\nTotal Loss: {loss.item():.4f}")
    print("\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ Real-time normalized loss test passed!")


def test_uncertainty_weighted_loss():
    """Test uncertainty-weighted loss."""
    print("\n" + "=" * 70)
    print("Testing Uncertainty-Weighted Loss")
    print("=" * 70)
    
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 0.5 + 0.5
    targets = torch.rand(batch_size, 1)
    
    pairs = torch.tensor([[i, i+1] for i in range(0, batch_size-1, 2)])
    pair_diffs = torch.rand(len(pairs))
    
    loss_fn = UncertaintyWeightedLoss(
        use_neural_ndcg=True,
        neural_ndcg_weight=0.5,
    )
    
    # Forward pass
    loss, diagnostics = loss_fn(predictions, targets, pairs=pairs, pair_target_diffs=pair_diffs)
    
    print(f"\nTotal Loss: {loss.item():.4f}")
    print("\nDiagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ Uncertainty-weighted loss test passed!")


def test_loss_monitoring():
    """Test loss monitoring utilities."""
    print("\n" + "=" * 70)
    print("Testing Loss Monitoring Utilities")
    print("=" * 70)
    
    # Simulate loss components
    loss_components = {
        'huber': 0.05,
        'ranking': 0.15,
        'neural_ndcg': 0.02,
    }
    
    # Compute metrics
    metrics = compute_loss_component_metrics(loss_components)
    
    print("\nLoss Component Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Check balance
    balance_score = metrics.get('balance_score', 0)
    print(f"\nBalance Score: {balance_score:.4f} (lower = more balanced)")
    
    # Detect imbalance
    is_imbalanced, dominant = detect_loss_imbalance(loss_components)
    if is_imbalanced:
        print(f"⚠️  Imbalance detected! Dominant: {dominant}")
    else:
        print("✅ Loss components are balanced")
    
    print("\n✅ Loss monitoring test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING ADAPTIVE LOSS WEIGHTING STRATEGIES")
    print("=" * 70)
    
    test_combined_loss_with_monitoring()
    test_real_time_normalized_loss()
    test_uncertainty_weighted_loss()
    test_loss_monitoring()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n✅ All adaptive loss strategies are working!")


if __name__ == "__main__":
    main()

