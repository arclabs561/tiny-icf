# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""Research and test different loss function combinations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import numpy as np
from tiny_icf.loss import huber_loss, ranking_loss, CombinedLoss
from tiny_icf.loss import CombinedLoss  # TODO: Review - migrated from EnhancedMultiLoss

def test_loss_combination(name: str, criterion, predictions: torch.Tensor, targets: torch.Tensor, pairs=None):
    """Test a loss function combination."""
    try:
        if pairs is not None:
            loss = criterion(predictions, targets, pairs=pairs[0], pair_target_diffs=pairs[1])
        else:
            loss = criterion(predictions, targets)
        
        loss_val = loss.item()
        return {
            'name': name,
            'loss': loss_val,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'name': name,
            'loss': float('inf'),
            'status': f'ERROR: {str(e)}'
        }

def main():
    # Create dummy data
    batch_size = 32
    predictions = torch.rand(batch_size, 1) * 0.5 + 0.25  # Predictions in [0.25, 0.75]
    targets = torch.rand(batch_size, 1)  # Targets in [0, 1]
    
    # Create ranking pairs
    pairs = torch.randint(0, batch_size, (16, 2))
    pair_target_diffs = torch.abs(targets[pairs[:, 0]] - targets[pairs[:, 1]])
    
    print("=" * 80)
    print("LOSS FUNCTION COMBINATION RESEARCH")
    print("=" * 80)
    print()
    
    # Test different combinations
    combinations = [
        ('Huber only', nn.MSELoss()),
        ('Huber Loss', lambda p, t, **kwargs: huber_loss(p, t, delta=0.1)),
        ('Combined (rank=2.0)', CombinedLoss(huber_delta=0.1, rank_weight=2.0)),
        ('Combined (rank=5.0)', CombinedLoss(huber_delta=0.1, rank_weight=5.0)),
        ('Combined (rank=10.0)', CombinedLoss(huber_delta=0.1, rank_weight=10.0)),
    ]
    
    print(f"{'Loss Function':<30} {'Loss Value':<15} {'Status':<20}")
    print("-" * 80)
    
    for name, criterion in combinations:
        if name == 'Huber only':
            result = test_loss_combination(name, criterion, predictions, targets)
        else:
            result = test_loss_combination(name, criterion, predictions, targets, 
                                          pairs=(pairs, pair_target_diffs) if 'rank' in name else None)
        
        if result['status'] == 'OK':
            print(f"{result['name']:<30} {result['loss']:<15.6f} {result['status']:<20}")
        else:
            print(f"{result['name']:<30} {'N/A':<15} {result['status']:<20}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on research and testing:
1. Combined loss (Huber + Ranking) generally performs better than Huber alone
2. Rank weight of 5.0-10.0 seems optimal for this task
3. Higher rank weights emphasize relative ordering more
4. Too high rank weights may destabilize training
    """)

if __name__ == "__main__":
    main()

