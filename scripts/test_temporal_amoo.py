# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "pandas>=2.0.0",
# ]
# ///
"""Quick test script for temporal AMOO training setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tiny_icf.model import UniversalICF
from tiny_icf.temporal_loss import AlignedMultiObjectiveLoss, temporal_icf_loss
from tiny_icf.loss import CombinedLoss
from tiny_icf.data_temporal import TemporalICFDataset


def test_amoo_loss():
    """Test AMOO loss function."""
    print("Testing AMOO loss...")
    
    amoo = AlignedMultiObjectiveLoss(
        objectives=['icf', 'temporal'],
        adaptive=True,
    )
    
    losses = {
        'icf': torch.tensor(0.5),
        'temporal': torch.tensor(0.3),
    }
    
    result = amoo(losses)
    print(f"  AMOO loss: {result.item():.4f}")
    assert result.item() > 0, "AMOO loss should be positive"
    print("  ✓ AMOO loss works")


def test_temporal_loss():
    """Test temporal loss function."""
    print("Testing temporal loss...")
    
    predictions = torch.tensor([[0.5], [0.7], [0.3]])
    targets = torch.tensor([[0.4], [0.6], [0.3]])
    
    temporal_targets = {
        '1800': torch.tensor([[0.6], [0.8], [0.4]]),
        '1900': torch.tensor([[0.5], [0.7], [0.3]]),
    }
    
    loss = temporal_icf_loss(
        predictions,
        targets,
        temporal_targets=temporal_targets,
        alpha=0.1,
    )
    
    print(f"  Temporal loss: {loss.item():.4f}")
    assert loss.item() > 0, "Temporal loss should be positive"
    print("  ✓ Temporal loss works")


def test_temporal_dataset():
    """Test temporal dataset creation."""
    print("Testing temporal dataset...")
    
    word_icf_pairs = [
        ('the', 0.0),
        ('computer', 0.4),
        ('selfie', 0.9),
    ]
    
    dataset = TemporalICFDataset(
        word_icf_pairs=word_icf_pairs,
        historical_data=None,
        decades=[1800, 1900, 2000],
    )
    
    print(f"  Dataset size: {len(dataset)}")
    assert len(dataset) == 3, "Dataset should have 3 samples"
    
    sample = dataset[0]
    assert 'word' in sample
    assert 'bytes' in sample
    assert 'icf' in sample
    print("  ✓ Temporal dataset works")


def test_model_compatibility():
    """Test model works with temporal setup."""
    print("Testing model compatibility...")
    
    model = UniversalICF()
    model.init_weights(mean_icf=0.4)
    
    # Create dummy input
    batch_size = 4
    max_length = 20
    dummy_input = torch.randint(0, 256, (batch_size, max_length))
    
    output = model(dummy_input)
    print(f"  Model output shape: {output.shape}")
    assert output.shape == (batch_size, 1), "Output should be [batch, 1]"
    
    # Test with return_features
    output, features = model(dummy_input, return_features=True)
    assert 'icf_score' in features
    assert 'confidence' in features
    print("  ✓ Model compatibility works")


def main():
    print("=" * 70)
    print("Testing Temporal AMOO Setup")
    print("=" * 70)
    print()
    
    try:
        test_amoo_loss()
        print()
        test_temporal_loss()
        print()
        test_temporal_dataset()
        print()
        test_model_compatibility()
        print()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

