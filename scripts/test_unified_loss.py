"""Test script for unified multi-task loss integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tiny_icf.flexible_lightning_module import FlexibleIDFLightningModule
from tiny_icf.model import UniversalICF


def test_icf_only_backward_compatible():
    """Test that ICF-only training still works (backward compatibility)."""
    print("=" * 70)
    print("Test 1: ICF-Only Training (Backward Compatible)")
    print("=" * 70)
    
    config = {
        'model_type': 'universal',
        'use_unified_loss': False,  # Use legacy CombinedLoss
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.01,
    }
    
    module = FlexibleIDFLightningModule(config, learning_rate=1e-3)
    
    # Create dummy batch (legacy format: tuple)
    batch_size = 8
    max_length = 20
    byte_tensors = torch.randint(0, 256, (batch_size, max_length))
    icf_targets = torch.rand(batch_size, 1) * 0.5 + 0.25  # ICF in [0.25, 0.75]
    
    batch = (byte_tensors, icf_targets)
    
    # Test training step
    loss = module.training_step(batch, 0)
    
    print(f"✅ ICF-only training step successful")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Using legacy CombinedLoss: {not module.use_unified_loss}")
    
    return True


def test_unified_loss_icf_only():
    """Test unified loss with ICF-only (no other tasks)."""
    print("\n" + "=" * 70)
    print("Test 2: Unified Loss with ICF-Only")
    print("=" * 70)
    
    try:
        from tiny_icf.loss_unified import UnifiedMultiTaskLoss
    except ImportError:
        print("⚠️  UnifiedMultiTaskLoss not available - skipping test")
        return False
    
    config = {
        'model_type': 'universal',
        'use_unified_loss': True,  # Use UnifiedMultiTaskLoss
        'icf_weight': 1.0,
        'text_reduction_weight': 0.0,  # Disable other tasks
        'temporal_weight': 0.0,
        'language_weight': 0.0,
        'era_weight': 0.0,
        'use_amoo': False,  # Fixed weights for simplicity
        'spearman_weight': 10.0,
    }
    
    try:
        module = FlexibleIDFLightningModule(config, learning_rate=1e-3)
        
        # Create dummy batch (legacy format: tuple)
        batch_size = 8
        max_length = 20
        byte_tensors = torch.randint(0, 256, (batch_size, max_length))
        icf_targets = torch.rand(batch_size, 1) * 0.5 + 0.25
        
        batch = (byte_tensors, icf_targets)
        
        # Test training step
        loss = module.training_step(batch, 0)
        
        print(f"✅ Unified loss (ICF-only) training step successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Using UnifiedMultiTaskLoss: {module.use_unified_loss}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_loss_multi_task():
    """Test unified loss with all tasks (if data available)."""
    print("\n" + "=" * 70)
    print("Test 3: Unified Loss with Multi-Task")
    print("=" * 70)
    
    try:
        from tiny_icf.loss_unified import UnifiedMultiTaskLoss
    except ImportError:
        print("⚠️  UnifiedMultiTaskLoss not available - skipping test")
        return False
    
    config = {
        'model_type': 'universal',
        'use_unified_loss': True,
        'icf_weight': 1.0,
        'text_reduction_weight': 0.5,
        'temporal_weight': 0.3,
        'language_weight': 0.2,
        'era_weight': 0.2,
        'use_amoo': False,
        'spearman_weight': 10.0,
    }
    
    try:
        module = FlexibleIDFLightningModule(config, learning_rate=1e-3)
        
        # Create multi-task batch (dict format)
        batch_size = 8
        max_length = 20
        byte_tensors = torch.randint(0, 256, (batch_size, max_length))
        icf_targets = torch.rand(batch_size, 1) * 0.5 + 0.25
        language_targets = torch.randint(0, 10, (batch_size,))  # 10 languages
        era_targets = torch.randint(0, 5, (batch_size,))  # 5 eras
        
        batch = {
            'byte_tensors': byte_tensors,
            'icf_targets': icf_targets,
            'language_targets': language_targets,
            'era_targets': era_targets,
        }
        
        # Test training step
        loss = module.training_step(batch, 0)
        
        print(f"✅ Multi-task training step successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Using UnifiedMultiTaskLoss: {module.use_unified_loss}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Unified Loss Integration")
    print("=" * 70)
    
    results = []
    
    # Test 1: Backward compatibility
    results.append(("ICF-Only (Backward Compatible)", test_icf_only_backward_compatible()))
    
    # Test 2: Unified loss ICF-only
    results.append(("Unified Loss ICF-Only", test_unified_loss_icf_only()))
    
    # Test 3: Unified loss multi-task
    results.append(("Unified Loss Multi-Task", test_unified_loss_multi_task()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print(f"\n{'✅ All tests passed!' if all_passed else '⚠️  Some tests failed or skipped'}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

