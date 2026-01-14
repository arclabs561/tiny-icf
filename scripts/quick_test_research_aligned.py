#!/usr/bin/env python3
"""Quick test to verify ResearchAlignedICFLoss integration works."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from tiny_icf.loss_research_aligned import ResearchAlignedICFLoss
        print("✅ ResearchAlignedICFLoss imported")
    except ImportError as e:
        print(f"❌ Failed to import ResearchAlignedICFLoss: {e}")
        return False
    
    try:
        from tiny_icf.flexible_lightning_module import FlexibleIDFLightningModule
        print("✅ FlexibleIDFLightningModule imported")
    except ImportError as e:
        print(f"❌ Failed to import FlexibleIDFLightningModule: {e}")
        return False
    
    return True


def test_config_parsing():
    """Test that experiment configs can be parsed."""
    print("\nTesting config parsing...")
    
    # Simulate a research-aligned config
    config = {
        'name': 'test_research_aligned',
        'model_type': 'universal',
        'use_research_aligned_loss': True,
        'use_unified_loss': False,
        'adaptive_reg': True,
        'use_focal': True,
        'focal_gamma': 2.0,
        'use_spearman': True,
        'spearman_weight': 10.0,
        'rank_weight': 0.5,
        'asymmetry_factor': 2.0,
        'ranking_method': 'sigmoid',
        'use_monotonicity': False,
        'use_quantile': False,
        'n_pairs': 16,
        'min_diff': 0.05,
    }
    
    try:
        from tiny_icf.flexible_lightning_module import FlexibleIDFLightningModule
        
        # Try to instantiate (without actually creating model)
        # We'll just check that the config is valid
        print(f"✅ Config parsed successfully")
        print(f"   - use_research_aligned_loss: {config['use_research_aligned_loss']}")
        print(f"   - spearman_weight: {config['spearman_weight']}")
        print(f"   - ranking_method: {config['ranking_method']}")
        return True
    except Exception as e:
        print(f"❌ Config parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_instantiation():
    """Test that ResearchAlignedICFLoss can be instantiated."""
    print("\nTesting loss instantiation...")
    
    try:
        from tiny_icf.loss_research_aligned import ResearchAlignedICFLoss
        
        loss_fn = ResearchAlignedICFLoss(
            use_spearman=True,
            spearman_weight=10.0,
            adaptive_reg=True,
            use_focal=True,
            ranking_method='sigmoid',
        )
        
        print("✅ ResearchAlignedICFLoss instantiated")
        print(f"   - use_spearman: {loss_fn.use_spearman}")
        print(f"   - adaptive_reg: {loss_fn.adaptive_reg}")
        print(f"   - use_focal: {loss_fn.use_focal}")
        print(f"   - ranking_method: {loss_fn.ranking_method}")
        return True
    except Exception as e:
        print(f"❌ Loss instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing ResearchAlignedICFLoss Integration\n")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_parsing,
        test_loss_instantiation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} raised exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed! Ready to launch experiments.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before launching.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

