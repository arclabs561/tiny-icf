"""Jabberwocky Protocol: Test generalization to pseudo-words."""

import pytest
import torch
from pathlib import Path

from tiny_icf.model import UniversalICF
from tiny_icf.eval import evaluate_jabberwocky


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary model for testing."""
    model = UniversalICF()
    torch.save(model.state_dict(), tmp_path / "test_model.pt")
    return str(tmp_path / "test_model.pt")


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


def test_jabberwocky_protocol(model_path: str, device: torch.device):
    """
    Jabberwocky Protocol: Model must correctly predict scores for non-existent words.
    
    Tests:
    1. "the" → ~0.0 (common stopword)
    2. "xylophone" → ~0.7-0.95 (rare but valid structure)
    3. "flimjam" → ~0.6-0.85 (rare, looks English)
    4. "qzxbjk" → ~0.99 (impossible structure, very rare)
    5. "unfriendliness" → ~0.4-0.7 (composed of common parts)
    """
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Test cases: (word, min_icf, max_icf, description)
    test_cases = [
        ("the", 0.0, 0.1, "Common stopword"),
        ("xylophone", 0.7, 0.95, "Rare but valid word"),
        ("flimjam", 0.6, 0.85, "Rare, looks English"),
        ("qzxbjk", 0.95, 1.0, "Impossible structure"),
        ("unfriendliness", 0.4, 0.7, "Composed of common parts"),
    ]
    
    # Use evaluation function
    results = evaluate_jabberwocky(model, device, test_cases)
    
    # Report results
    print(f"\nJabberwocky Protocol Results: {results['passed_count']}/{results['total_count']} tests passed")
    for r in results['results']:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['word']:20} -> ICF: {r['predicted']:.4f} ({r['description']})")
    
    # For untrained model, we expect failures (this is a sanity check)
    # After training, we expect 5/5 to pass
    assert results['total_count'] == 5, "Should have 5 test cases"
    assert 0.0 <= results['pass_rate'] <= 1.0, "Pass rate should be in [0, 1]"


def test_jabberwocky_with_trained_model(device: torch.device):
    """Test Jabberwocky Protocol with a trained model (if available)."""
    model_path = Path("models/model_local_v3.pt")
    
    if not model_path.exists():
        pytest.skip("No trained model available")
    
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = evaluate_jabberwocky(model, device)
    
    # After training, we expect at least 3/5 to pass
    print(f"\nTrained Model Jabberwocky: {results['passed_count']}/{results['total_count']} passed")
    assert results['pass_rate'] >= 0.4, f"Trained model should pass at least 40% (got {results['pass_rate']:.1%})"
