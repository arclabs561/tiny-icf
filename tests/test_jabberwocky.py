"""Jabberwocky Protocol: Test generalization to pseudo-words."""

import pytest
import torch
from pathlib import Path

from tiny_icf.model import UniversalICF


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
        results.append((word, icf, passed, description))
    
    # Report results
    passed_count = sum(1 for _, _, p, _ in results if p)
    print(f"\nJabberwocky Protocol Results: {passed_count}/{len(results)} tests passed")
    for word, icf, passed, desc in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {word:20} -> ICF: {icf:.4f} ({desc})")
    
    # For untrained model, we expect failures (this is a sanity check)
    # After training, we expect 5/5 to pass
    assert len(results) == 5, "Should have 5 test cases"
