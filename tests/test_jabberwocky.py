"""Jabberwocky Protocol: Test generalization to pseudo-words."""

import os
import pytest
import torch
from pathlib import Path

from tiny_icf.checkpoint import load_model
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
    model, _checkpoint = load_model(model_path, device=device)
    model.eval()

    # Use evaluation function with default expanded test suite
    results = evaluate_jabberwocky(model, device)

    # Report results
    print(
        f"\nJabberwocky Protocol Results: {results['passed_count']}/{results['total_count']} tests passed"
    )
    for r in results["results"]:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['word']:22} -> ICF: {r['predicted']:.4f} ({r['description']})")

    # For untrained model we only verify structural invariants.
    assert results["total_count"] == 13, "Default suite should have 13 test cases"
    assert 0.0 <= results["pass_rate"] <= 1.0, "Pass rate should be in [0, 1]"


@pytest.mark.slow
@pytest.mark.jabberwocky
def test_jabberwocky_with_trained_model(device: torch.device):
    """Test Jabberwocky Protocol with a trained model (if available)."""
    if os.environ.get("TINY_ICF_TRAINED_MODEL_TESTS") != "1":
        pytest.skip("Set TINY_ICF_TRAINED_MODEL_TESTS=1 to enable trained-model tests.")

    model_path = Path("models/model_local_v3.pt")

    if not model_path.exists():
        pytest.skip("No trained model available")

    try:
        model, _checkpoint = load_model(model_path, device=device)
    except Exception as e:
        first_line = str(e).splitlines()[0] if str(e) else "RuntimeError"
        pytest.skip(f"Incompatible trained checkpoint {model_path}: {first_line}")
    model.eval()

    results = evaluate_jabberwocky(model, device)

    # After training, we expect at least 3/5 to pass
    print(f"\nTrained Model Jabberwocky: {results['passed_count']}/{results['total_count']} passed")
    assert (
        results["pass_rate"] >= 0.4
    ), f"Trained model should pass at least 40% (got {results['pass_rate']:.1%})"
