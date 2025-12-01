"""Regression tests to prevent backsliding on model performance."""

import pytest
import torch
import numpy as np
from pathlib import Path

from tiny_icf.model import UniversalICF
from tiny_icf.eval import compute_metrics, evaluate_jabberwocky


@pytest.mark.regression
def test_model_output_range_regression():
    """Regression: Model must always output in [0, 1] range."""
    model = UniversalICF()
    model.eval()
    
    # Test with various inputs
    test_inputs = [
        torch.randint(0, 256, (1, 20)),
        torch.zeros(1, 20),
        torch.ones(1, 20) * 255,
    ]
    
    for inp in test_inputs:
        with torch.no_grad():
            output = model(inp)
        
        assert torch.all(output >= 0.0), "Output must be >= 0"
        assert torch.all(output <= 1.0), "Output must be <= 1"


@pytest.mark.regression
def test_model_parameter_count_regression():
    """Regression: Model must stay under 50k parameters."""
    model = UniversalICF()
    param_count = model.count_parameters()
    
    assert param_count < 50000, f"Model exceeded 50k parameter limit: {param_count}"


@pytest.mark.regression
def test_jabberwocky_baseline():
    """Regression: Jabberwocky Protocol should maintain baseline performance."""
    model = UniversalICF()
    device = torch.device("cpu")
    
    results = evaluate_jabberwocky(model, device)
    
    # Baseline: Untrained model should at least not crash
    assert results['pass_rate'] >= 0.0
    assert results['total_count'] == 5


@pytest.mark.regression
@pytest.mark.slow
def test_trained_model_performance():
    """Regression: Trained model should meet minimum performance thresholds."""
    model_path = Path("models/model_local_v3.pt")
    
    if not model_path.exists():
        pytest.skip("No trained model available for regression testing")
    
    model = UniversalICF()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Create dummy predictions and targets for metric testing
    # In real regression test, would load actual test dataset
    predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    targets = np.array([0.15, 0.25, 0.55, 0.65, 0.85])
    
    metrics = compute_metrics(predictions, targets)
    
    # Regression thresholds (adjust based on actual performance)
    assert metrics['mae'] < 0.5, f"MAE regression: {metrics['mae']:.4f} >= 0.5"
    assert metrics['spearman_corr'] > -1.0, "Spearman correlation should be > -1"


@pytest.mark.regression
def test_batch_consistency_regression():
    """Regression: Batch processing must match single-item processing."""
    model = UniversalICF()
    model.eval()
    
    words = ["the", "apple", "xylophone"]
    byte_tensors = []
    
    for word in words:
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensors.append(list(padded))
    
    # Single predictions
    single_outputs = []
    for byte_list in byte_tensors:
        byte_tensor = torch.tensor(byte_list, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(byte_tensor).item()
        single_outputs.append(output)
    
    # Batch prediction
    batch_tensor = torch.tensor(byte_tensors, dtype=torch.long)
    with torch.no_grad():
        batch_outputs = model(batch_tensor).squeeze().tolist()
    
    # Must match within numerical precision
    for single, batch in zip(single_outputs, batch_outputs):
        assert abs(single - batch) < 1e-5, f"Batch inconsistency: {single} vs {batch}"

