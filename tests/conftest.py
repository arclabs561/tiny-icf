"""Pytest configuration and shared fixtures."""

import pytest
import torch
from pathlib import Path

from tiny_icf.model import UniversalICF


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def untrained_model(device):
    """Create an untrained model for testing."""
    model = UniversalICF().to(device)
    model.eval()
    return model


@pytest.fixture
def trained_model_path():
    """Path to trained model (if available)."""
    paths = [
        Path("models/model_local_v3.pt"),
        Path("models/model_local_final.pt"),
        Path("models/model.pt"),
    ]
    
    for path in paths:
        if path.exists():
            return str(path)
    
    return None


@pytest.fixture
def trained_model(trained_model_path, device):
    """Load trained model if available."""
    if trained_model_path is None:
        pytest.skip("No trained model available")
    
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()
    return model

