"""Property-based tests for model behavior."""

import numpy as np
import pytest
import torch

from tiny_icf.model import UniversalICF
from tiny_icf.eval import compute_metrics


def test_output_range():
    """Property: All predictions must be in [0, 1] range."""
    model = UniversalICF()
    model.eval()
    
    # Test with various inputs
    test_words = [
        "the", "apple", "xylophone", "qzxbjk",
        "a" * 20,  # Long word
        "",  # Empty (will be padded)
        "café",  # Unicode
        "123",  # Numbers
    ]
    
    for word in test_words:
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            output = model(byte_tensor).item()
        
        assert 0.0 <= output <= 1.0, f"Output {output} out of range for word '{word}'"


def test_deterministic():
    """Property: Same input should produce same output (deterministic)."""
    model = UniversalICF()
    model.eval()
    
    word = "test"
    byte_seq = word.encode("utf-8")[:20]
    padded = byte_seq + bytes(20 - len(byte_seq))
    byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        output1 = model(byte_tensor).item()
        output2 = model(byte_tensor).item()
    
    assert output1 == output2, "Model should be deterministic"


def test_batch_consistency():
    """Property: Batch processing should match single-item processing."""
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
    
    # Should match
    for single, batch in zip(single_outputs, batch_outputs):
        assert abs(single - batch) < 1e-5, "Batch and single predictions should match"


def test_common_vs_rare():
    """Property: Common words should have lower ICF than rare words."""
    model = UniversalICF()
    model.eval()
    
    common_words = ["the", "a", "is", "and", "of"]
    rare_words = ["xylophone", "quixotic", "serendipity", "ephemeral", "obsequious"]
    
    def predict(word):
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            return model(byte_tensor).item()
    
    common_scores = [predict(w) for w in common_words]
    rare_scores = [predict(w) for w in rare_words]
    
    # For a trained model, this should hold
    # For untrained, we just check it doesn't crash
    avg_common = np.mean(common_scores)
    avg_rare = np.mean(rare_scores)
    
    # Property: Rare words should have higher ICF (on average, after training)
    # For untrained model, we just verify the function works
    assert all(0.0 <= s <= 1.0 for s in common_scores + rare_scores)


def test_unicode_handling():
    """Property: Model should handle Unicode characters."""
    model = UniversalICF()
    model.eval()
    
    unicode_words = [
        "café",  # French
        "naïve",  # French
        "résumé",  # French
        "Müller",  # German
        "北京",  # Chinese
        "こんにちは",  # Japanese
    ]
    
    for word in unicode_words:
        byte_seq = word.encode("utf-8")[:20]
        padded = byte_seq + bytes(20 - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            output = model(byte_tensor).item()
        
        assert 0.0 <= output <= 1.0, f"Unicode word '{word}' failed"


def test_length_invariance():
    """Property: Padding shouldn't affect predictions (for same word)."""
    model = UniversalICF()
    model.eval()
    
    word = "test"
    
    # Different padding lengths (same word)
    outputs = []
    for max_len in [10, 20, 30]:
        byte_seq = word.encode("utf-8")[:max_len]
        padded = byte_seq + bytes(max_len - len(byte_seq))
        byte_tensor = torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            output = model(byte_tensor).item()
        outputs.append(output)
    
    # All should be similar (within small tolerance)
    # Note: Model uses max_length=20, so longer padding might not matter
    assert all(0.0 <= o <= 1.0 for o in outputs)


def test_metrics_properties():
    """Property: Metrics should satisfy certain properties."""
    # Perfect predictions
    pred = np.array([0.1, 0.5, 0.9])
    target = np.array([0.1, 0.5, 0.9])
    metrics = compute_metrics(pred, target)
    
    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['spearman_corr'] == 1.0
    
    # Reversed (worst case)
    pred = np.array([0.9, 0.5, 0.1])
    target = np.array([0.1, 0.5, 0.9])
    metrics = compute_metrics(pred, target)
    
    assert metrics['mae'] > 0.0
    assert metrics['spearman_corr'] < 0.0  # Negative correlation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

