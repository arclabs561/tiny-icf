# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Consolidated prediction interface with feature flags.

This consolidates predict.py, predict_enhanced.py, and predict_advanced.py
into a single interface with feature flags.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import from existing modules
from tiny_icf.predict import predict_icf, word_to_bytes


def predict(
    word: str,
    model: torch.nn.Module,
    device: torch.device,
    enhanced: bool = False,
    advanced: bool = False,
    max_length: int = 20,
) -> Dict[str, Any]:
    """
    Unified prediction interface with feature flags.
    
    Args:
        word: Input word
        model: Trained model
        device: Device for computation
        enhanced: If True, use enhanced prediction (interpretation, confidence)
        advanced: If True, use advanced prediction (all features)
        max_length: Maximum word length
    
    Returns:
        Dictionary with prediction results
    """
    # Use predict_icf from predict.py which handles all features
    if enhanced or advanced:
        # Get detailed prediction with all features
        result = predict_icf(model, word, device, return_details=True)
        # Rename 'icf_score' to 'icf' for consistency
        if 'icf_score' in result:
            result['icf'] = result.pop('icf_score')
    else:
        # Basic prediction only
        icf_score = predict_icf(model, word, device, return_details=False)
        result = {
            'word': word,
            'icf': float(icf_score),
        }
    
    return result


def predict_batch(
    words: List[str],
    model: torch.nn.Module,
    device: torch.device,
    enhanced: bool = False,
    advanced: bool = False,
    batch_size: int = 64,
    max_length: int = 20,
) -> List[Dict[str, Any]]:
    """
    Predict ICF for a batch of words.
    
    Args:
        words: List of words
        model: Trained model
        device: Device for computation
        enhanced: If True, use enhanced prediction
        advanced: If True, use advanced prediction
        batch_size: Batch size for processing
        max_length: Maximum word length
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    # Process in batches
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        
        # Prepare batch
        byte_tensors = []
        for word in batch_words:
            byte_seq = word.encode("utf-8")[:max_length]
            padded = byte_seq + bytes(max_length - len(byte_seq))
            byte_tensors.append(torch.tensor(list(padded), dtype=torch.long))
        
        byte_tensors = torch.stack(byte_tensors).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            icf_scores = model(byte_tensors).cpu().numpy()
        
        # Build results
        for word, icf in zip(batch_words, icf_scores):
            result = predict(word, model, device, enhanced, advanced, max_length)
            results.append(result)
    
    return results

