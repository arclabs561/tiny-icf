"""Inference script for Universal ICF model."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tiny_icf.model import UniversalICF
from tiny_icf.language_detection import detect_languages, format_languages
from tiny_icf.temporal_detection import estimate_usage_period, format_temporal_analysis


def word_to_bytes(word: str, max_length: int = 20) -> torch.Tensor:
    """
    Convert word to byte tensor with character-boundary aware truncation.
    
    Truncates at character boundaries (not byte boundaries) to preserve UTF-8 validity.
    Similar to Rust's bstr approach - ensures multi-byte characters aren't corrupted.
    """
    import unicodedata
    
    # Normalize to NFC (canonical composition) for consistency
    word = unicodedata.normalize('NFC', word)
    
    # Truncate characters first (preserves UTF-8 validity)
    chars = list(word)[:max_length]
    byte_seq = ''.join(chars).encode("utf-8")
    
    # Truncate bytes if needed (multi-byte chars can exceed max_length)
    if len(byte_seq) > max_length:
        byte_seq = byte_seq[:max_length]
    
    # Pad to max_length bytes (may be < max_length if multi-byte chars)
    pad_length = max(0, max_length - len(byte_seq))
    padded = byte_seq + bytes(pad_length)
    return torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)


def predict_icf(
    model: UniversalICF, 
    word: str, 
    device: torch.device,
    return_details: bool = False,
    reference_scores: Optional[np.ndarray] = None,
) -> float | dict:
    """
    Predict ICF score for a single word.
    
    Args:
        model: Trained UniversalICF model
        word: Word to predict ICF for
        device: Device for computation
        return_details: If True, return dict with score, interpretation, confidence, etc.
    
    Returns:
        If return_details=False: float ICF score
        If return_details=True: dict with keys:
            - 'icf_score': float ICF score (0.0=common, 1.0=rare)
            - 'interpretation': str category (Very Common, Common, Rare, Very Rare)
            - 'confidence': float confidence estimate (0.0-1.0)
            - 'raw_output': float raw model output before clamping
            - 'category': str one of 'very_common', 'common', 'rare', 'very_rare'
    """
    model.eval()
    byte_tensor = word_to_bytes(word).to(device)
    
    with torch.no_grad():
        # Try to get features if model supports it
        try:
            if return_details:
                prediction, features = model(byte_tensor, return_features=True)
                # prediction is [1, 1] tensor, extract scalar
                icf_score = float(prediction.squeeze().item())
                raw_output = float(features.get('raw_output', prediction).squeeze().item())
                confidence = float(features.get('confidence', torch.tensor(0.5)).squeeze().item())
            else:
                prediction = model(byte_tensor)
                icf_score = float(prediction.squeeze().item())
                raw_output = None
                confidence = None
        except (TypeError, IndexError, AttributeError):
            # Model doesn't support return_features, use basic prediction
            prediction = model(byte_tensor)
            icf_score = float(prediction.squeeze().item())
            raw_output = icf_score
            confidence = 0.5  # Default confidence
    
    if not return_details:
        return icf_score
    
    # Determine interpretation
    if icf_score < 0.2:
        interpretation = "Very Common (stopword-like)"
        category = "very_common"
    elif icf_score < 0.5:
        interpretation = "Common"
        category = "common"
    elif icf_score < 0.8:
        interpretation = "Rare"
        category = "rare"
    else:
        interpretation = "Very Rare/Unique"
        category = "very_rare"
    
    result = {
        'icf_score': icf_score,
        'interpretation': interpretation,
        'category': category,
        'confidence': confidence if confidence is not None else 0.5,
        'raw_output': raw_output if raw_output is not None else icf_score,
        'word': word,
    }
    
    # Add percentile rank if reference scores provided
    if reference_scores is not None and len(reference_scores) > 0:
        percentile = (reference_scores <= icf_score).sum() / len(reference_scores) * 100.0
        result['percentile_rank'] = float(percentile)
    
    # Add language detection
    languages = detect_languages(word, method='combined')
    result['languages'] = format_languages(languages, top_k=3)
    
    # Add temporal/era detection
    temporal = estimate_usage_period(word, icf_score=icf_score)
    result['temporal'] = format_temporal_analysis(temporal)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict ICF scores for words")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Words to predict (space-separated string or single word)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--detailed", action="store_true", help="Return detailed predictions with confidence")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Parse words (handle both string and list)
    words = args.words.split() if isinstance(args.words, str) else args.words
    
    # Predict
    results = []
    for word in words:
        if args.detailed or args.json:
            result = predict_icf(model, word, device, return_details=True)
            results.append(result)
        else:
            score = predict_icf(model, word, device, return_details=False)
            result = {
                'word': word,
                'icf_score': score,
                'interpretation': (
                    "Very Common (stopword-like)" if score < 0.2 else
                    "Common" if score < 0.5 else
                    "Rare" if score < 0.8 else
                    "Very Rare/Unique"
                ),
            }
            results.append(result)
    
    # Output
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    elif args.detailed:
        print(f"{'Word':<20} {'ICF Score':<12} {'Confidence':<12} {'Interpretation':<30}")
        print("-" * 80)
        for result in results:
            print(f"{result['word']:<20} {result['icf_score']:<12.4f} {result['confidence']:<12.4f} {result['interpretation']:<30}")
    else:
        print(f"{'Word':<20} {'ICF Score':<12} {'Interpretation':<30}")
        print("-" * 80)
        for result in results:
            print(f"{result['word']:<20} {result['icf_score']:<12.4f} {result['interpretation']:<30}")


if __name__ == "__main__":
    main()

