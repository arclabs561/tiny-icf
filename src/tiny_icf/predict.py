"""Inference script for Universal ICF model."""

import argparse
from pathlib import Path

import torch

from tiny_icf.model import UniversalICF


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


def predict_icf(model: UniversalICF, word: str, device: torch.device) -> float:
    """Predict ICF score for a single word."""
    model.eval()
    byte_tensor = word_to_bytes(word).to(device)
    
    with torch.no_grad():
        prediction = model(byte_tensor)
        return prediction.item()


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
    print(f"{'Word':<20} {'ICF Score':<12} {'Interpretation':<30}")
    print("-" * 80)
    
    for word in words:
        score = predict_icf(model, word, device)
        if score < 0.2:
            interpretation = "Very Common (stopword-like)"
        elif score < 0.5:
            interpretation = "Common"
        elif score < 0.8:
            interpretation = "Rare"
        else:
            interpretation = "Very Rare/Unique"
        
        print(f"{word:<20} {score:<12.4f} {interpretation:<30}")


if __name__ == "__main__":
    main()

