#!/usr/bin/env -S uv run
"""Quick ICF prediction utility - interactive or batch mode."""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.model import UniversalICF
from tiny_icf.predict import word_to_bytes, predict_icf


def interactive_mode(model_path: Path, device: torch.device):
    """Interactive prediction mode."""
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("=" * 70)
    print("Interactive ICF Predictor")
    print("=" * 70)
    print("Enter words to predict ICF scores (one per line, or space-separated)")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 70)
    print()
    
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            
            if line.lower() in ('quit', 'exit', 'q'):
                break
            
            # Parse words (handle both single word and space-separated)
            words = line.split()
            
            print()
            print(f"{'Word':<25} {'ICF':<10} {'Interpretation':<30}")
            print("-" * 70)
            
            for word in words:
                score = predict_icf(model, word, device)
                
                if score < 0.2:
                    interpretation = "Very Common"
                elif score < 0.5:
                    interpretation = "Common"
                elif score < 0.8:
                    interpretation = "Rare"
                else:
                    interpretation = "Very Rare"
                
                print(f"{word:<25} {score:<10.4f} {interpretation:<30}")
            
            print()
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


def batch_mode(model_path: Path, words: list[str], device: torch.device, output_file: Path | None = None):
    """Batch prediction mode."""
    # Load model
    model = UniversalICF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = []
    
    for word in words:
        score = predict_icf(model, word, device)
        results.append((word, score))
    
    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write("word,icf_score\n")
            for word, score in results:
                f.write(f"{word},{score:.6f}\n")
        print(f"✓ Results saved to {output_file}")
    else:
        print(f"{'Word':<25} {'ICF Score':<12} {'Interpretation':<30}")
        print("-" * 70)
        for word, score in results:
            if score < 0.2:
                interpretation = "Very Common"
            elif score < 0.5:
                interpretation = "Common"
            elif score < 0.8:
                interpretation = "Rare"
            else:
                interpretation = "Very Rare"
            print(f"{word:<25} {score:<12.4f} {interpretation:<30}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick ICF prediction utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/quick_predict.py --model models/model.pt

  # Batch mode (command line)
  python scripts/quick_predict.py --model models/model.pt --words "the apple xylophone"

  # Batch mode (from file)
  python scripts/quick_predict.py --model models/model.pt --file words.txt

  # Save to CSV
  python scripts/quick_predict.py --model models/model.pt --words "the apple" --output results.csv
        """
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--words", type=str, help="Words to predict (space-separated)")
    parser.add_argument("--file", type=Path, help="File with words (one per line)")
    parser.add_argument("--output", type=Path, help="Output CSV file (batch mode only)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Check model exists
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    # Determine mode
    if args.interactive or (not args.words and not args.file):
        interactive_mode(args.model, device)
    else:
        # Batch mode - collect words
        words = []
        if args.words:
            words.extend(args.words.split())
        if args.file:
            if not args.file.exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            with open(args.file, 'r') as f:
                words.extend(line.strip() for line in f if line.strip())
        
        if not words:
            print("Error: No words provided", file=sys.stderr)
            sys.exit(1)
        
        batch_mode(args.model, words, device, args.output)


if __name__ == "__main__":
    main()

