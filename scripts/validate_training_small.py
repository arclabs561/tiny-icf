#!/usr/bin/env -S uv run
"""Small validation experiment to test training pipeline before full training."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import subprocess
import tempfile
import csv
from pathlib import Path


def create_small_test_data(source_file: Path, output_file: Path, n_lines: int = 1000):
    """Create a small test dataset from the first N lines of source file."""
    print(f"Creating small test dataset: {n_lines} words from {source_file}")
    
    with open(source_file, "r", encoding="utf-8") as f_in:
        with open(output_file, "w", encoding="utf-8") as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            
            # Write header if source has one, otherwise skip
            first_row = next(reader, None)
            if first_row and first_row[0].lower() in ['word', 'token', 'text']:
                # Skip header
                pass
            else:
                # First row is data, process it
                if first_row and len(first_row) >= 2:
                    try:
                        word = first_row[0].strip().lower()
                        count = int(first_row[1])
                        writer.writerow([word, count])
                    except (ValueError, IndexError):
                        pass
            
            # Copy remaining lines
            for i, row in enumerate(reader):
                if i >= n_lines - 1:  # Already wrote first row if it was data
                    break
                if len(row) >= 2:
                    try:
                        word = row[0].strip().lower()
                        count = int(row[1])
                        writer.writerow([word, count])
                    except (ValueError, IndexError):
                        continue
    
    print(f"Created test dataset: {output_file}")


def main():
    """Run small validation experiment."""
    project_root = Path(__file__).parent.parent
    source_data = project_root / "data" / "word_frequency.csv"
    
    if not source_data.exists():
        print(f"Error: Source data file not found: {source_data}")
        sys.exit(1)
    
    # Create temporary test data file
    test_data_file = project_root / "test_frequencies_small.csv"
    create_small_test_data(source_data, test_data_file, n_lines=1000)
    
    # Run training with minimal settings
    print("\n" + "="*60)
    print("Running small validation experiment")
    print("="*60)
    
    cmd = [
        sys.executable,
        "-m", "tiny_icf.train",
        "--data", str(test_data_file),
        "--epochs", "3",
        "--batch-size", "32",
        "--lr", "1e-3",
        "--output", str(project_root / "model_validation.pt"),
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ Validation experiment passed!")
        print("="*60)
        
        # Check if model was created
        model_file = project_root / "model_validation.pt"
        if model_file.exists():
            size = model_file.stat().st_size
            print(f"Model saved: {model_file} ({size:,} bytes)")
        
        # Clean up test data
        test_data_file.unlink()
        print(f"Cleaned up: {test_data_file}")
        
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Validation experiment failed!")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

