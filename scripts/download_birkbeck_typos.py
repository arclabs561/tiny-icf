"""Download Birkbeck misspelling corpus."""

import csv
import re
import sys
from pathlib import Path

import requests


def download_birkbeck_corpus(output_dir: Path):
    """Download Birkbeck misspelling corpus."""
    # Birkbeck corpus: http://www.dcs.bbk.ac.uk/~ROGER/corpora.html
    url = "http://www.dcs.bbk.ac.uk/~roger/missp.dat"
    output_path = output_dir / "birkbeck_typos.dat"
    
    try:
        print("Downloading Birkbeck misspelling corpus...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download Birkbeck corpus: {e}")
        return None


def parse_birkbeck_format(input_path: Path, output_path: Path):
    """Parse Birkbeck format and convert to CSV."""
    # Format: $word (correct spelling)
    #         misspelling1
    #         misspelling2
    #         ...
    
    typo_pairs = []
    current_word = None
    
    with open(input_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a correct word (starts with $)
            if line.startswith('$'):
                current_word = line[1:].strip().lower()
            elif current_word:
                # It's a misspelling of current_word
                typo = line.strip().lower()
                if typo and typo != current_word:
                    typo_pairs.append((typo, current_word))
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['typo', 'correction'])
        for typo, correction in sorted(set(typo_pairs)):
            writer.writerow([typo, correction])
    
    print(f"  Parsed {len(set(typo_pairs))} unique typo-correction pairs")
    return output_path


def analyze_typo_patterns(typo_csv: Path):
    """Analyze typo patterns to extract real distributions."""
    patterns = defaultdict(int)
    total = 0
    
    with open(typo_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            typo = row['typo']
            correction = row['correction']
            total += 1
            
            # Classify pattern
            if len(typo) == len(correction):
                # Same length: substitution or swap
                diffs = [(i, a, b) for i, (a, b) in enumerate(zip(typo, correction)) if a != b]
                if len(diffs) == 1:
                    # Single substitution
                    patterns['substitution'] += 1
                elif len(diffs) == 2 and abs(diffs[0][0] - diffs[1][0]) == 1:
                    # Adjacent swap
                    patterns['adjacent_swap'] += 1
                else:
                    patterns['other_swap'] += 1
            elif len(typo) < len(correction):
                # Character dropped
                patterns['char_drop'] += 1
            elif len(typo) > len(correction):
                # Character inserted
                patterns['char_insert'] += 1
    
    # Normalize to frequencies
    if total > 0:
        frequencies = {k: v / total for k, v in patterns.items()}
        print("\nReal Typo Pattern Frequencies:")
        for pattern, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {freq:.3f} ({patterns[pattern]} occurrences)")
        
        return frequencies
    
    return {}


def main():
    output_dir = Path("data/typos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Real Typo Corpus")
    print("=" * 80)
    
    dat_path = download_birkbeck_corpus(output_dir)
    
    if dat_path:
        csv_path = output_dir / "birkbeck_typos.csv"
        parse_birkbeck_format(dat_path, csv_path)
        frequencies = analyze_typo_patterns(csv_path)
        
        # Save frequencies for use in augmentation
        if frequencies:
            import json
            freq_path = output_dir / "typo_pattern_frequencies.json"
            with open(freq_path, 'w') as f:
                json.dump(frequencies, f, indent=2)
            print(f"\n✓ Saved pattern frequencies to {freq_path}")
            print(f"  Use this to update TYPO_PATTERN_FREQUENCIES in keyboard_augmentation.py")
    else:
        print("\n⚠️  Could not download typo corpus.")
        print("  You may need to manually download from:")
        print("  http://www.dcs.bbk.ac.uk/~ROGER/corpora.html")


if __name__ == "__main__":
    from collections import defaultdict
    main()

