"""Download real typo/misspelling datasets."""

import csv
import sys
from pathlib import Path

import requests


def download_github_typo_corpus(output_dir: Path):
    """Download GitHub Typo Corpus if available."""
    # GitHub Typo Corpus: https://github.com/sigtyp/ST2024?tab=readme-ov-file
    # Or search for other typo datasets
    
    sources = [
        {
            "name": "common_misspellings",
            "url": "https://raw.githubusercontent.com/mhagiwara/typo-corpus/master/data/common_misspellings.txt",
            "format": "typo_correction",
        },
        # Add more typo corpus sources
    ]
    
    downloaded = []
    for source in sources:
        try:
            output_path = output_dir / f"{source['name']}.txt"
            print(f"Downloading {source['name']}...")
            
            response = requests.get(source['url'], stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✓ Downloaded to {output_path}")
            downloaded.append((output_path, source['format']))
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    return downloaded


def convert_typo_corpus(input_path: Path, output_path: Path):
    """Convert typo corpus to CSV format: typo,correction."""
    typo_pairs = {}
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try different formats
            if '->' in line or '→' in line:
                parts = line.replace('→', '->').split('->')
            elif ',' in line:
                parts = line.split(',')
            elif '\t' in line:
                parts = line.split('\t')
            elif ' ' in line and len(line.split()) == 2:
                parts = line.split()
            else:
                continue
            
            if len(parts) >= 2:
                typo = parts[0].strip().lower()
                correction = parts[1].strip().lower()
                if typo and correction and typo != correction:
                    typo_pairs[typo] = correction
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['typo', 'correction'])
        for typo, correction in sorted(typo_pairs.items()):
            writer.writerow([typo, correction])
    
    print(f"  Converted {len(typo_pairs)} typo-correction pairs")
    return output_path


def analyze_typo_patterns(typo_csv: Path):
    """Analyze typo patterns to match real distributions."""
    patterns = {
        'adjacent_swap': 0,
        'char_drop': 0,
        'char_insert': 0,
        'vowel_swap': 0,
        'double_letter': 0,
        'single_letter': 0,
    }
    
    with open(typo_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            typo = row['typo']
            correction = row['correction']
            
            # Classify typo pattern
            if len(typo) == len(correction):
                # Same length: likely swap or substitution
                diffs = sum(1 for a, b in zip(typo, correction) if a != b)
                if diffs == 2:
                    patterns['adjacent_swap'] += 1
                elif any(a in 'aeiou' and b in 'aeiou' for a, b in zip(typo, correction)):
                    patterns['vowel_swap'] += 1
            elif len(typo) < len(correction):
                # Typo is shorter: character was dropped
                patterns['char_drop'] += 1
            elif len(typo) > len(correction):
                # Typo is longer: character was inserted
                patterns['char_insert'] += 1
    
    total = sum(patterns.values())
    if total > 0:
        print("\nTypo Pattern Distribution:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    return patterns


def main():
    output_dir = Path("data/typos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Typo/Misspelling Datasets")
    print("=" * 80)
    
    downloaded = download_github_typo_corpus(output_dir)
    
    if downloaded:
        for input_path, format_type in downloaded:
            csv_path = output_dir / f"{input_path.stem}.csv"
            convert_typo_corpus(input_path, csv_path)
            analyze_typo_patterns(csv_path)
        
        print(f"\n✓ Typo datasets ready in {output_dir}")
    else:
        print("\n⚠️  Could not download typo corpus. Using synthetic augmentation.")
        print("  You may need to manually download typo datasets.")


if __name__ == "__main__":
    main()

