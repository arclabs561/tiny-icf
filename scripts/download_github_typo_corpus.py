"""Download GitHub Typo Corpus (350k edits, 15+ languages)."""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests


def download_github_typo_corpus(output_dir: Path):
    """Download GitHub Typo Corpus."""
    url = "https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz"
    output_path = output_dir / "github_typo_corpus.jsonl.gz"
    
    try:
        print("Downloading GitHub Typo Corpus (350k edits, ~500MB)...")
        print("This may take a few minutes...")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 16):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"\n✗ Failed to download: {e}")
        return None


def extract_typo_pairs(jsonl_gz_path: Path, output_csv: Path, max_edits: int = 100000):
    """Extract typo-correction pairs from GitHub Typo Corpus."""
    typo_pairs = []
    languages = defaultdict(int)
    
    print(f"\nExtracting typo pairs (max {max_edits:,} edits)...")
    
    with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_edits:
                break
            
            try:
                commit = json.loads(line)
                for edit in commit.get('edits', []):
                    if edit.get('is_typo', False) or edit.get('prob_typo', 0) > 0.5:
                        src_text = edit.get('src', {}).get('text', '').strip()
                        tgt_text = edit.get('tgt', {}).get('text', '').strip()
                        lang = edit.get('src', {}).get('lang', 'unknown')
                        
                        if src_text and tgt_text and src_text != tgt_text:
                            # Extract word-level typos (simple: split by space)
                            src_words = src_text.lower().split()
                            tgt_words = tgt_text.lower().split()
                            
                            # Simple alignment: match words by position
                            for src_word, tgt_word in zip(src_words, tgt_words):
                                if src_word != tgt_word and len(src_word) > 1 and len(tgt_word) > 1:
                                    typo_pairs.append((src_word, tgt_word))
                                    languages[lang] += 1
            except (json.JSONDecodeError, KeyError):
                continue
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} commits...")
    
    # Deduplicate
    unique_pairs = list(set(typo_pairs))
    
    # Write CSV
    import csv
    with open(output_csv, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['typo', 'correction'])
        for typo, correction in sorted(unique_pairs):
            writer.writerow([typo, correction])
    
    print(f"\n✓ Extracted {len(unique_pairs):,} unique typo-correction pairs")
    print(f"  Languages: {dict(languages)}")
    return output_csv


def analyze_patterns(typo_csv: Path):
    """Analyze typo patterns from real data."""
    from collections import defaultdict
    
    patterns = defaultdict(int)
    total = 0
    
    import csv
    with open(typo_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            typo = row['typo']
            correction = row['correction']
            total += 1
            
            # Classify
            if len(typo) == len(correction):
                diffs = sum(1 for a, b in zip(typo, correction) if a != b)
                if diffs == 1:
                    patterns['substitution'] += 1
                elif diffs == 2:
                    # Check if adjacent
                    diff_positions = [i for i, (a, b) in enumerate(zip(typo, correction)) if a != b]
                    if len(diff_positions) == 2 and abs(diff_positions[0] - diff_positions[1]) == 1:
                        patterns['adjacent_swap'] += 1
                    else:
                        patterns['distant_swap'] += 1
            elif len(typo) < len(correction):
                patterns['char_drop'] += 1
            elif len(typo) > len(correction):
                patterns['char_insert'] += 1
    
    # Normalize
    if total > 0:
        frequencies = {k: v / total for k, v in patterns.items()}
        print("\nReal Typo Pattern Frequencies (from GitHub corpus):")
        for pattern, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {freq:.3f} ({patterns[pattern]:,} occurrences)")
        
        # Save
        import json
        freq_path = typo_csv.parent / "github_typo_pattern_frequencies.json"
        with open(freq_path, 'w') as f:
            json.dump(frequencies, f, indent=2)
        print(f"\n✓ Saved frequencies to {freq_path}")
        return frequencies
    
    return {}


def main():
    output_dir = Path("data/typos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading GitHub Typo Corpus")
    print("=" * 80)
    
    jsonl_gz = download_github_typo_corpus(output_dir)
    
    if jsonl_gz:
        csv_path = output_dir / "github_typos.csv"
        extract_typo_pairs(jsonl_gz, csv_path, max_edits=50000)  # Sample first 50k
        analyze_patterns(csv_path)
        print(f"\n✓ Typo corpus ready: {csv_path}")
    else:
        print("\n⚠️  Could not download. You can manually download from:")
        print("  https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz")


if __name__ == "__main__":
    main()

