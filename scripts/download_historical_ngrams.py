# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0.0",
#   "numpy>=1.24.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""Download and process historical n-gram data for temporal ICF training."""

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


# Google Books n-gram dataset URLs
# Note: Actual URLs may vary - check http://storage.googleapis.com/books/ngrams/books/datasetsv3.html
# Files are organized by starting letter: eng-1gram-a-{year}-{version}.gz
NGRAM_BASE_URL = "https://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-1gram-{letter}-{year}-{version}.gz"
YEARS = list(range(1800, 2020, 10))  # Decade-level data
LETTERS = list("abcdefghijklmnopqrstuvwxyz")  # Files split by starting letter
VERSIONS = ["1gram"]


def download_ngram_file(letter: str, year: int, version: str, output_dir: Path) -> Path:
    """Download a single n-gram file from Google Books dataset.
    
    Note: Google Books n-gram files are organized by starting letter.
    For full dataset, you need to download files for all letters a-z.
    """
    url = NGRAM_BASE_URL.format(letter=letter, year=year, version=version)
    filename = f"eng-1gram-{letter}-{year}-{version}.gz"
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"Already exists: {filename}")
        return output_path
    
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded: {filename}")
        return output_path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        # Try alternative URL format
        alt_url = f"https://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-1gram-{letter}-{year}.gz"
        try:
            urllib.request.urlretrieve(alt_url, output_path)
            print(f"Downloaded via alternative URL: {filename}")
            return output_path
        except:
            return None


def parse_ngram_line(line: str) -> Optional[Tuple[str, int, int, int]]:
    """Parse a line from Google Books n-gram file.
    
    Format: word\tyear\tmatch_count\tpage_count\tvolume_count
    """
    try:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            return None
        
        word = parts[0]
        # Filter: only alphanumeric words, reasonable length
        if not word.replace('_', '').isalnum() or len(word) < 2 or len(word) > 50:
            return None
        
        year = int(parts[1])
        match_count = int(parts[2])
        
        return (word.lower(), year, match_count, 0)
    except (ValueError, IndexError):
        return None


def process_ngram_file(filepath: Path, min_count: int = 5) -> Dict[str, Dict[int, int]]:
    """Process a single n-gram file and extract word frequencies by year."""
    word_freqs = defaultdict(lambda: defaultdict(int))
    
    if not filepath or not filepath.exists():
        return word_freqs
    
    print(f"Processing {filepath.name}...")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc=f"Reading {filepath.name}"):
                parsed = parse_ngram_line(line)
                if parsed is None:
                    continue
                
                word, year, count, _ = parsed
                
                # Filter: only alphanumeric words, reasonable length
                if not word.isalnum() or len(word) < 2 or len(word) > 50:
                    continue
                
                # Normalize to lowercase
                # Already lowercased in parse_ngram_line
                # Aggregate by decade
                decade = (year // 10) * 10
                word_freqs[word][decade] += count
                
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
    
    return word_freqs


def compute_temporal_icf(
    word_freqs: Dict[str, Dict[int, int]],
    decade: int,
    min_count: int = 5
) -> Dict[str, float]:
    """Compute ICF scores for a specific decade."""
    # Get total frequency for this decade
    total_tokens = sum(
        sum(freqs.get(decade, 0) for freqs in word_freqs.values())
    )
    
    if total_tokens == 0:
        return {}
    
    # Compute ICF for each word in this decade
    icf_scores = {}
    for word, freqs in word_freqs.items():
        count = freqs.get(decade, 0)
        if count < min_count:
            continue
        
        # ICF = 1 - (count / total_tokens)
        icf = 1.0 - (count / total_tokens)
        icf_scores[word] = max(0.0, min(1.0, icf))
    
    return icf_scores


def aggregate_historical_data(
    word_freqs: Dict[str, Dict[int, int]],
    decades: List[int]
) -> pd.DataFrame:
    """Aggregate historical frequency data across decades."""
    records = []
    
    for word, freqs in word_freqs.items():
        total_count = sum(freqs.values())
        if total_count < 5:  # Minimum threshold
            continue
        
        # Get frequency per decade
        decade_counts = {decade: freqs.get(decade, 0) for decade in decades}
        
        # Compute ICF for each decade
        decade_icfs = {}
        for decade in decades:
            decade_total = sum(
                freqs.get(decade, 0) for freqs in word_freqs.values()
            )
            if decade_total > 0:
                count = freqs.get(decade, 0)
                icf = 1.0 - (count / decade_total) if count > 0 else 1.0
                decade_icfs[f"icf_{decade}"] = max(0.0, min(1.0, icf))
        
        records.append({
            'word': word,
            'total_count': total_count,
            **decade_counts,
            **decade_icfs,
        })
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Download and process historical n-gram data"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/historical_ngrams'),
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[1800, 1900, 2000],
        help='Years to download data for'
    )
    parser.add_argument(
        '--ngram-type',
        type=str,
        default='1gram',
        choices=['1gram', '2gram', '3gram'],
        help='Type of n-gram to download'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=5,
        help='Minimum word count threshold'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download files, do not process'
    )
    parser.add_argument(
        '--process-only',
        action='store_true',
        help='Only process existing files, do not download'
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    if not args.process_only:
        print("Downloading n-gram files...")
        print("Note: Google Books n-gram files are large and organized by starting letter.")
        print("For a complete dataset, you may need to download files for all letters (a-z).")
        print("This script downloads a sample for demonstration.")
        
        downloaded_files = []
        # Download sample files (just 'a' and 't' for common words)
        sample_letters = ['a', 't']  # 'a' and 't' cover many common words
        for year in args.years:
            for letter in sample_letters:
                filepath = download_ngram_file(
                    letter=letter,
                    year=year,
                    version=args.ngram_type,
                    output_dir=output_dir
                )
                if filepath:
                    downloaded_files.append(filepath)
        
        if args.download_only:
            print("Download complete. Use --process-only to process files.")
            return
    
    # Process files
    if not args.download_only:
        print("Processing n-gram files...")
        all_word_freqs = defaultdict(lambda: defaultdict(int))
        
        for year in args.years:
            filename = f"eng-{args.ngram_type}-{year}-{args.ngram_type}.gz"
            filepath = output_dir / filename
            
            if filepath.exists():
                word_freqs = process_ngram_file(filepath, min_count=args.min_count)
                
                # Merge into aggregate
                for word, freqs in word_freqs.items():
                    for decade, count in freqs.items():
                        all_word_freqs[word][decade] += count
        
        # Aggregate and save
        decades = sorted(set(
            decade
            for freqs in all_word_freqs.values()
            for decade in freqs.keys()
        ))
        
        df = aggregate_historical_data(all_word_freqs, decades)
        
        output_csv = output_dir / f"historical_icf_{args.ngram_type}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved historical ICF data to {output_csv}")
        print(f"Total words: {len(df)}")
        print(f"Decades: {decades}")


if __name__ == '__main__':
    main()

