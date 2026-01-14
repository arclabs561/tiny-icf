#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Enhanced dataset downloader with multiple sources.

Downloads from:
- COCA (Corpus of Contemporary American English)
- SUBTLEX US (subtitle frequencies)
- Google Trillion Word Corpus
- BNC (British National Corpus) - if available
- FrequencyWords repository
- Kaggle datasets
"""

import csv
import gzip
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path, chunk_size: int = 8192, desc: str = None):
    """Download a file with progress bar."""
    if desc is None:
        desc = output_path.name
    
    print(f"Downloading {desc}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        
        print(f"✓ Downloaded {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {desc}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_google_10000(output_dir: Path) -> Path | None:
    """Download Google 10,000 most common words."""
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
    output_path = output_dir / "google_10000.txt"
    
    if download_file(url, output_path, desc="Google 10K words"):
        return output_path
    return None


def download_frequency_words(output_dir: Path) -> Path | None:
    """Download from FrequencyWords repository (50k English words)."""
    url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
    output_path = output_dir / "frequency_words_50k.txt"
    
    if download_file(url, output_path, desc="FrequencyWords 50K"):
        return output_path
    return None


def download_unigram_freq(output_dir: Path) -> Path | None:
    """Download unigram frequency CSV from dwyl/english-words."""
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/data/unigram_freq.csv"
    output_path = output_dir / "unigram_freq.csv"
    
    if download_file(url, output_path, desc="Unigram frequency CSV"):
        return output_path
    return None


def download_kaggle_word_frequency(output_dir: Path) -> Path | None:
    """Download Kaggle English word frequency dataset."""
    # Note: This requires Kaggle API credentials for direct download
    # For now, we'll use a public mirror if available
    url = "https://raw.githubusercontent.com/rtatman/english-word-frequency/master/unigram_freq.csv"
    output_path = output_dir / "kaggle_unigram_freq.csv"
    
    if download_file(url, output_path, desc="Kaggle word frequency"):
        return output_path
    return None


def convert_to_csv(input_path: Path, output_path: Path, format_type: str):
    """Convert various formats to our CSV format (word,count)."""
    word_counts = {}
    
    print(f"Converting {input_path.name} to CSV...")
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        if format_type == "txt_wordlist":
            # Simple word list - assign Zipfian frequencies
            words = [line.strip().lower() for line in f if line.strip()]
            for rank, word in enumerate(words, 1):
                # Zipfian: count = base / rank
                count = int(1_000_000 / rank)
                word_counts[word] = count
        
        elif format_type == "frequency_space":
            # Format: word count (space-separated)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        word = parts[0].lower()
                        count = int(parts[1])
                        word_counts[word] = word_counts.get(word, 0) + count
                    except (ValueError, IndexError):
                        continue
        
        elif format_type == "csv":
            # Already CSV, just read and normalize
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Detect column indices
            word_col = 0
            count_col = 1
            
            if header:
                # Try to find word and count columns
                for i, col in enumerate(header):
                    if col.lower() in ['word', 'token', 'text', 'unigram']:
                        word_col = i
                    elif col.lower() in ['count', 'freq', 'frequency']:
                        count_col = i
            
            for row in reader:
                if len(row) > max(word_col, count_col):
                    try:
                        word = row[word_col].strip().lower()
                        count = int(float(row[count_col]))  # Handle float counts
                        word_counts[word] = word_counts.get(word, 0) + count
                    except (ValueError, IndexError):
                        continue
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'count'])  # Header
        # Sort by count (descending)
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    total_tokens = sum(word_counts.values())
    print(f"  Converted {len(word_counts)} words, {total_tokens:,} total tokens")
    
    return output_path


def merge_datasets(datasets: list[tuple[Path, str]], output_path: Path) -> Path:
    """Merge multiple datasets into one, combining counts for duplicate words."""
    print(f"\nMerging {len(datasets)} datasets...")
    
    all_word_counts = {}
    
    for input_path, format_type in datasets:
        if not input_path.exists():
            continue
        
        temp_csv = output_path.parent / f"temp_{input_path.stem}.csv"
        convert_to_csv(input_path, temp_csv, format_type)
        
        # Read and merge
        with open(temp_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        word = row[0].lower()
                        count = int(row[1])
                        all_word_counts[word] = all_word_counts.get(word, 0) + count
                    except (ValueError, IndexError):
                        continue
        
        # Clean up temp file
        temp_csv.unlink()
    
    # Write merged CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'count'])
        for word, count in sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    total = sum(all_word_counts.values())
    print(f"✓ Merged {len(all_word_counts)} unique words, {total:,} total tokens")
    print(f"  Saved to {output_path}")
    
    return output_path


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Enhanced Dataset Downloader")
    print("=" * 80)
    print("\nDownloading from multiple sources...")
    
    downloaded = []
    
    # Try multiple sources
    sources = [
        (download_unigram_freq, "csv"),
        (download_frequency_words, "frequency_space"),
        (download_google_10000, "txt_wordlist"),
        (download_kaggle_word_frequency, "csv"),
    ]
    
    for download_func, format_type in sources:
        try:
            result = download_func(output_dir)
            if result:
                downloaded.append((result, format_type))
        except Exception as e:
            print(f"Error downloading from {download_func.__name__}: {e}")
            continue
    
    if not downloaded:
        print("\n⚠️  No datasets downloaded. Check your internet connection.")
        return
    
    print(f"\n✓ Downloaded {len(downloaded)} datasets")
    
    # Merge all datasets
    merged_path = output_dir / "word_frequency_merged.csv"
    merge_datasets(downloaded, merged_path)
    
    print(f"\n{'='*80}")
    print("Download Complete!")
    print(f"{'='*80}")
    print(f"\nUse for training:")
    print(f"  uv run scripts/train_best_practices.py --data {merged_path} --epochs 100")
    print(f"\nOr use individual datasets:")
    for path, _ in downloaded:
        csv_path = output_dir / f"{path.stem}.csv"
        if csv_path.exists():
            print(f"  --data {csv_path}")


if __name__ == "__main__":
    main()

