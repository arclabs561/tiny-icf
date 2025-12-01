"""Download and prepare real frequency datasets for training."""

import csv
import gzip
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

import requests


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress."""
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\nDownloaded to {output_path}")


def download_google_ngrams(output_dir: Path):
    """Download Google N-grams frequency data."""
    # Google Books N-grams are available but very large
    # We'll use a smaller, processed version if available
    print("Google N-grams are very large. Looking for processed frequency lists...")
    
    # Alternative: Use a pre-processed frequency list
    # Many researchers publish cleaned versions
    return None


def download_wikipedia_frequencies(output_dir: Path):
    """Download Wikipedia word frequencies."""
    # Wikipedia dumps are available but need processing
    # For now, we'll use a simpler approach
    print("Wikipedia dumps require processing. Using alternative sources...")
    return None


def download_common_words(output_dir: Path):
    """Download common English word frequency lists."""
    # Try multiple sources
    
    sources = [
        {
            "name": "english_words_frequency",
            "url": "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt",
            "format": "txt",
        },
        {
            "name": "word_frequency",
            "url": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt",
            "format": "frequency",
        },
    ]
    
    downloaded = []
    
    for source in sources:
        try:
            output_path = output_dir / f"{source['name']}.txt"
            print(f"\nTrying {source['name']}...")
            download_file(source['url'], output_path)
            downloaded.append((output_path, source['format']))
        except Exception as e:
            print(f"Failed to download {source['name']}: {e}")
            continue
    
    return downloaded


def convert_frequency_format(input_path: Path, output_path: Path, format_type: str):
    """Convert various frequency formats to our CSV format."""
    print(f"Converting {input_path.name} to CSV...")
    
    word_counts = {}
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        if format_type == "txt":
            # Simple word list - assign synthetic frequencies based on rank
            words = [line.strip().lower() for line in f if line.strip()]
            for rank, word in enumerate(words, 1):
                # Zipfian distribution: count = 1,000,000 / rank
                count = int(1_000_000 / rank)
                word_counts[word] = count
        
        elif format_type == "frequency":
            # Format: word count (space-separated)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        word = parts[0].lower()
                        count = int(parts[1])
                        word_counts[word] = count
                    except (ValueError, IndexError):
                        continue
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Sort by count (descending)
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    total_tokens = sum(word_counts.values())
    print(f"Converted {len(word_counts)} words, {total_tokens:,} total tokens")
    print(f"Saved to {output_path}")
    
    return output_path


def download_large_frequency_list(output_dir: Path):
    """Download a large, real frequency list."""
    # Try to find a good source
    # Option 1: Use a GitHub repository with processed data
    sources = [
        {
            "name": "unigram_freq",
            "url": "https://raw.githubusercontent.com/dwyl/english-words/master/data/unigram_freq.csv",
            "format": "csv",
        },
    ]
    
    for source in sources:
        try:
            output_path = output_dir / f"{source['name']}.csv"
            print(f"\nDownloading {source['name']}...")
            download_file(source['url'], output_path)
            
            # Verify format
            with open(output_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if ',' in first_line:
                    print(f"✓ CSV format detected")
                    return output_path
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    return None


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Downloading Frequency Datasets")
    print("=" * 80)
    
    # Try to download large frequency list first
    large_list = download_large_frequency_list(output_dir)
    
    if large_list:
        print(f"\n✓ Successfully downloaded large frequency list: {large_list}")
        print(f"  Use this for training: --data {large_list}")
        return
    
    # Fallback: Download smaller lists and combine
    print("\nTrying alternative sources...")
    downloaded = download_common_words(output_dir)
    
    if downloaded:
        # Convert and combine
        combined_path = output_dir / "combined_frequencies.csv"
        all_word_counts = {}
        
        for input_path, format_type in downloaded:
            temp_csv = output_dir / f"{input_path.stem}.csv"
            convert_frequency_format(input_path, temp_csv, format_type)
            
            # Merge into combined
            with open(temp_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        word = row[0].lower()
                        count = int(row[1])
                        all_word_counts[word] = all_word_counts.get(word, 0) + count
        
        # Write combined
        with open(combined_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for word, count in sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([word, count])
        
        total = sum(all_word_counts.values())
        print(f"\n✓ Combined {len(all_word_counts)} words, {total:,} total tokens")
        print(f"  Use for training: --data {combined_path}")
    else:
        print("\n⚠️  Could not download datasets. Using test data instead.")
        print("  You may need to manually download frequency lists.")


if __name__ == "__main__":
    main()

