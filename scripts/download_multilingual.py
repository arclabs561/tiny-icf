"""Download multilingual frequency datasets."""

import csv
import sys
from pathlib import Path

import requests


def download_frequency_words(language: str, output_dir: Path):
    """Download frequency words dataset for a language."""
    base_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"
    
    url = f"{base_url}/{language}/{language}_50k.txt"
    output_path = output_dir / f"{language}_50k.txt"
    
    try:
        print(f"Downloading {language} frequency list...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download {language}: {e}")
        return None


def convert_to_csv(input_path: Path, output_path: Path):
    """Convert frequency words format to CSV."""
    word_counts = {}
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    total = sum(word_counts.values())
    print(f"  Converted {len(word_counts)} words, {total:,} total tokens")
    return output_path


def combine_multilingual(csv_files: list[Path], output_path: Path):
    """Combine multiple language frequency lists."""
    all_word_counts = {}
    language_counts = {}
    
    for csv_file in csv_files:
        lang = csv_file.stem.replace('_50k', '')
        language_counts[lang] = 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    word = row[0].lower()
                    count = int(row[1])
                    # Prefix with language code to avoid collisions
                    key = f"{lang}:{word}"
                    all_word_counts[key] = count
                    language_counts[lang] += count
    
    # Write combined
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for word, count in sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])
    
    total = sum(all_word_counts.values())
    print(f"\n✓ Combined multilingual dataset:")
    print(f"  Total words: {len(all_word_counts):,}")
    print(f"  Total tokens: {total:,}")
    print(f"  Languages: {', '.join(language_counts.keys())}")
    for lang, count in language_counts.items():
        print(f"    {lang}: {count:,} tokens")
    
    return output_path


def main():
    output_dir = Path("data/multilingual")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common languages to download
    languages = [
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "ru",  # Russian
        "zh",  # Chinese (simplified)
        "ja",  # Japanese
        "ko",  # Korean
    ]
    
    print("=" * 80)
    print("Downloading Multilingual Frequency Datasets")
    print("=" * 80)
    
    downloaded = []
    csv_files = []
    
    for lang in languages:
        txt_path = download_frequency_words(lang, output_dir)
        if txt_path:
            csv_path = output_dir / f"{lang}_50k.csv"
            convert_to_csv(txt_path, csv_path)
            downloaded.append(txt_path)
            csv_files.append(csv_path)
    
    if csv_files:
        combined_path = output_dir / "multilingual_combined.csv"
        combine_multilingual(csv_files, combined_path)
        print(f"\n✓ Use for training: --data {combined_path}")
    else:
        print("\n⚠️  No datasets downloaded. Check your internet connection.")


if __name__ == "__main__":
    main()

