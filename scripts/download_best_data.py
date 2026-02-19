# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "wordfreq",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Download/build the best available training data for tiny-icf.

Right now, the highest-quality *practical* frequency source that works cross-language
without multi-GB downloads is `wordfreq` (built from multiple large corpora).

This script writes:
- data/word_frequency.csv                 (English, clean words + injected noise tokens)
- data/word_frequency_multilingual.csv    (lang:word keys for the built-in 10 languages)

For temporal data, use:
  bash scripts/setup_historical_data.sh
or:
  uv run python scripts/build_googlebooks_temporal_icf.py ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_wordfreq_frequencies import build_frequency, write_csv  # type: ignore


DEFAULT_LANGS = ["en", "es", "fr", "de", "it", "pt", "ru", "ko", "zh", "ja"]


def main() -> int:
    p = argparse.ArgumentParser(description="Build best training datasets for tiny-icf")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--english-words", type=int, default=500_000)
    p.add_argument("--multilingual-words-per-lang", type=int, default=200_000)
    p.add_argument("--min-zipf", type=float, default=1.0)
    p.add_argument("--noise-n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # English canonical dataset
    en_counts = build_frequency(
        languages=["en"],
        words_per_lang=int(args.english_words),
        scale=1_000_000_000,
        multilingual_keys=False,
        min_zipf=float(args.min_zipf),
        seed=int(args.seed),
        include_noise=True,
        noise_n=int(args.noise_n),
    )
    write_csv(en_counts, data_dir / "word_frequency.csv")

    # Multilingual dataset (lang:word keys)
    ml_counts = build_frequency(
        languages=list(DEFAULT_LANGS),
        words_per_lang=int(args.multilingual_words_per_lang),
        scale=1_000_000_000,
        multilingual_keys=True,
        min_zipf=float(args.min_zipf),
        seed=int(args.seed),
        include_noise=True,
        noise_n=int(args.noise_n),
    )
    write_csv(ml_counts, data_dir / "word_frequency_multilingual.csv")

    print("Done.")
    print(f"- wrote {data_dir / 'word_frequency.csv'}")
    print(f"- wrote {data_dir / 'word_frequency_multilingual.csv'}")
    print("")
    print("Next:")
    print("  bash scripts/setup_historical_data.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

