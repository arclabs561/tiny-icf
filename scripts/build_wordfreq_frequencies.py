# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "wordfreq",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Build high-quality word frequency CSVs using `wordfreq`.

Outputs `word,count` CSVs compatible with `tiny_icf.data.load_frequency_list`.

Key feature: can emit multilingual keys as `lang:token`, which enables
`compute_normalized_icf(..., multilingual=True)` / `--multilingual` training.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from tqdm import tqdm
import wordfreq


DEFAULT_LANGS = ["en", "es", "fr", "de", "it", "pt", "ru", "ko", "zh", "ja"]


def _zipf_to_count(zipf: float, scale: int) -> int:
    """
    Convert Zipf frequency to pseudo-count.

    In `wordfreq`, Zipf is log10(occurrences per billion words). So:
      expected_count_per_billion = 10**zipf

    We scale counts linearly; only relative frequency matters for ICF.
    """
    if not math.isfinite(zipf):
        return 0
    # occurrences per 1e9 tokens:
    per_billion = 10.0 ** float(zipf)
    # scale defaults to 1e9, so count ~= per_billion
    count = int(round(per_billion * (float(scale) / 1_000_000_000.0)))
    return max(0, count)


def _is_single_token(s: str) -> bool:
    # We intentionally avoid whitespace-separated multiword entries.
    return bool(s) and not any(ch.isspace() for ch in s)


def _generate_noise_tokens(n: int, seed: int) -> list[tuple[str, int]]:
    """
    Generate contamination tokens that `filter_frequency_list` will remove.

    We return (token, count). Tokens are deliberately "word-ish" enough to enter
    the raw list, but should be filtered from the clean ICF list.
    """
    rng = random.Random(int(seed))
    out: list[tuple[str, int]] = []

    # Fixed set of common HTML entities.
    html = ["&amp;", "&nbsp;", "&lt;", "&gt;", "&quot;"]
    for t in html:
        out.append((t, rng.randint(10_000, 50_000)))

    # URLs
    for i in range(max(1, n // 5)):
        dom = rng.choice(["example.com", "github.com", "wikipedia.org", "nytimes.com", "bbc.co.uk"])
        path = rng.choice(["/a", "/news", "/docs", "/q", "/search"])
        token = f"https://{dom}{path}?q={rng.randint(1,999)}"
        out.append((token, rng.randint(1_000, 20_000)))

    # Emails
    for i in range(max(1, n // 8)):
        user = rng.choice(["test", "info", "support", "admin", "contact"])
        dom = rng.choice(["example.com", "example.org", "mail.com", "company.co"])
        out.append((f"{user}{i}@{dom}", rng.randint(500, 10_000)))

    # Code-like
    code = [
        "def foo(x): return x",
        "function(x){return x}",
        "import torch",
        "#include<stdio.h>",
        "<script>alert(1)</script>",
    ]
    for t in code:
        out.append((t, rng.randint(200, 5_000)))

    # Pure numbers (removed)
    for i in range(max(1, n // 6)):
        out.append((str(rng.randint(0, 10_000_000)), rng.randint(200, 5_000)))

    # Mojibake / encoding errors
    moj = ["Ã©clair", "â€™", "â€œ", "â€\"", "Ã±andÃº"]
    for t in moj:
        out.append((t, rng.randint(200, 5_000)))

    # Dedup while keeping max count
    dedup: dict[str, int] = {}
    for w, c in out:
        dedup[w] = max(dedup.get(w, 0), int(c))
    return list(dedup.items())


def build_frequency(
    *,
    languages: list[str],
    words_per_lang: int,
    scale: int,
    multilingual_keys: bool,
    min_zipf: float,
    seed: int,
    include_noise: bool,
    noise_n: int,
) -> dict[str, int]:
    rng = random.Random(int(seed))
    counts: dict[str, int] = {}

    for lang in languages:
        top = wordfreq.top_n_list(lang, int(words_per_lang))
        # Some languages (notably ja/ko) may require optional tokenizers (MeCab).
        # If zipf_frequency() fails due to missing deps, fall back to a rank-only
        # synthetic Zipf curve. Ranking is the important signal for ICF anyway.
        use_rank_fallback = False
        if top:
            try:
                _ = float(wordfreq.zipf_frequency(top[0], lang))
            except Exception:
                use_rank_fallback = True

        for rank, w in enumerate(tqdm(top, desc=f"wordfreq[{lang}]")):
            if not _is_single_token(w):
                continue
            if use_rank_fallback:
                # Synthetic Zipf: top word ~ 1e8/billion, then decays ~ 1/rank.
                z = 8.0 - math.log10(rank + 1.0)
            else:
                try:
                    z = float(wordfreq.zipf_frequency(w, lang))
                except Exception:
                    # If a single token triggers a tokenizer edge-case, fall back for it.
                    z = 8.0 - math.log10(rank + 1.0)
            if z < float(min_zipf):
                continue
            c = _zipf_to_count(z, int(scale))
            if c <= 0:
                continue
            key = f"{lang}:{w}" if multilingual_keys else w
            counts[key] = counts.get(key, 0) + c

    if include_noise:
        noise = _generate_noise_tokens(int(noise_n), seed=int(seed) + 1)
        rng.shuffle(noise)
        for w, c in noise:
            counts[w] = counts.get(w, 0) + int(c)

    return counts


def write_csv(counts: dict[str, int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "count"])
        for word, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([word, int(count)])


def main() -> int:
    p = argparse.ArgumentParser(description="Build word frequency CSVs using wordfreq")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path")
    p.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated language codes",
    )
    p.add_argument("--words-per-lang", type=int, default=200_000)
    p.add_argument(
        "--scale",
        type=int,
        default=1_000_000_000,
        help="Pseudo token budget per language for count scaling",
    )
    p.add_argument(
        "--multilingual-keys",
        action="store_true",
        help="Prefix keys as lang:token (recommended for multilingual training)",
    )
    p.add_argument("--min-zipf", type=float, default=1.0, help="Drop extremely rare tokens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-noise", action="store_true", help="Include contamination tokens")
    p.add_argument("--noise-n", type=int, default=5000)
    args = p.parse_args()

    langs = [s.strip() for s in args.languages.split(",") if s.strip()]
    counts = build_frequency(
        languages=langs,
        words_per_lang=int(args.words_per_lang),
        scale=int(args.scale),
        multilingual_keys=bool(args.multilingual_keys),
        min_zipf=float(args.min_zipf),
        seed=int(args.seed),
        include_noise=bool(args.include_noise),
        noise_n=int(args.noise_n),
    )
    write_csv(counts, args.output)
    print(f"Wrote {len(counts):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

