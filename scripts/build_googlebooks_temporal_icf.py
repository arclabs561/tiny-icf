# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
#   "tqdm>=4.65.0",
#   "msgpack==1.1.2",
# ]
# ///
"""
Build a temporal ICF dataset from Google Books Ngram Viewer exports (20200217).

This downloads the 1-gram partition files for a corpus (default: `eng`) and
computes per-decade ICF values for a chosen vocabulary.

Output format:
  word,icf_1800,icf_1900,icf_2000,...   (columns match --decades)

This is designed to feed `--temporal-data` for `scripts/train_all_fronts.py`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import msgpack
import requests
from tqdm import tqdm


@dataclass(frozen=True)
class CorpusSpec:
    release: str
    corpus: str

    @property
    def base_url(self) -> str:
        return f"https://storage.googleapis.com/books/ngrams/books/{self.release}/{self.corpus}/"

    @property
    def index_url(self) -> str:
        return f"{self.base_url}{self.corpus}-1-ngrams_exports.html"

    @property
    def totalcounts_url(self) -> str:
        return f"{self.base_url}totalcounts-1"


_PART_RE = re.compile(r"1-(\d{5})-of-(\d{5})\.gz")


def _download_text(url: str, *, timeout_s: int = 60) -> str:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def _download_stream(url: str, *, timeout_s: int = 60) -> requests.Response:
    r = requests.get(url, stream=True, timeout=timeout_s)
    r.raise_for_status()
    return r


def list_part_urls(spec: CorpusSpec) -> list[str]:
    """
    Parse the corpus index page to find all `1-xxxxx-of-yyyyy.gz` part URLs.
    """
    text = _download_text(spec.index_url)
    # Some extractions HTML-escape URLs; unescape defensively.
    text = html.unescape(text)
    matches = list(_PART_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"Could not find part files on index page: {spec.index_url}")

    total = int(matches[0].group(2))
    # There should be total parts 0..total-1.
    urls = [f"{spec.base_url}1-{i:05d}-of-{total:05d}.gz" for i in range(total)]
    return urls


def parse_totalcounts(text: str) -> dict[int, int]:
    """
    totalcounts-1 is whitespace-separated entries like:
      year,match_count,page_count,volume_count
    We use year->match_count.
    """
    totals: dict[int, int] = {}
    for chunk in text.strip().split():
        parts = chunk.split(",")
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            match_count = int(parts[1])
        except ValueError:
            continue
        totals[year] = match_count
    return totals


def decade_totals_from_year_totals(year_totals: dict[int, int], decades: list[int]) -> dict[int, int]:
    out: dict[int, int] = {int(d): 0 for d in decades}
    for year, total in year_totals.items():
        dec = (int(year) // 10) * 10
        if dec in out:
            out[dec] += int(total)
    return out


def normalize_ngram_token(token: str) -> str:
    """
    Normalize a Google Books 1-gram token to match our vocab conventions:
    - strip trailing POS tag like `_NOUN` / `_NUM` / `_ADJ` when present
    - lowercase
    """
    tok = token.strip()
    if "_" in tok:
        stem, tag = tok.rsplit("_", 1)
        if tag.isupper():
            tok = stem
    return tok.lower()


def load_vocab_words(
    vocab_csv: Path,
    *,
    max_words: int,
    require_lang: str | None,
) -> set[str]:
    """
    Load a word frequency CSV and return a set of base tokens to keep.

    - If tokens are `lang:word` and require_lang is set, keep only that lang and strip the prefix.
    - If tokens are `lang:word` and require_lang is None, strip the prefix and keep all.
    """
    rows: list[tuple[str, int]] = []
    with open(vocab_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        # Skip header
        if first and len(first) >= 2 and first[0].lower() in {"word", "token"}:
            pass
        else:
            if first and len(first) >= 2:
                try:
                    rows.append((first[0], int(first[1])))
                except Exception:
                    pass
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append((row[0], int(row[1])))
            except Exception:
                continue

    # Sort by count desc and truncate
    rows.sort(key=lambda kv: kv[1], reverse=True)
    if max_words > 0:
        rows = rows[: int(max_words)]

    out: set[str] = set()
    for w, _c in rows:
        w = str(w).strip()
        if not w:
            continue
        if ":" in w:
            lang, base = w.split(":", 1)
            if require_lang is not None and lang != require_lang:
                continue
            w = base
        out.add(w.lower())
    return out


def _iter_lines_from_gz_url(url: str) -> Iterable[bytes]:
    r = _download_stream(url)
    f = gzip.GzipFile(fileobj=r.raw)
    while True:
        line = f.readline()
        if not line:
            break
        yield line


def process_part(
    url: str,
    *,
    decades: set[int],
    vocab: set[str],
    min_count: int,
) -> dict[str, dict[int, int]]:
    """
    Process one partition file and return counts for vocab words, aggregated by decade.
    """
    out: dict[str, dict[int, int]] = {}
    for raw in _iter_lines_from_gz_url(url):
        try:
            s = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not s:
            continue
        fields = s.split("\t")
        if len(fields) < 2:
            continue

        tok = normalize_ngram_token(fields[0])
        if tok not in vocab:
            continue

        # Each remaining field is like "year,match_count,volume_count"
        # We aggregate match_count into the chosen decades.
        dmap = out.get(tok)
        if dmap is None:
            dmap = {}
            out[tok] = dmap

        for triplet in fields[1:]:
            parts = triplet.split(",")
            if len(parts) < 2:
                continue
            try:
                year = int(parts[0])
                count = int(parts[1])
            except ValueError:
                continue
            if count < 0:
                continue
            dec = (year // 10) * 10
            if dec not in decades:
                continue
            dmap[dec] = dmap.get(dec, 0) + count

    # Optional prune: drop words that never reach min_count anywhere (keeps msgpack smaller).
    if min_count > 0:
        pruned: dict[str, dict[int, int]] = {}
        for w, m in out.items():
            if any(int(v) >= min_count for v in m.values()):
                pruned[w] = m
        return pruned

    return out


def write_msgpack(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(msgpack.packb(obj, use_bin_type=True))


def read_msgpack(path: Path) -> object:
    with open(path, "rb") as f:
        # We store decade keys as ints; allow non-str map keys.
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def compute_icf_for_decade(total_tokens: int, count: int, min_count: int) -> float:
    if count < min_count:
        return 1.0
    if total_tokens <= 0:
        return 1.0
    if count >= total_tokens:
        return 0.0
    log_total = math.log(total_tokens + 1)
    icf = math.log((total_tokens + 1) / (count + 1)) / log_total
    return float(max(0.0, min(1.0, icf)))


def main() -> int:
    p = argparse.ArgumentParser(description="Build temporal ICF from Google Books exports")
    p.add_argument("--corpus", type=str, default="eng", help="Google Books corpus code (e.g. eng)")
    p.add_argument("--release", type=str, default="20200217", help="Release directory under books/ngrams/books/")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path")
    p.add_argument("--cache-dir", type=Path, default=Path("data/historical_ngrams/cache"))
    p.add_argument("--resume", action="store_true", help="Reuse cached per-part msgpack outputs")
    p.add_argument("--max-files", type=int, default=0, help="Limit number of parts to process (0=all)")

    p.add_argument("--vocab", type=Path, required=True, help="Frequency CSV used to define vocabulary")
    p.add_argument("--vocab-max", type=int, default=200_000, help="Keep only top-N vocab words (0=all)")
    p.add_argument("--vocab-lang", type=str, default=None, help="If vocab is lang:word, keep only this lang")

    p.add_argument("--decades", type=str, default="1800,1900,2000", help="Comma-separated decades")
    p.add_argument("--min-count", type=int, default=5, help="Min count per decade to avoid ICF=1.0")
    args = p.parse_args()

    spec = CorpusSpec(release=str(args.release), corpus=str(args.corpus))

    decades_list = [int(x.strip()) for x in str(args.decades).split(",") if x.strip()]
    decades_set = set(decades_list)
    if not decades_list:
        raise RuntimeError("--decades must be non-empty")

    # Cache key must include parameters that affect counts/ICF.
    decades_key = "-".join(str(d) for d in decades_list)
    vocab_name = Path(args.vocab).name.replace(".", "_")
    vocab_lang = str(args.vocab_lang) if args.vocab_lang else "all"
    cache_dir = (
        Path(args.cache_dir)
        / f"{spec.release}_{spec.corpus}_1grams"
        / f"decades_{decades_key}"
        / f"vocab_{vocab_name}_top{int(args.vocab_max)}_lang{vocab_lang}_min{int(args.min_count)}"
    )
    parts_dir = cache_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Vocab
    vocab = load_vocab_words(
        Path(args.vocab),
        max_words=int(args.vocab_max),
        require_lang=str(args.vocab_lang) if args.vocab_lang else None,
    )
    if not vocab:
        raise RuntimeError("Vocab is empty after filtering")

    # Totals
    totals_text = _download_text(spec.totalcounts_url)
    year_totals = parse_totalcounts(totals_text)
    decade_totals = decade_totals_from_year_totals(year_totals, decades_list)

    # Parts
    part_urls = list_part_urls(spec)
    if int(args.max_files) > 0:
        part_urls = part_urls[: int(args.max_files)]

    # Process parts (cache per part)
    for url in tqdm(part_urls, desc="parts"):
        m = _PART_RE.search(url)
        part_name = m.group(0) if m else Path(url).name
        out_path = parts_dir / f"{part_name}.msgpack"
        if args.resume and out_path.exists():
            continue
        part_counts = process_part(
            url,
            decades=decades_set,
            vocab=vocab,
            min_count=int(args.min_count),
        )
        write_msgpack(out_path, part_counts)

    # Merge
    merged: dict[str, dict[int, int]] = {}
    for mp in sorted(parts_dir.glob("1-*-of-*.gz.msgpack")):
        obj = read_msgpack(mp)
        if not isinstance(obj, dict):
            continue
        for w, d in obj.items():
            if not isinstance(w, str) or not isinstance(d, dict):
                continue
            cur = merged.get(w)
            if cur is None:
                cur = {}
                merged[w] = cur
            for dec, c in d.items():
                try:
                    dec_i = int(dec)
                    c_i = int(c)
                except Exception:
                    continue
                cur[dec_i] = cur.get(dec_i, 0) + c_i

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word"] + [f"icf_{dec}" for dec in decades_list])
        for word in sorted(merged.keys()):
            row = [word]
            counts = merged[word]
            for dec in decades_list:
                total = int(decade_totals.get(int(dec), 0))
                c = int(counts.get(int(dec), 0))
                row.append(compute_icf_for_decade(total, c, int(args.min_count)))
            w.writerow(row)

    print(f"Wrote temporal ICF for {len(merged):,} words to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

