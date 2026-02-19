# /// script
# requires-python = ">=3.10"
# ///
"""
Prefix a `word,count` frequency CSV with a language code (lang:word).

This is useful when you want to fine-tune a model on a corpus where downstream
training expects `lang:token` keys, without changing counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Prefix frequency CSV keys as lang:word")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lang", type=str, required=True)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as fin, open(
        args.output, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        first = next(reader, None)
        wrote_header = False
        if first and len(first) >= 2 and first[0].lower() in {"word", "token"}:
            writer.writerow(first)
            wrote_header = True
        else:
            if first and len(first) >= 2:
                try:
                    word = str(first[0]).strip()
                    count = int(first[1])
                    if word and ":" not in word:
                        word = f"{args.lang}:{word}"
                    writer.writerow([word, count])
                except Exception:
                    pass

        if not wrote_header:
            # Ensure canonical header exists.
            # If the input had no header, we already wrote the first row above,
            # so don't rewind; just proceed.
            pass

        for row in reader:
            if len(row) < 2:
                continue
            try:
                word = str(row[0]).strip()
                count = int(row[1])
            except Exception:
                continue
            if not word:
                continue
            if ":" not in word:
                word = f"{args.lang}:{word}"
            writer.writerow([word, count])

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

