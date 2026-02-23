#!/usr/bin/env -S uv run
"""Debug why common words like 'the' get wrong ICF: base vs language-conditioned output."""

import argparse
from pathlib import Path

import torch

from tiny_icf.calibration import apply_affine, load_calibration
from tiny_icf.checkpoint import load_model
from tiny_icf.data_multi_task import LANGUAGE_CODES
from tiny_icf.predict import word_to_bytes


def main():
    parser = argparse.ArgumentParser(description="Debug base vs lang-corrected ICF for probe words")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt or .ckpt")
    parser.add_argument("--calibration", type=Path, default=None, help="Optional calibration JSON")
    parser.add_argument(
        "--words",
        type=str,
        nargs="+",
        default=["the", "and", "of", "qzxbjk"],
        help="Probe words",
    )
    parser.add_argument("--data", type=Path, default=None, help="Optional CSV to show target ICF for probe words")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_model(args.model, device=device)
    model.eval()

    cal = load_calibration(args.calibration) if args.calibration else None

    targets = {}
    if args.data and args.data.exists():
        from tiny_icf.data import load_frequency_list, compute_normalized_icf
        wc, total = load_frequency_list(args.data)
        icf_map = compute_normalized_icf(wc, total, multilingual=False)
        for w in args.words:
            targets[w] = icf_map.get(w) or icf_map.get(f"en:{w}")

    # MultiTaskICF exposes .base and forward(..., return_all=True)
    has_base = hasattr(model, "base")
    has_return_all = hasattr(model, "output_tasks") and "language" in getattr(
        model, "output_tasks", []
    )

    print("Probe words:", args.words)
    print("Has base:", has_base, "| Has lang correction (return_all):", has_return_all)
    if cal:
        a, b = cal
        print("Calibration: a={:.4f} b={:.4f}".format(a, b))
    print()

    with torch.no_grad():
        for word in args.words:
            x = word_to_bytes(word).to(device)
            base_icf = None
            if has_base:
                base_icf = model.base(x).squeeze().item()
            if has_return_all:
                out = model(x, return_all=True)
                final_icf = out["icf"].squeeze().item()
                lang_logits = out.get("language")
                if lang_logits is not None:
                    probs = torch.softmax(lang_logits, dim=-1).squeeze().tolist()
                    top = sorted(
                        zip(LANGUAGE_CODES, probs), key=lambda p: p[1], reverse=True
                    )[:3]
                    lang_str = " ".join(f"{l}:{p:.3f}" for l, p in top)
                else:
                    lang_str = "N/A"
            else:
                final_icf = model(x).squeeze().item()
                lang_str = "N/A"

            correction = (final_icf - base_icf) if base_icf is not None else None
            line = f"  {word!r}: final={final_icf:.4f}"
            if base_icf is not None:
                line += f" base={base_icf:.4f} correction={correction:+.4f}"
            if lang_str != "N/A":
                line += f"  lang_top={lang_str}"
            if cal:
                a, b = cal
                cal_icf = apply_affine(final_icf, a, b)
                line += f"  calibrated={cal_icf:.4f}"
            if word in targets and targets[word] is not None:
                line += f"  target={targets[word]:.4f}"
            print(line)

    print("\nInterpretation: if base is already high for 'the', the bug is in the base model.")
    print("If correction is large and positive for 'the', lang_icf_cond may be pushing common words up.")
    print("Target for 'the' in typical data is ~0.14; base >> 0.14 means head words are underfit (try frequency-weighted sampling).")


if __name__ == "__main__":
    main()
