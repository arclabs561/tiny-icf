"""Inference script for Universal ICF model."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import unicodedata

from tiny_icf.calibration import apply_affine, load_calibration
from tiny_icf.checkpoint import load_model
from tiny_icf.language_detection import detect_languages, format_languages
from tiny_icf.oov_calibration import (
    DEFAULT_SATURATION_FIX,
    SaturationFixConfig,
    apply_saturation_fix,
)
from tiny_icf.temporal_detection import estimate_usage_period, format_temporal_analysis


def word_to_bytes(word: str, max_length: int = 20) -> torch.Tensor:
    """
    Convert a word to a fixed-length UTF-8 byte tensor.

    This truncates *only at character boundaries* (never mid-codepoint), and pads
    with `0x00` bytes to `max_length`.
    """
    # Normalize to NFC (canonical composition) for consistency
    word = unicodedata.normalize("NFC", word)

    # Build a byte sequence without ever cutting a character's UTF-8 bytes.
    out = bytearray()
    for ch in word:
        b = ch.encode("utf-8")
        if len(out) + len(b) > max_length:
            break
        out.extend(b)

    pad_length = max(0, max_length - len(out))
    padded = bytes(out) + bytes(pad_length)
    return torch.tensor(list(padded), dtype=torch.long).unsqueeze(0)


def predict_icf(
    model: torch.nn.Module,
    word: str,
    device: torch.device,
    return_details: bool = False,
    reference_scores: Optional[np.ndarray] = None,
    *,
    max_length: int = 20,
    saturation_fix: bool = False,
    saturation_fix_config: SaturationFixConfig = DEFAULT_SATURATION_FIX,
    calibration: Optional[tuple[float, float]] = None,
) -> float | dict:
    """
    Predict ICF score for a single word.

    Args:
        model: Trained UniversalICF model
        word: Word to predict ICF for
        device: Device for computation
        return_details: If True, return dict with score, interpretation, confidence, etc.

    Returns:
        If return_details=False: float ICF score
        If return_details=True: dict with keys:
            - 'icf_score': float ICF score (0.0=common, 1.0=rare)
            - 'interpretation': str category (Very Common, Common, Rare, Very Rare)
            - 'confidence': float confidence estimate (0.0-1.0)
            - 'raw_output': float raw model output before clamping
            - 'category': str one of 'very_common', 'common', 'rare', 'very_rare'
    """
    model.eval()
    byte_tensor = word_to_bytes(word, max_length=max_length).to(device)

    with torch.no_grad():
        # Try to get features if model supports it
        try:
            if return_details:
                prediction, features = model(byte_tensor, return_features=True)
                # prediction is [1, 1] tensor, extract scalar
                icf_score_model = float(prediction.squeeze().item())
                raw_output = float(features.get("raw_output", prediction).squeeze().item())
                confidence = float(features.get("confidence", torch.tensor(0.5)).squeeze().item())
            else:
                prediction = model(byte_tensor)
                icf_score_model = float(prediction.squeeze().item())
                raw_output = None
                confidence = None
                features = {}
        except (TypeError, IndexError, AttributeError):
            # Model doesn't support return_features, use basic prediction
            prediction = model(byte_tensor)
            icf_score_model = float(prediction.squeeze().item())
            raw_output = icf_score_model
            confidence = 0.5  # Default confidence
            features = {}

    icf_score = icf_score_model
    saturation_fix_applied = False
    if calibration is not None:
        a, b = calibration
        icf_score = apply_affine(icf_score, a, b)
    if saturation_fix:
        # If we don't have raw_output (no return_features support), we can't fix saturation.
        if raw_output is not None:
            icf_score, saturation_fix_applied = apply_saturation_fix(
                icf_score=icf_score_model,
                raw_output=float(raw_output),
                confidence=float(confidence) if confidence is not None else None,
                config=saturation_fix_config,
            )

    if not return_details:
        return icf_score

    # Determine interpretation
    if icf_score < 0.2:
        interpretation = "Very Common (stopword-like)"
        category = "very_common"
    elif icf_score < 0.5:
        interpretation = "Common"
        category = "common"
    elif icf_score < 0.8:
        interpretation = "Rare"
        category = "rare"
    else:
        interpretation = "Very Rare/Unique"
        category = "very_rare"

    result = {
        "icf_score": icf_score,
        "icf_score_model": icf_score_model,
        "interpretation": interpretation,
        "category": category,
        "confidence": confidence if confidence is not None else 0.5,
        "raw_output": raw_output if raw_output is not None else icf_score,
        "saturation_fix_applied": bool(saturation_fix_applied),
        "word": word,
    }

    # Add percentile rank if reference scores provided
    if reference_scores is not None and len(reference_scores) > 0:
        percentile = (reference_scores <= icf_score).sum() / len(reference_scores) * 100.0
        result["percentile_rank"] = float(percentile)

    # Add language detection
    languages = detect_languages(word, method="combined")
    result["languages"] = format_languages(languages, top_k=3)

    # Add temporal/era detection
    temporal = estimate_usage_period(word, icf_score=icf_score)
    result["temporal"] = format_temporal_analysis(temporal)

    # If the model exposes learned auxiliary heads (MultiTaskICF), surface them.
    if isinstance(features, dict) and features:
        try:
            from tiny_icf.data_multi_task import ERA_CODES, HYGIENE_CODES, LANGUAGE_CODES
        except Exception:
            ERA_CODES = ["archaic", "early_modern", "modern", "contemporary", "neologism"]
            HYGIENE_CODES = [
                "clean_word",
                "url",
                "email",
                "code",
                "html_entity",
                "encoding_error",
                "number",
                "gibberish",
            ]
            LANGUAGE_CODES = ["en", "es", "fr", "de", "it", "pt", "ru", "ko", "zh", "ja"]

        def _topk_from_logits(logits: torch.Tensor, labels: list[str], k: int = 3) -> list[dict]:
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            kk = min(k, int(probs.numel()))
            vals, idx = torch.topk(probs, k=kk)
            out = []
            for v, i in zip(vals.tolist(), idx.tolist()):
                name = labels[int(i)] if 0 <= int(i) < len(labels) else str(int(i))
                out.append({"label": name, "p": float(v)})
            return out

        if "language_logits" in features:
            try:
                result["learned_language"] = _topk_from_logits(
                    features["language_logits"], list(LANGUAGE_CODES), k=3
                )
            except Exception:
                pass
        if "era_logits" in features:
            try:
                result["learned_era"] = _topk_from_logits(
                    features["era_logits"], list(ERA_CODES), k=3
                )
            except Exception:
                pass
        if "hygiene_logits" in features:
            try:
                result["learned_hygiene"] = _topk_from_logits(
                    features["hygiene_logits"], list(HYGIENE_CODES), k=3
                )
            except Exception:
                pass
        if "temporal_logits" in features and hasattr(model, "temporal_decades"):
            try:
                decades = list(getattr(model, "temporal_decades"))
                vec = features["temporal_logits"].squeeze(0)
                if vec.dim() == 1 and len(decades) == int(vec.numel()):
                    result["learned_temporal_icf"] = {
                        str(int(dec)): float(vec[i].item()) for i, dec in enumerate(decades)
                    }
            except Exception:
                pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict ICF scores for words")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Words to predict (space-separated string or single word)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--max-length", type=int, default=20, help="Max UTF-8 byte length per word")
    parser.add_argument(
        "--detailed", action="store_true", help="Return detailed predictions with confidence"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--saturation-fix",
        action="store_true",
        help="Optional OOV-focused fix: unsaturate clamp-to-1.0 outputs using raw_output.",
    )
    parser.add_argument(
        "--fix-center",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.center),
        help="Saturation-fix center parameter (raw_output at which fixed score is ~0.5).",
    )
    parser.add_argument(
        "--fix-scale",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.scale),
        help="Saturation-fix scale parameter (smaller = steeper mapping).",
    )
    parser.add_argument(
        "--fix-conf-weight",
        type=float,
        default=float(DEFAULT_SATURATION_FIX.confidence_weight),
        help="Optional saturation-fix confidence weight (0 disables).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration JSON (a, b). Apply learned affine calibration.",
    )

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model, _checkpoint = load_model(args.model, device=device)
    model.eval()

    # Optional learned calibration
    cal = load_calibration(args.calibration) if args.calibration else None
    if args.calibration and cal is None:
        raise SystemExit(f"Calibration file not found or invalid: {args.calibration}")

    # Parse words (handle both string and list)
    words = args.words.split() if isinstance(args.words, str) else args.words

    fix_config = SaturationFixConfig(
        eps=float(DEFAULT_SATURATION_FIX.eps),
        center=float(args.fix_center),
        scale=float(args.fix_scale),
        confidence_weight=float(args.fix_conf_weight),
        confidence_center=float(DEFAULT_SATURATION_FIX.confidence_center),
    )

    # Predict
    results = []
    for word in words:
        if args.detailed or args.json:
            result = predict_icf(
                model,
                word,
                device,
                return_details=True,
                max_length=args.max_length,
                saturation_fix=bool(args.saturation_fix),
                saturation_fix_config=fix_config,
                calibration=cal,
            )
            results.append(result)
        else:
            score = predict_icf(
                model,
                word,
                device,
                return_details=False,
                max_length=args.max_length,
                saturation_fix=bool(args.saturation_fix),
                saturation_fix_config=fix_config,
                calibration=cal,
            )
            result = {
                "word": word,
                "icf_score": score,
                "interpretation": (
                    "Very Common (stopword-like)"
                    if score < 0.2
                    else "Common" if score < 0.5 else "Rare" if score < 0.8 else "Very Rare/Unique"
                ),
            }
            results.append(result)

    # Output
    if args.json:
        import json

        print(json.dumps(results, indent=2))
    elif args.detailed:
        print(f"{'Word':<20} {'ICF Score':<12} {'Confidence':<12} {'Interpretation':<30}")
        print("-" * 80)
        for result in results:
            print(
                f"{result['word']:<20} {result['icf_score']:<12.4f} {result['confidence']:<12.4f} {result['interpretation']:<30}"
            )
    else:
        print(f"{'Word':<20} {'ICF Score':<12} {'Interpretation':<30}")
        print("-" * 80)
        for result in results:
            print(
                f"{result['word']:<20} {result['icf_score']:<12.4f} {result['interpretation']:<30}"
            )


if __name__ == "__main__":
    main()
