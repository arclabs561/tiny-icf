"""Checkpoint helpers (model + metadata).

This repo historically saved plain `state_dict`s (a dict of parameter tensors).
As experiments grew (Nano/Residual variants, attention on/off, output activation),
we also support saving *checkpoint dicts* that include:

- `model_type`: a short string like "UniversalICF"
- `model_kwargs`: constructor kwargs to re-create the module
- `model_state_dict`: the actual weights

All loaders in the repo should accept both formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _is_checkpoint_dict(obj: Any) -> bool:
    """Heuristic: distinguish checkpoint dicts from plain state_dicts."""
    return isinstance(obj, dict) and (
        "model_state_dict" in obj or "model_type" in obj or "model_kwargs" in obj
    )


def _strip_common_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Some trainers wrap state_dict keys with a common prefix (e.g. DataParallel: "module.").
    Strip a single common prefix if *all* keys share it.
    """
    if not state_dict:
        return state_dict

    prefixes = ("module.", "model.")
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if all(isinstance(k, str) and k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _infer_model_type_from_state_dict(state_dict: dict[str, Any]) -> str:
    """
    Best-effort inference for state_dict-only artifacts.

    This is intentionally conservative: when in doubt, we fall back to UniversalICF.
    """
    keys = set(k for k in state_dict.keys() if isinstance(k, str))

    # MultiTaskICF wraps a base model under `base.*` and adds task heads.
    if any(k.startswith("base.") for k in keys) and any(
        k.startswith(("language_head.", "era_head.", "temporal_head.", "hygiene_head."))
        for k in keys
    ):
        return "MultiTaskICF"

    # ResidualICF has distinct block/head naming.
    if any(k.startswith("conv3_block") for k in keys) or any(
        k.startswith("head_linear1") for k in keys
    ):
        return "ResidualICF"

    # NanoICF uses a single `conv` and `head` (no conv3/5/7).
    if (
        "conv.weight" in keys
        and "head.weight" in keys
        and not any(k.startswith("conv3") for k in keys)
    ):
        return "NanoICF"

    # Default: UniversalICF (covers both newer "conv3.0.*" and older "conv3.*" variants).
    return "UniversalICF"


def _infer_model_kwargs_from_state_dict(
    model_type: str, state_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Infer minimal constructor kwargs from tensor shapes when model_kwargs metadata is missing.

    Limits: we can't infer training-only knobs (dropout) reliably, and attention head counts
    are not recoverable from weights alone. Defaults are chosen to match current code.
    """
    keys = set(k for k in state_dict.keys() if isinstance(k, str))

    if model_type == "MultiTaskICF":
        # Best-effort: infer base kwargs from the wrapped `base.*` tensors.
        base_state = {k[len("base.") :]: v for k, v in state_dict.items() if k.startswith("base.")}
        base_type = _infer_model_type_from_state_dict(base_state)
        base_kwargs = _infer_model_kwargs_from_state_dict(base_type, base_state)
        return {
            "output_tasks": ["icf", "language", "era", "temporal", "hygiene"],
            "base_model_kwargs": base_kwargs,
            "num_languages": 10,
            "num_eras": 5,
            "num_hygiene": 8,
            "temporal_decades": (1800, 1900, 2000),
        }

    try:
        emb_w = state_dict.get("emb.weight")
        if hasattr(emb_w, "shape") and len(getattr(emb_w, "shape", ())) == 2:
            vocab_size, emb_dim = int(emb_w.shape[0]), int(emb_w.shape[1])
        else:
            vocab_size, emb_dim = 256, 36
    except Exception:
        vocab_size, emb_dim = 256, 36

    if model_type == "NanoICF":
        conv_w = state_dict.get("conv.weight")
        if hasattr(conv_w, "shape") and len(getattr(conv_w, "shape", ())) == 3:
            conv_channels = int(conv_w.shape[0])
            kernel_size = int(conv_w.shape[2])
        else:
            conv_channels, kernel_size = 32, 5
        # Stride is not stored in state_dict; default matches NanoICF defaults.
        return {
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "conv_channels": conv_channels,
            "kernel_size": kernel_size,
            "stride": 2,
        }

    if model_type == "ResidualICF":
        conv3_w = state_dict.get("conv3_block.0.weight")
        if hasattr(conv3_w, "shape") and len(getattr(conv3_w, "shape", ())) == 3:
            conv_channels = int(conv3_w.shape[0])
        else:
            conv_channels = 18
        head1_w = state_dict.get("head_linear1.weight")
        if hasattr(head1_w, "shape") and len(getattr(head1_w, "shape", ())) == 2:
            hidden_dim = int(head1_w.shape[0])
        else:
            hidden_dim = 36
        return {
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "conv_channels": conv_channels,
            "hidden_dim": hidden_dim,
            "dropout": 0.4,
        }

    # UniversalICF
    # Prefer newer conv key; fall back to older (pre-Sequential) key if present.
    conv3_w = state_dict.get("conv3.0.weight", state_dict.get("conv3.weight"))
    if hasattr(conv3_w, "shape") and len(getattr(conv3_w, "shape", ())) == 3:
        conv_channels = int(conv3_w.shape[0])
    else:
        conv_channels = 18
    head0_w = state_dict.get("head.0.weight")
    if hasattr(head0_w, "shape") and len(getattr(head0_w, "shape", ())) == 2:
        hidden_dim = int(head0_w.shape[0])
    else:
        hidden_dim = 36

    use_attention = any(k.startswith("attention.") for k in keys)

    # We can't recover attention_heads; default matches current code.
    return {
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "conv_channels": conv_channels,
        "hidden_dim": hidden_dim,
        "dropout": 0.4,
        "use_attention": use_attention,
        "attention_heads": 3,
        "output_activation": "clamp",
        "sigmoid_temperature": 1.0,
        # Note: head_in is determined by code (conv_channels*9). If the checkpoint was from
        # an older architecture with different pooling, load_state_dict will fail loudly.
    }


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load a model artifact and normalize it to (checkpoint, state_dict, model_kwargs).

    Returns:
        checkpoint: dict with at least model_type/model_kwargs/model_state_dict keys
        state_dict: dict of tensors suitable for `nn.Module.load_state_dict`
        model_kwargs: kwargs for the model constructor (may be empty)
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    if _is_checkpoint_dict(raw):
        checkpoint: dict[str, Any] = dict(raw)
        state_dict = checkpoint.get("model_state_dict", {})
        if not isinstance(state_dict, dict) or not state_dict:
            # Allow the (unlikely) case where someone stored state_dict at top-level.
            # If it's not present, treat the whole object as state_dict.
            state_dict = raw if isinstance(raw, dict) else {}
        state_dict = _strip_common_prefix(state_dict)
        model_kwargs = checkpoint.get("model_kwargs", {}) or {}
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}
        if "model_type" not in checkpoint:
            checkpoint["model_type"] = _infer_model_type_from_state_dict(state_dict)

        # If model_kwargs is missing/empty, try to infer minimal kwargs from shapes.
        if not model_kwargs:
            model_kwargs = _infer_model_kwargs_from_state_dict(checkpoint["model_type"], state_dict)

        checkpoint["model_kwargs"] = model_kwargs
        checkpoint.setdefault("model_state_dict", state_dict)
        return checkpoint, state_dict, model_kwargs

    # PyTorch Lightning-style checkpoint: weights stored under `state_dict`.
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = _strip_common_prefix(raw["state_dict"])
        model_type = _infer_model_type_from_state_dict(state_dict)
        model_kwargs = _infer_model_kwargs_from_state_dict(model_type, state_dict)
        checkpoint = {
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "model_state_dict": state_dict,
            "format": "lightning_ckpt",
        }
        return checkpoint, state_dict, model_kwargs

    # Plain state_dict only.
    state_dict = _strip_common_prefix(raw if isinstance(raw, dict) else {})
    model_type = _infer_model_type_from_state_dict(state_dict)
    model_kwargs = _infer_model_kwargs_from_state_dict(model_type, state_dict)
    checkpoint = {
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "model_state_dict": state_dict,
        "format": "state_dict_only",
    }
    return checkpoint, state_dict, model_kwargs


def _infer_multitask_kwargs(model_sd: "dict[str, Any]") -> "dict[str, Any]":
    """
    Infer MultiTaskICF constructor kwargs from a (possibly Lightning-stripped) state_dict.
    Handles cases where model_kwargs metadata is missing from the checkpoint.
    """
    kw: "dict[str, Any]" = {}
    # conv_channels: first dim of conv3.0.weight  [out, in, k]
    for prefix in ("", "base."):
        w = model_sd.get(f"{prefix}conv3.0.weight")
        if w is not None and hasattr(w, "shape") and len(getattr(w, "shape", ())) >= 1:
            kw.setdefault("base_model_kwargs", {})["conv_channels"] = int(w.shape[0])
            break
    # hidden_dim: first dim of head.0.weight  [hidden, conv*9]
    for prefix in ("", "base."):
        w = model_sd.get(f"{prefix}head.0.weight")
        if w is not None and hasattr(w, "shape") and len(getattr(w, "shape", ())) >= 1:
            kw.setdefault("base_model_kwargs", {})["hidden_dim"] = int(w.shape[0])
            break
    # num_languages
    lw = model_sd.get("language_head.3.weight")
    if lw is not None and hasattr(lw, "shape"):
        kw["num_languages"] = int(lw.shape[0])
    # temporal_decades
    tw = model_sd.get("temporal_head.3.weight")
    if tw is not None and hasattr(tw, "shape"):
        n = int(tw.shape[0])
        kw["temporal_decades"] = list(range(1800, 1800 + n * 10, 10))
    else:
        kw["temporal_decades"] = []
    # output_tasks
    tasks = ["icf"]
    if "language_head.0.weight" in model_sd:
        tasks.append("language")
    if "era_head.0.weight" in model_sd:
        tasks.append("era")
    if "temporal_head.0.weight" in model_sd and kw.get("temporal_decades"):
        tasks.append("temporal")
    if "hygiene_head.0.weight" in model_sd:
        tasks.append("hygiene")
    kw["output_tasks"] = tasks
    return kw


def load_lightning_checkpoint(
    path: "str | Path",
    *,
    device: "torch.device",
) -> "tuple[torch.nn.Module, dict[str, Any]]":
    """
    Load a Lightning .ckpt file that contains a MultiTaskICF or UniversalICF model.
    Infers architecture dimensions from the state_dict when model_kwargs is absent.
    """
    import torch as _torch

    raw = _torch.load(str(path), map_location=device, weights_only=False)
    sd_full = raw.get("state_dict", {})

    # Extract model sub-dict (strip "model." prefix added by Lightning)
    model_sd = {
        k[6:]: v
        for k, v in sd_full.items()
        if k.startswith("model.") and not k.startswith("model.criterion")
    }
    if not model_sd:
        # No "model." prefix — assume it's a plain state_dict
        model_sd = sd_full

    # Check if it's MultiTaskICF (has language_head or era_head)
    is_multitask = any(
        k.startswith(("language_head.", "era_head.", "temporal_head.", "hygiene_head."))
        for k in model_sd
    )

    if is_multitask:
        from tiny_icf.model_multi_task import MultiTaskICF

        kw = _infer_multitask_kwargs(model_sd)
        model = MultiTaskICF(**kw).to(device)
    else:
        from tiny_icf.model import UniversalICF

        model = UniversalICF().to(device)

    model.load_state_dict(model_sd, strict=False)
    return model, raw


def load_model(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load a model for inference/evaluation from either format.

    Supports:
    - .pt checkpoint dicts (exported by train_all_fronts.py)
    - Lightning .ckpt files (direct trainer checkpoints)

    Returns:
        (model, checkpoint_dict)
    """
    path = Path(path)

    # Lightning .ckpt files have a "state_dict" key
    if path.suffix == ".ckpt":
        return load_lightning_checkpoint(path, device=device)

    checkpoint, state_dict, model_kwargs = load_checkpoint(path, device=device)
    model_type = checkpoint.get("model_type", "UniversalICF")

    if model_type == "ResidualICF":
        from tiny_icf.model_residual import ResidualICF

        model = ResidualICF(**model_kwargs).to(device)
    elif model_type == "NanoICF":
        from tiny_icf.nano_model import NanoICF

        model = NanoICF(**model_kwargs).to(device)
    elif model_type == "MultiTaskICF":
        from tiny_icf.model_multi_task import MultiTaskICF

        model = MultiTaskICF(**model_kwargs).to(device)
        # Use strict=False so older checkpoints missing newly-added modules
        # (e.g. lang_icf_cond) still load without error.
        model.load_state_dict(state_dict, strict=False)
        return model, checkpoint
    else:
        from tiny_icf.model import UniversalICF

        model = UniversalICF(**model_kwargs).to(device)

    model.load_state_dict(state_dict)
    return model, checkpoint
