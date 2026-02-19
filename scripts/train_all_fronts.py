#!/usr/bin/env python3
"""
Train a multi-task tiny-icf model that adds:
- token hygiene classification (URLs/emails/code/numbers/etc)
- language + era classification (heuristic labels)
- optional temporal ICF prediction (requires historical CSV)

This uses the existing Lightning infrastructure:
- `MultiTaskIDFDataModule` for data
- `FlexibleIDFLightningModule` + `MultiTaskICF` for modeling
- `UnifiedMultiTaskLoss` for loss aggregation (with optional AMOO weights)

Export:
By default, Lightning writes `.ckpt` files. This script can also export a
portable `.pt` checkpoint dict compatible with `tiny_icf.checkpoint.load_model`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

try:
    from lightning import Trainer
    from lightning.pytorch.callbacks import (
        Callback,
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger

    HAS_LIGHTNING = True
except Exception:
    HAS_LIGHTNING = False

from tiny_icf.checkpoint import load_model
from tiny_icf.flexible_lightning_module import FlexibleIDFLightningModule
from tiny_icf.lightning_data_multi_task import MultiTaskIDFDataModule


def _parse_decades(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _device_accel() -> str:
    if torch.cuda.is_available():
        return "gpu"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def main() -> int:
    if not HAS_LIGHTNING:
        raise RuntimeError("PyTorch Lightning is required. Install `lightning`.")

    p = argparse.ArgumentParser(description="Train multi-task tiny-icf (all fronts)")
    p.add_argument("--data", type=Path, required=True, help="Path to frequency CSV")
    p.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory")
    p.add_argument("--export", type=Path, default=None, help="Optional export .pt checkpoint path")
    p.add_argument("--init-from", type=Path, default=None, help="Optional base UniversalICF .pt")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=20)
    p.add_argument("--augment-prob", type=float, default=0.2)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is recommended on macOS/MPS to avoid spawn/pickling overhead.",
    )
    p.add_argument(
        "--curriculum-stages",
        type=int,
        default=5,
        help="Curriculum stages (set to 1 to disable curriculum and use a single stage).",
    )
    p.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs for curriculum.")
    p.add_argument(
        "--train-max-samples",
        type=int,
        default=200_000,
        help="Cap total stratified training samples (before curriculum splitting).",
    )
    p.add_argument(
        "--val-max-samples",
        type=int,
        default=50_000,
        help="Cap total stratified validation samples.",
    )
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="16-mixed")

    p.add_argument("--multilingual", action="store_true", help="Compute ICF per-language for lang:word keys")
    p.add_argument(
        "--strip-lang-prefix-in-bytes",
        action="store_true",
        help="If set, encode only base token (drop leading lang: prefix) for bytes.",
    )

    # Task toggles
    p.add_argument("--no-language", action="store_true", help="Disable language head/loss")
    p.add_argument("--no-era", action="store_true", help="Disable era head/loss")
    p.add_argument("--hygiene", action="store_true", help="Enable token hygiene head/loss")
    p.add_argument(
        "--hygiene-noise-ratio",
        type=float,
        default=0.25,
        help="When --hygiene is set, add this fraction of noise tokens per epoch (relative to ICF samples).",
    )
    p.add_argument("--temporal", action="store_true", help="Enable temporal head/loss (requires CSV)")
    p.add_argument("--temporal-data", type=Path, default=None, help="Historical ICF CSV with icf_YYYY columns")
    p.add_argument(
        "--temporal-decades",
        type=str,
        default="1800,1900,2000",
        help="Comma-separated decades to train temporal head on",
    )

    # Unified loss weights (simple defaults)
    p.add_argument("--icf-weight", type=float, default=1.0)
    p.add_argument("--language-weight", type=float, default=0.2)
    p.add_argument("--era-weight", type=float, default=0.2)
    p.add_argument("--temporal-weight", type=float, default=0.3)
    p.add_argument("--hygiene-weight", type=float, default=0.2)
    p.add_argument("--use-amoo", action="store_true", help="Use adaptive weighting (AMOO-style)")

    args = p.parse_args()

    decades: Sequence[int] = _parse_decades(args.temporal_decades)

    output_tasks: list[str] = ["icf"]
    if not args.no_language:
        output_tasks.append("language")
    if not args.no_era:
        output_tasks.append("era")
    if args.temporal:
        output_tasks.append("temporal")
    if args.hygiene:
        output_tasks.append("hygiene")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MultiTaskIDFDataModule(
        data_path=args.data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        augment_prob=args.augment_prob,
        num_workers=args.num_workers,
        train_max_samples=int(args.train_max_samples),
        val_max_samples=int(args.val_max_samples),
        curriculum_stages=int(args.curriculum_stages),
        warmup_epochs=int(args.warmup_epochs),
        max_epochs=int(args.epochs),
        include_language=not args.no_language,
        include_era=not args.no_era,
        include_hygiene=bool(args.hygiene),
        hygiene_noise_ratio=float(args.hygiene_noise_ratio) if args.hygiene else 0.0,
        include_temporal=bool(args.temporal),
        temporal_data_path=args.temporal_data,
        temporal_decades=list(decades),
        multilingual=bool(args.multilingual),
        strip_language_prefix_in_bytes=bool(args.strip_lang_prefix_in_bytes),
    )

    config = {
        "use_multi_task_model": True,
        "output_tasks": output_tasks,
        "num_languages": 10,
        "num_eras": 5,
        "num_hygiene": 8,
        "temporal_decades": list(decades),
        # Loss selection
        "use_unified_loss": True,
        "use_amoo": bool(args.use_amoo),
        "icf_weight": float(args.icf_weight),
        "language_weight": float(args.language_weight),
        "era_weight": float(args.era_weight),
        "temporal_weight": float(args.temporal_weight),
        "hygiene_weight": float(args.hygiene_weight),
        # Pair sampling for ICF ranking component
        "n_pairs": 16,
        "min_diff": 0.05,
        "use_weighted_sampling": True,
        "clip_grad_norm": 1.0,
        # Base model construction (UniversalICF kwargs)
        "base_model_kwargs": {
            "use_attention": False,
            "attention_heads": 3,
            "output_activation": "clamp",
            "sigmoid_temperature": 1.0,
        },
    }

    module = FlexibleIDFLightningModule(config=config, learning_rate=args.lr, weight_decay=args.weight_decay)

    # Optional init-from: load a UniversalICF checkpoint into the base model.
    if args.init_from is not None:
        base, _ckpt = load_model(args.init_from, device=torch.device("cpu"))
        if hasattr(module.model, "base"):
            module.model.base.load_state_dict(base.state_dict())

    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="multitask-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    logger = CSVLogger(save_dir=str(args.output_dir / "logs"))

    callbacks = [ckpt_cb, lr_cb, early_cb]

    class _CurriculumCallback(Callback):
        def __init__(self, dm: MultiTaskIDFDataModule):
            super().__init__()
            self.dm = dm

        def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
            self.dm.advance_curriculum()

    reload_every = 0
    if int(args.curriculum_stages) > 1 or (args.hygiene and float(args.hygiene_noise_ratio) > 0.0):
        # We rebuild the training dataloader each epoch so:
        # - the curriculum stage can advance
        # - optional hygiene noise tokens can be re-sampled deterministically per epoch
        reload_every = 1
        callbacks.append(_CurriculumCallback(datamodule))

    trainer = Trainer(
        accelerator=_device_accel(),
        devices=args.devices,
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        benchmark=False,
        reload_dataloaders_every_n_epochs=reload_every,
    )

    trainer.fit(module, datamodule)

    if args.export is not None:
        export_path = args.export
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_ckpt = {
            "model_type": "MultiTaskICF",
            "model_kwargs": {
                "output_tasks": output_tasks,
                "base_model_kwargs": config.get("base_model_kwargs", {}),
                "num_languages": int(config["num_languages"]),
                "num_eras": int(config["num_eras"]),
                "num_hygiene": int(config["num_hygiene"]),
                "temporal_decades": list(decades),
            },
            "model_state_dict": module.model.state_dict(),
            "train_args": vars(args),
        }
        torch.save(export_ckpt, export_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

