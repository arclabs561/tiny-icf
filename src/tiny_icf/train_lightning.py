"""Non-interactive training script using PyTorch Lightning for RunPod batch jobs."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

try:
    from lightning import Trainer
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    print("⚠️  PyTorch Lightning not installed. Install with: pip install lightning")

from tiny_icf.curriculum import get_stage_schedule
from tiny_icf.lightning_data import IDFDataModule
from tiny_icf.lightning_module import IDFLightningModule


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


from lightning.pytorch.callbacks import Callback


class CurriculumCallback(Callback):
    """Callback to advance curriculum after each epoch."""
    
    def __init__(self, datamodule: IDFDataModule):
        super().__init__()
        self.datamodule = datamodule
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Advance curriculum after each training epoch."""
        self.datamodule.advance_curriculum()


def main():
    if not HAS_LIGHTNING:
        print("❌ PyTorch Lightning is required for this script")
        print("   Install with: pip install lightning")
        return 1
    
    parser = argparse.ArgumentParser(description="Lightning training for RunPod batch jobs")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--augment-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--curriculum-stages", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--typo-corpus", type=Path)
    parser.add_argument("--emoji-freq", type=Path)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--include-symbols", action="store_true", default=True)
    parser.add_argument("--include-emojis", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="16-mixed", help="16-mixed, 32, or bf16-mixed")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data module
    datamodule = IDFDataModule(
        data_path=args.data,
        typo_corpus_path=args.typo_corpus,
        emoji_freq_path=args.emoji_freq,
        batch_size=args.batch_size,
        max_length=args.max_length,
        augment_prob=args.augment_prob,
        num_workers=args.num_workers,
        curriculum_stages=args.curriculum_stages,
        warmup_epochs=args.warmup_epochs,
        multilingual=args.multilingual,
        include_symbols=args.include_symbols,
        include_emojis=args.include_emojis,
    )
    
    # Model
    model = IDFLightningModule(
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="tiny-icf-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = CSVLogger(save_dir=args.output_dir / "logs")
    
    # Setup data first (needed for callback and to update curriculum schedule)
    datamodule.setup("fit")
    
    # Update curriculum schedule with actual max_epochs
    if datamodule.curriculum:
        schedule = get_stage_schedule(args.epochs, args.curriculum_stages)
        datamodule.curriculum.schedule = schedule
    
    # Add curriculum callback
    curriculum_callback = CurriculumCallback(datamodule)
    callbacks = [checkpoint_callback, early_stopping, lr_monitor, curriculum_callback]
    
    # Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        benchmark=False,  # For reproducibility
    )
    
    print("🚀 Starting Lightning training...")
    print(f"   Output: {args.output_dir}")
    print(f"   Devices: {args.devices}")
    print(f"   Precision: {args.precision}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    
    # Train
    trainer.fit(model, datamodule)
    
    # Save final model
    final_model_path = args.output_dir / "model_final.pt"
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\n✅ Training complete!")
    print(f"   Best model: {checkpoint_callback.best_model_path}")
    print(f"   Final model: {final_model_path}")
    
    # Exit cleanly (important for non-interactive batch jobs)
    return 0


if __name__ == "__main__":
    exit(main())

