"""PyTorch Lightning DataModule for IDF estimation."""

from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from tiny_icf.curriculum import create_curriculum_schedule, CurriculumSampler, get_stage_schedule
from tiny_icf.data import WordICFDataset, compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.data_universal import UniversalICFDataset, load_frequency_list_with_emojis


class IDFDataModule(LightningDataModule):
    """DataModule for IDF estimation with curriculum learning."""
    
    def __init__(
        self,
        data_path: Path,
        typo_corpus_path: Optional[Path] = None,
        emoji_freq_path: Optional[Path] = None,
        batch_size: int = 256,
        max_length: int = 20,
        augment_prob: float = 0.2,
        num_workers: int = 4,
        curriculum_stages: int = 5,
        warmup_epochs: int = 5,
        multilingual: bool = False,
        include_symbols: bool = True,
        include_emojis: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.typo_corpus_path = typo_corpus_path
        self.emoji_freq_path = emoji_freq_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.num_workers = num_workers
        self.curriculum_stages = curriculum_stages
        self.warmup_epochs = warmup_epochs
        self.multilingual = multilingual
        self.include_symbols = include_symbols
        self.include_emojis = include_emojis
        
        self.train_samples = None
        self.val_samples = None
        self.curriculum = None
        self.val_dataset = None
    
    def setup(self, stage: str):
        """Load and prepare data."""
        if stage == "fit":
            # Load frequency data
            if self.emoji_freq_path:
                word_counts, total_tokens = load_frequency_list_with_emojis(
                    self.data_path, self.emoji_freq_path
                )
            else:
                word_counts, total_tokens = load_frequency_list(self.data_path)
            
            # Compute ICF
            word_icf = compute_normalized_icf(
                word_counts, total_tokens, multilingual=self.multilingual
            )
            
            # Split train/val
            import random
            random.seed(42)
            all_samples = list(word_icf.items())
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * 0.8)
            train_samples_raw = all_samples[:split_idx]
            val_samples_raw = all_samples[split_idx:]
            
            # Stratified sampling for training
            train_word_icf = dict(train_samples_raw)
            train_word_counts = {word: word_counts.get(word, 1) for word in train_word_icf.keys()}
            self.train_samples = stratified_sample(
                train_word_icf, word_counts=train_word_counts, use_token_frequency=False
            )
            
            # Create curriculum (schedule will be set by trainer max_epochs)
            stages = create_curriculum_schedule(self.train_samples, num_stages=self.curriculum_stages)
            # Use a default schedule for now, will be updated if trainer provides max_epochs
            schedule = get_stage_schedule(50, self.curriculum_stages)
            self.curriculum = CurriculumSampler(stages, schedule, warmup_epochs=self.warmup_epochs)
            
            # Validation set
            self.val_samples = val_samples_raw
            self.val_dataset = WordICFDataset(
                self.val_samples, max_length=self.max_length, augment_prob=0.0
            )
    
    def train_dataloader(self):
        """Get training dataloader for current curriculum stage."""
        if not self.train_samples or not self.curriculum:
            raise RuntimeError("Must call setup('fit') first")
        
        current_stage_words = self.curriculum.get_current_stage_words()
        
        train_dataset = UniversalICFDataset(
            current_stage_words,
            max_length=self.max_length,
            augment_prob=self.augment_prob,
            typo_corpus_path=self.typo_corpus_path,
            emoji_freq_path=self.emoji_freq_path,
            include_symbols=self.include_symbols,
            include_emojis=self.include_emojis,
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        if not self.val_dataset:
            raise RuntimeError("Must call setup('fit') first")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def advance_curriculum(self):
        """Advance curriculum to next stage (called after each epoch)."""
        if self.curriculum:
            self.curriculum.advance_epoch()

