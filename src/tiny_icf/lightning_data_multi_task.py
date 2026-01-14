"""Multi-task Lightning DataModule supporting all tasks."""

from pathlib import Path
from typing import Optional, Dict, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from typing import Optional, Dict, List, Tuple

from tiny_icf.data import compute_normalized_icf, load_frequency_list, stratified_sample
from tiny_icf.data_multi_task import MultiTaskICFDataset, collate_multi_task_batch
from tiny_icf.curriculum import create_curriculum_schedule, CurriculumSampler, get_stage_schedule


class MultiTaskIDFDataModule(LightningDataModule):
    """DataModule for multi-task training (ICF, language, era, temporal)."""
    
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 256,
        max_length: int = 20,
        augment_prob: float = 0.2,
        num_workers: int = 4,
        curriculum_stages: int = 5,
        warmup_epochs: int = 5,
        include_language: bool = True,
        include_era: bool = True,
        include_temporal: bool = False,
        temporal_data_path: Optional[Path] = None,
        multilingual: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.num_workers = num_workers
        self.curriculum_stages = curriculum_stages
        self.warmup_epochs = warmup_epochs
        self.include_language = include_language
        self.include_era = include_era
        self.include_temporal = include_temporal
        self.temporal_data_path = temporal_data_path
        self.multilingual = multilingual
        
        self.train_samples: Optional[List[Tuple[str, float]]] = None
        self.val_samples: Optional[List[Tuple[str, float]]] = None
        self.curriculum: Optional[CurriculumSampler] = None
        self.val_dataset: Optional[MultiTaskICFDataset] = None
        self.temporal_data: Optional[Dict[str, Dict[int, float]]] = None
    
    def setup(self, stage: str):
        """Load and prepare data."""
        if stage == "fit":
            # Load frequency data
            word_counts, total_tokens = load_frequency_list(self.data_path)
            
            # Compute ICF
            word_icf = compute_normalized_icf(
                word_counts, total_tokens, multilingual=self.multilingual
            )
            
            # Load temporal data if enabled
            if self.include_temporal and self.temporal_data_path and self.temporal_data_path.exists():
                # Load temporal ICF data (format: word, icf_1800, icf_1900, icf_2000)
                import csv
                self.temporal_data = {}
                with open(self.temporal_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word = row.get('word', '').strip().lower()
                        if word in word_icf:
                            self.temporal_data[word] = {
                                1800: float(row.get('icf_1800', word_icf[word])),
                                1900: float(row.get('icf_1900', word_icf[word])),
                                2000: float(row.get('icf_2000', word_icf[word])),
                            }
            
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
            
            # Create curriculum
            stages = create_curriculum_schedule(self.train_samples, num_stages=self.curriculum_stages)
            schedule = get_stage_schedule(50, self.curriculum_stages)
            self.curriculum = CurriculumSampler(stages, schedule, warmup_epochs=self.warmup_epochs)
            
            # Validation set
            self.val_samples = val_samples_raw
            self.val_dataset = MultiTaskICFDataset(
                self.val_samples,
                max_length=self.max_length,
                augment_prob=0.0,
                include_language=self.include_language,
                include_era=self.include_era,
                include_temporal=self.include_temporal,
                temporal_data=self.temporal_data,
            )
    
    def train_dataloader(self):
        """Get training dataloader for current curriculum stage."""
        if not self.train_samples or not self.curriculum:
            raise RuntimeError("Must call setup('fit') first")
        
        current_stage_words = self.curriculum.get_current_stage_words()
        
        train_dataset = MultiTaskICFDataset(
            current_stage_words,
            max_length=self.max_length,
            augment_prob=self.augment_prob,
            include_language=self.include_language,
            include_era=self.include_era,
            include_temporal=self.include_temporal,
            temporal_data=self.temporal_data,
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_multi_task_batch,  # type: ignore
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
            collate_fn=collate_multi_task_batch,  # type: ignore
        )
    
    def advance_curriculum(self):
        """Advance curriculum to next stage (called after each epoch)."""
        if self.curriculum:
            self.curriculum.advance_epoch()

