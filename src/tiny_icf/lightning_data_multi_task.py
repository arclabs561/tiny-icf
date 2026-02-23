"""Multi-task Lightning DataModule supporting all tasks."""

from pathlib import Path
from typing import Optional, Dict, List

from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader

from typing import Tuple

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
        train_max_samples: int = 200_000,
        val_max_samples: int = 50_000,
        curriculum_stages: int = 5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        include_language: bool = True,
        include_era: bool = True,
        include_hygiene: bool = False,
        hygiene_noise_ratio: float = 0.0,
        include_temporal: bool = False,
        temporal_data_path: Optional[Path] = None,
        temporal_decades: Optional[List[int]] = None,
        multilingual: bool = False,
        strip_language_prefix_in_bytes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.num_workers = num_workers
        self.train_max_samples = int(train_max_samples)
        self.val_max_samples = int(val_max_samples)
        self.curriculum_stages = curriculum_stages
        self.warmup_epochs = warmup_epochs
        self.max_epochs = int(max_epochs)
        self.include_language = include_language
        self.include_era = include_era
        self.include_hygiene = include_hygiene
        self.hygiene_noise_ratio = float(hygiene_noise_ratio)
        self.include_temporal = include_temporal
        self.temporal_data_path = temporal_data_path
        self.temporal_decades = temporal_decades or [1800, 1900, 2000]
        self.multilingual = multilingual
        self.strip_language_prefix_in_bytes = strip_language_prefix_in_bytes

        self.train_samples: Optional[List[Tuple[str, float]]] = None
        self.val_samples: Optional[List[Tuple[str, float]]] = None
        self.curriculum: Optional[CurriculumSampler] = None
        self.val_dataset: Optional[MultiTaskICFDataset] = None
        self.temporal_data: Optional[Dict[str, Dict[int, float]]] = None
        self._noise_tokens: Optional[List[str]] = None

    def setup(self, stage: str):
        """Load and prepare data."""
        if stage == "fit":
            # Load frequency data.
            # If hygiene is enabled, we also keep an unfiltered view to sample "junk tokens".
            word_counts_raw, _total_raw = load_frequency_list(self.data_path, filter_noise=False)
            word_counts_clean, total_tokens_clean = load_frequency_list(
                self.data_path, filter_noise=True
            )

            # Compute ICF on the *clean* corpus only.
            word_icf = compute_normalized_icf(
                word_counts_clean, total_tokens_clean, multilingual=self.multilingual
            )

            if self.include_hygiene and self.hygiene_noise_ratio > 0.0:
                clean_set = set(word_icf.keys())
                noise_tokens = [w for w in word_counts_raw.keys() if w not in clean_set]
                self._noise_tokens = noise_tokens

            # Load temporal data if enabled
            if (
                self.include_temporal
                and self.temporal_data_path
                and self.temporal_data_path.exists()
            ):
                # Load temporal ICF data (format: word, icf_1800, icf_1900, icf_2000)
                import csv

                self.temporal_data = {}
                with open(self.temporal_data_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        word = row.get("word", "").strip().lower()
                        # Store by base word; dataset will also match multilingual keys via base.
                        if not word:
                            continue

                        decade_map: dict[int, float] = {}
                        for dec in self.temporal_decades:
                            col = f"icf_{int(dec)}"
                            if col not in row:
                                continue
                            raw_val = row.get(col)
                            if raw_val is None:
                                continue
                            raw_val = str(raw_val).strip()
                            if not raw_val:
                                continue
                            try:
                                decade_map[int(dec)] = float(raw_val)
                            except ValueError:
                                continue

                        if decade_map:
                            self.temporal_data[word] = decade_map

            # Split train/val
            import random
            import numpy as np

            random.seed(42)
            np.random.seed(42)
            all_samples = list(word_icf.items())
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * 0.8)
            train_samples_raw = all_samples[:split_idx]
            val_samples_raw = all_samples[split_idx:]

            # Stratified sampling for training (optionally weighted by token frequency so head words like "the" get proper gradient signal)
            train_word_icf = dict(train_samples_raw)
            train_word_counts = {w: word_counts_clean.get(w, 1) for w in train_word_icf}
            self.train_samples = stratified_sample(
                train_word_icf,
                word_counts=train_word_counts,
                use_token_frequency=True,
                max_samples=self.train_max_samples,
            )

            # Create curriculum
            stages = create_curriculum_schedule(
                self.train_samples, num_stages=self.curriculum_stages
            )
            schedule = get_stage_schedule(self.max_epochs, self.curriculum_stages)
            self.curriculum = CurriculumSampler(stages, schedule, warmup_epochs=self.warmup_epochs)

            # Validation set (same frequency-weighted sampling for consistent metrics)
            val_word_icf = dict(val_samples_raw)
            val_word_counts = {w: word_counts_clean.get(w, 1) for w in val_word_icf}
            self.val_samples = stratified_sample(
                val_word_icf,
                word_counts=val_word_counts,
                use_token_frequency=True,
                max_samples=self.val_max_samples,
            )
            self.val_dataset = MultiTaskICFDataset(
                [(w, y, True) for (w, y) in self.val_samples],
                max_length=self.max_length,
                augment_prob=0.0,
                include_language=self.include_language,
                include_era=self.include_era,
                include_hygiene=self.include_hygiene,
                include_temporal=self.include_temporal,
                temporal_data=self.temporal_data,
                temporal_decades=self.temporal_decades,
                strip_language_prefix_in_bytes=self.strip_language_prefix_in_bytes,
            )

    def train_dataloader(self):
        """Get training dataloader for current curriculum stage."""
        if not self.train_samples or not self.curriculum:
            raise RuntimeError("Must call setup('fit') first")

        current_stage_words = self.curriculum.get_current_stage_words()

        # Optionally add hygiene-only noise samples (no ICF supervision).
        samples: list[tuple[str, float, bool]] = [(w, y, True) for (w, y) in current_stage_words]
        if self.include_hygiene and self.hygiene_noise_ratio > 0.0 and self._noise_tokens:
            import random

            random.seed(
                42 + int(getattr(self.curriculum, "current_epoch", 0))
            )  # deterministic per epoch
            n_noise = int(max(0, round(len(samples) * self.hygiene_noise_ratio)))
            if n_noise > 0:
                picked = random.sample(self._noise_tokens, k=min(n_noise, len(self._noise_tokens)))
                samples.extend([(w, 0.5, False) for w in picked])

        train_dataset = MultiTaskICFDataset(
            samples,
            max_length=self.max_length,
            augment_prob=self.augment_prob,
            include_language=self.include_language,
            include_era=self.include_era,
            include_hygiene=self.include_hygiene,
            include_temporal=self.include_temporal,
            temporal_data=self.temporal_data,
            temporal_decades=self.temporal_decades,
            strip_language_prefix_in_bytes=self.strip_language_prefix_in_bytes,
        )

        pin_memory = bool(torch.cuda.is_available())
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_multi_task_batch,  # type: ignore
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        if not self.val_dataset:
            raise RuntimeError("Must call setup('fit') first")

        pin_memory = bool(torch.cuda.is_available())
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_multi_task_batch,  # type: ignore
        )

    def advance_curriculum(self):
        """Advance curriculum to next stage (called after each epoch)."""
        if self.curriculum:
            self.curriculum.advance_epoch()
