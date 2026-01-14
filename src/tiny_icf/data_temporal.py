"""Temporal dataset for historical ICF training."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TemporalICFDataset(Dataset):
    """
    Dataset that includes historical ICF scores across multiple decades.
    
    Supports multi-objective training where we predict ICF for multiple time periods.
    """
    
    def __init__(
        self,
        word_icf_pairs: List[Tuple[str, float]],
        historical_data: Optional[pd.DataFrame] = None,
        decades: Optional[List[int]] = None,
        max_length: int = 20,
    ):
        """
        Args:
            word_icf_pairs: List of (word, current_icf) pairs
            historical_data: DataFrame with columns: word, icf_1800, icf_1900, icf_2000, etc.
            decades: List of decades to include (e.g., [1800, 1900, 2000])
            max_length: Maximum word length
        """
        self.word_icf_pairs = word_icf_pairs
        self.max_length = max_length
        
        # Create word -> ICF mapping
        self.word_to_icf = {word: icf for word, icf in word_icf_pairs}
        
        # Load historical data if provided
        self.historical_data = {}
        self.decades = decades or []
        
        if historical_data is not None:
            for _, row in historical_data.iterrows():
                word = row['word']
                if word not in self.word_to_icf:
                    continue
                
                decade_icfs = {}
                for decade in self.decades:
                    col_name = f'icf_{decade}'
                    if col_name in row and pd.notna(row[col_name]):
                        decade_icfs[decade] = float(row[col_name])
                
                if decade_icfs:
                    self.historical_data[word] = decade_icfs
    
    def __len__(self) -> int:
        return len(self.word_icf_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        word, icf = self.word_icf_pairs[idx]
        
        # Convert word to bytes
        word_bytes = word.encode('utf-8')[:self.max_length]
        byte_tensor = torch.zeros(self.max_length, dtype=torch.long)
        for i, b in enumerate(word_bytes):
            byte_tensor[i] = b
        
        result = {
            'word': word,
            'bytes': byte_tensor,
            'icf': torch.tensor(icf, dtype=torch.float32),
        }
        
        # Add historical ICF scores if available
        if word in self.historical_data:
            for decade, hist_icf in self.historical_data[word].items():
                result[f'icf_{decade}'] = torch.tensor(hist_icf, dtype=torch.float32)
        
        return result
    
    @classmethod
    def from_files(
        cls,
        current_data_path: Path,
        historical_data_path: Optional[Path] = None,
        decades: Optional[List[int]] = None,
        max_length: int = 20,
    ) -> 'TemporalICFDataset':
        """Load dataset from CSV files."""
        # Load current ICF data
        df_current = pd.read_csv(current_data_path)
        word_icf_pairs = [
            (row['word'], row['icf_score'])
            for _, row in df_current.iterrows()
        ]
        
        # Load historical data if provided
        historical_data = None
        if historical_data_path and historical_data_path.exists():
            historical_data = pd.read_csv(historical_data_path)
        
        return cls(
            word_icf_pairs=word_icf_pairs,
            historical_data=historical_data,
            decades=decades,
            max_length=max_length,
        )


def load_historical_icf_data(
    historical_csv: Path,
    decades: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Load historical ICF data from CSV.
    
    Returns:
        Dict mapping word -> Dict mapping decade -> ICF score
    """
    df = pd.read_csv(historical_csv)
    
    historical = {}
    for _, row in df.iterrows():
        word = row['word']
        decade_icfs = {}
        
        for decade in decades:
            col_name = f'icf_{decade}'
            if col_name in row and pd.notna(row[col_name]):
                decade_icfs[decade] = float(row[col_name])
        
        if decade_icfs:
            historical[word] = decade_icfs
    
    return historical

