"""Universal ICF Model: Byte-level CNN for word frequency estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalICF(nn.Module):
    """
    Universal Frequency Estimator using byte-level CNNs.
    
    Architecture:
    - Byte embedding (256 -> 64)
    - Parallel 1D CNNs (kernel sizes 3, 5, 7) for morphological patterns
    - Global max pooling
    - MLP head with sigmoid output (0.0 = common, 1.0 = rare)
    
    Total parameters: < 50k
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # UTF-8 bytes
        emb_dim: int = 48,  # Reduced from 64
        conv_channels: int = 24,  # Reduced from 32
        hidden_dim: int = 48,  # Reduced from 64
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Byte-level embedding
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Parallel Convs: capture n-grams of different sizes
        # Kernel 3: trigrams like "ing", "pre"
        # Kernel 5: roots like "graph"
        # Kernel 7: complex affixes
        self.conv3 = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, conv_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(emb_dim, conv_channels, kernel_size=7, padding=3)
        
        # MLP Head
        # Use clipped linear instead of sigmoid to avoid saturation
        # Sigmoid saturates at extremes, causing gradient vanishing
        # Input size is now conv_channels * 9 (multi-scale pooling)
        self.head = nn.Sequential(
            nn.Linear(conv_channels * 9, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # No sigmoid - we'll clip in forward pass
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Max_Char_Len] byte indices
        
        Returns:
            [Batch, 1] normalized ICF scores
        """
        # Embed: [Batch, Len] -> [Batch, Len, Emb]
        x_emb = self.emb(x)
        
        # Transpose for Conv1d: [Batch, Emb, Len]
        x_emb = x_emb.transpose(1, 2)
        
        # Extract morphological features
        c3 = F.relu(self.conv3(x_emb))
        c5 = F.relu(self.conv5(x_emb))
        c7 = F.relu(self.conv7(x_emb))
        
        # Multi-scale pooling: Max + Mean + Last token
        # Global max pooling loses positional information
        # Multi-scale captures more information than max alone
        p3_max = F.max_pool1d(c3, c3.size(2)).squeeze(2)  # [Batch, Channels]
        p3_mean = F.avg_pool1d(c3, c3.size(2)).squeeze(2)  # Mean pooling
        p3_last = c3[:, :, -1]  # Last position
        
        p5_max = F.max_pool1d(c5, c5.size(2)).squeeze(2)
        p5_mean = F.avg_pool1d(c5, c5.size(2)).squeeze(2)
        p5_last = c5[:, :, -1]
        
        p7_max = F.max_pool1d(c7, c7.size(2)).squeeze(2)
        p7_mean = F.avg_pool1d(c7, c7.size(2)).squeeze(2)
        p7_last = c7[:, :, -1]
        
        # Concatenate multi-scale features (3 scales × 3 kernels = 9 feature sets)
        combined = torch.cat([
            p3_max, p3_mean, p3_last,
            p5_max, p5_mean, p5_last,
            p7_max, p7_mean, p7_last,
        ], dim=1)  # [Batch, Channels*9]
        
        # Predict ICF with clipped linear (avoids sigmoid saturation)
        output = self.head(combined)  # [Batch, 1]
        # Clip to [0, 1] range (hard clip, no gradient vanishing at extremes)
        return torch.clamp(output, 0.0, 1.0)  # [Batch, 1]
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

