"""Hierarchical/Box Embedding Model for ICF estimation.

Explores hierarchical embeddings or box embeddings for better representation
while keeping model small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalICF(nn.Module):
    """
    Hierarchical embedding model for ICF estimation.
    
    Uses hierarchical structure:
    - Character level (bytes)
    - N-gram level (morphological patterns)
    - Word level (compositional)
    
    Smaller than UniversalICF but potentially more powerful.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        char_emb_dim: int = 32,  # Reduced from 48
        ngram_emb_dim: int = 16,  # For n-gram patterns
        word_emb_dim: int = 24,  # For word-level features
        hidden_dim: int = 32,  # Reduced from 48
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Character-level embedding (smaller)
        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        
        # Hierarchical: Character -> N-gram -> Word
        # Small convs for n-grams
        self.ngram_conv3 = nn.Conv1d(char_emb_dim, ngram_emb_dim, kernel_size=3, padding=1)
        self.ngram_conv5 = nn.Conv1d(char_emb_dim, ngram_emb_dim, kernel_size=5, padding=2)
        
        # Word-level composition (smaller)
        self.word_conv = nn.Conv1d(ngram_emb_dim * 2, word_emb_dim, kernel_size=3, padding=1)
        
        # Final head (much smaller)
        # Input: word_emb_dim (from pooling) + ngram features
        self.head = nn.Sequential(
            nn.Linear(word_emb_dim + ngram_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Max_Char_Len] byte indices
        
        Returns:
            [Batch, 1] normalized ICF scores
        """
        # Level 1: Character embeddings
        char_emb = self.char_emb(x)  # [Batch, Len, Char_Emb]
        char_emb = char_emb.transpose(1, 2)  # [Batch, Char_Emb, Len]
        
        # Level 2: N-gram patterns
        ngram3 = F.relu(self.ngram_conv3(char_emb))  # [Batch, Ngram_Emb, Len]
        ngram5 = F.relu(self.ngram_conv5(char_emb))  # [Batch, Ngram_Emb, Len]
        
        # Pool n-grams
        ngram3_pool = F.max_pool1d(ngram3, ngram3.size(2)).squeeze(2)  # [Batch, Ngram_Emb]
        ngram5_pool = F.max_pool1d(ngram5, ngram5.size(2)).squeeze(2)  # [Batch, Ngram_Emb]
        
        # Level 3: Word-level composition
        ngram_combined = torch.cat([ngram3, ngram5], dim=1)  # [Batch, Ngram_Emb*2, Len]
        word_features = F.relu(self.word_conv(ngram_combined))  # [Batch, Word_Emb, Len]
        word_pool = F.max_pool1d(word_features, word_features.size(2)).squeeze(2)  # [Batch, Word_Emb]
        
        # Combine all levels
        combined = torch.cat([word_pool, ngram3_pool, ngram5_pool], dim=1)  # [Batch, Word_Emb + Ngram_Emb*2]
        
        # Predict
        output = self.head(combined)
        return torch.clamp(output, 0.0, 1.0)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, mean_icf: float = 0.4):
        """Initialize model weights with proper strategies."""
        def init_layer(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.apply(init_layer)
        
        # Initialize final layer bias
        with torch.no_grad():
            if hasattr(self.head, '__getitem__'):
                final_layer = self.head[-1]
                if isinstance(final_layer, nn.Linear):
                    final_layer.weight.data *= 0.1
                    if final_layer.bias is not None:
                        final_layer.bias.fill_(mean_icf)


class BoxEmbeddingICF(nn.Module):
    """
    Box embedding model for ICF estimation.
    
    Uses box embeddings (hyperrectangles) to represent word frequency ranges.
    More expressive than point embeddings, potentially better for ranking.
    
    Reference: "Learning Representations Using Complex-Valued Box Embeddings"
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        emb_dim: int = 32,
        box_dim: int = 16,  # Dimension for box embeddings
        hidden_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Character embedding
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Convs for patterns
        self.conv3 = nn.Conv1d(emb_dim, box_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, box_dim, kernel_size=5, padding=2)
        
        # Box embedding: each pattern gets a box (min, max)
        # We'll use the pooled features as box centers, and learn widths
        self.box_center = nn.Linear(box_dim * 2, box_dim)
        self.box_width = nn.Linear(box_dim * 2, box_dim)
        
        # Head
        self.head = nn.Sequential(
            nn.Linear(box_dim * 2, hidden_dim),  # center + width
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with box embeddings."""
        # Character embeddings
        char_emb = self.char_emb(x).transpose(1, 2)
        
        # Pattern extraction
        c3 = F.relu(self.conv3(char_emb))
        c5 = F.relu(self.conv5(char_emb))
        
        # Pool
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        p5 = F.max_pool1d(c5, c5.size(2)).squeeze(2)
        
        # Box embeddings
        patterns = torch.cat([p3, p5], dim=1)
        box_center = self.box_center(patterns)
        box_width = F.softplus(self.box_width(patterns))  # Ensure positive width
        
        # Combine center and width
        box_features = torch.cat([box_center, box_width], dim=1)
        
        # Predict
        output = self.head(box_features)
        return torch.clamp(output, 0.0, 1.0)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def init_weights(self, mean_icf: float = 0.4):
        """Initialize model weights with proper strategies."""
        def init_layer(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.apply(init_layer)
        
        # Initialize final layer bias
        with torch.no_grad():
            if hasattr(self.head, '__getitem__'):
                final_layer = self.head[-1]
                if isinstance(final_layer, nn.Linear):
                    final_layer.weight.data *= 0.1
                    if final_layer.bias is not None:
                        final_layer.bias.fill_(mean_icf)

