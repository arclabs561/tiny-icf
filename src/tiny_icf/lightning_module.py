"""PyTorch Lightning module for IDF estimation training."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF


class IDFLightningModule(LightningModule):
    """Lightning module for Universal ICF model."""
    
    def __init__(
        self,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = UniversalICF()
        self.criterion = CombinedLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        
        # For validation metrics
        self.validation_predictions = []
        self.validation_targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        byte_tensors, icf_targets = batch
        predictions = self(byte_tensors)
        
        # Generate ranking pairs
        n_pairs = len(icf_targets) // 2
        pairs = self._generate_ranking_pairs(icf_targets, n_pairs)
        
        loss = self.criterion(predictions, icf_targets, pairs=pairs)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        byte_tensors, icf_targets = batch
        predictions = self(byte_tensors)
        
        # Generate ranking pairs
        n_pairs = len(icf_targets) // 2
        pairs = self._generate_ranking_pairs(icf_targets, n_pairs)
        
        loss = self.criterion(predictions, icf_targets, pairs=pairs)
        
        # Store for metrics computation
        self.validation_predictions.append(predictions.detach().cpu())
        self.validation_targets.append(icf_targets.detach().cpu())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at end of epoch."""
        if not self.validation_predictions:
            return
        
        pred_tensor = torch.cat(self.validation_predictions).squeeze()
        target_tensor = torch.cat(self.validation_targets).squeeze()
        
        # MAE
        mae = torch.mean(torch.abs(pred_tensor - target_tensor))
        self.log("val_mae", mae, prog_bar=True)
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred_tensor - target_tensor) ** 2))
        self.log("val_rmse", rmse)
        
        # Spearman correlation (if scipy available)
        try:
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(
                pred_tensor.numpy(),
                target_tensor.numpy()
            )
            self.log("val_spearman_corr", corr)
            self.log("val_spearman_p", p_value)
            if corr > 0.8:
                self.log("val_excellent_corr", 1.0)
        except ImportError:
            pass
        
        # Prediction statistics
        self.log("val_pred_min", pred_tensor.min())
        self.log("val_pred_max", pred_tensor.max())
        self.log("val_pred_std", pred_tensor.std())
        
        # Clear for next epoch
        self.validation_predictions.clear()
        self.validation_targets.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def _generate_ranking_pairs(self, targets: torch.Tensor, n_pairs: int) -> torch.Tensor:
        """Generate pairs for ranking loss."""
        batch_size = len(targets)
        if batch_size < 2:
            return torch.empty((0, 2), dtype=torch.long, device=targets.device)
        
        pairs = []
        if targets.dim() > 1 and targets.size(1) == 1:
            targets_flat = targets.squeeze(1)
        else:
            targets_flat = targets
        
        for _ in range(n_pairs):
            i, j = torch.randint(0, batch_size, (2,), device=targets.device)
            if i == j:
                continue
            if targets_flat[i] < targets_flat[j]:
                pairs.append([i.item(), j.item()])
            elif targets_flat[j] < targets_flat[i]:
                pairs.append([j.item(), i.item()])
        
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long, device=targets.device)
        
        return torch.tensor(pairs, dtype=torch.long, device=targets.device)

