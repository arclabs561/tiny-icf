"""Adaptive learning rate schedulers for ICF training."""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class AdaptiveCosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing with adaptive restarts based on validation metrics.
    
    Restarts when validation metric plateaus, allowing the model to
    escape local minima and continue learning.
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        restart_threshold: float = 0.01,
        patience: int = 5,
        metric: str = "loss",
        mode: str = "min",
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of iterations per cycle
            eta_min: Minimum learning rate
            restart_threshold: Minimum improvement to avoid restart
            patience: Number of epochs without improvement before restart
            metric: Metric to track ("loss", "spearman", "mae")
            mode: "min" (lower is better) or "max" (higher is better)
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.restart_threshold = restart_threshold
        self.patience = patience
        self.metric = metric
        self.mode = mode
        
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.patience_counter = 0
        self.cycle = 0
        self.last_restart_epoch = 0
        
        super().__init__(optimizer, last_epoch=-1)
    
    def step(self, metrics: dict | None = None, epoch: int | None = None):
        """Step the scheduler, optionally with validation metrics."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        # Check if we should restart
        if metrics is not None:
            current_metric = metrics.get(self.metric)
            if current_metric is not None:
                improved = False
                if self.mode == "min":
                    improved = current_metric < (self.best_metric - self.restart_threshold)
                else:
                    improved = current_metric > (self.best_metric + self.restart_threshold)
                
                if improved:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Restart if patience exhausted
                if self.patience_counter >= self.patience:
                    self.cycle += 1
                    self.last_restart_epoch = epoch
                    self.patience_counter = 0
                    # Reset to initial LR
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = param_group.get("initial_lr", param_group["lr"])
        
        # Cosine annealing within cycle
        if epoch - self.last_restart_epoch < self.T_max:
            super().step(epoch=epoch)
        else:
            # End of cycle, restart
            self.cycle += 1
            self.last_restart_epoch = epoch
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group.get("initial_lr", param_group["lr"])
    
    def get_lr(self):
        """Compute learning rate using cosine annealing."""
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        
        cycle_epoch = self.last_epoch - self.last_restart_epoch
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + torch.cos(torch.tensor(cycle_epoch * 3.14159 / self.T_max, dtype=torch.float32)))
            / 2
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateauSpearman(_LRScheduler):
    """
    Reduce learning rate when Spearman correlation plateaus.
    
    Similar to ReduceLROnPlateau but specifically tuned for
    ranking metrics like Spearman correlation.
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = "max",
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 0.001,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            mode: "max" (higher Spearman is better) or "min"
            factor: Factor to reduce LR by
            patience: Number of epochs without improvement
            threshold: Minimum change to qualify as improvement
            min_lr: Minimum learning rate
            verbose: Print messages when LR is reduced
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1
        
        super().__init__(optimizer)
    
    def step(self, metrics: dict, epoch: int | None = None):
        """Step the scheduler based on validation metrics."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # Get Spearman correlation
        spearman = metrics.get("spearman_corr")
        if spearman is None:
            return
        
        # Check if improved
        if self.best is None:
            self.best = spearman
        else:
            if self.mode == "max":
                improved = spearman > (self.best + self.threshold)
            else:
                improved = spearman < (self.best - self.threshold)
            
            if improved:
                self.best = spearman
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        # Reduce LR if plateau
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0
    
    def _reduce_lr(self, epoch: int):
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr
            
            if self.verbose:
                print(f"Epoch {epoch}: Reducing LR of group {i} from {old_lr:.2e} to {new_lr:.2e}")

