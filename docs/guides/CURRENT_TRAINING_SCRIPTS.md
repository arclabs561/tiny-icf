# Current Training Scripts Status

## ⚠️ IMPORTANT: All Training Now Uses trainctl

**All training scripts have been moved to `../trainctl/training/scripts/`**

### Primary Training Script

**`../trainctl/training/scripts/train_flexible_opportunistic.py`**
- Purpose: Main flexible training script that adapts to resources
- Status: ✅ Current and Active
- Features:
  - Auto-detects GPU/CPU
  - Runs multiple experiments opportunistically
  - Supports ResidualICF and UniversalICF models
  - Uses PyTorch Lightning for multi-GPU
  - Automatic checkpointing and S3 sync
- Usage:
  ```bash
  cd /Users/arc/Documents/dev/trainctl
  uv run training/scripts/train_flexible_opportunistic.py --data ../idf-est/data/word_frequency.csv
  ```

### AWS Management Scripts (in trainctl)

- **`../trainctl/training/scripts/scale_gpu_training.sh`** - Launch/manage AWS instances
- **`../trainctl/training/scripts/monitor_aws_training.sh`** - Monitor training progress
- **`../trainctl/training/scripts/show_training_results.sh`** - Display results
- **`../trainctl/training/scripts/monitor_residual_experiments.sh`** - Monitor experiments

## 📦 Archived/Deprecated Scripts in idf-est

All training scripts in `idf-est/scripts/` are now archived or utility scripts:
- Old training scripts → Archived
- Analysis scripts → Still in idf-est (for analysis only)
- Utility scripts → Still in idf-est (for utilities only)

**Do NOT use scripts in `idf-est/scripts/` for training. Use trainctl instead.**

See `../trainctl/training/README.md` for complete documentation.
- `train_lightning.py` - PyTorch Lightning (experimental)
- `train_multi_loss.py` - Multi-loss (experimental)
- `train_optimized.py` - Optimized (experimental)
- `train_with_eval.py` - With evaluation (experimental)

## 🎯 Recommended Workflow

1. **For new experiments**: Use `scripts/train_*.py` scripts
2. **For ephemeral training**: Use `scripts/train_ephemeral_robust.py`
3. **For baseline comparisons**: Use `scripts/evaluate_with_baselines.py`
4. **For unified interface**: Use `scripts/run_experiment.py`

