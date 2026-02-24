# Documentation Summary

## Essential Documentation
- `../README.md` - Main entrypoint (install, CLI, smoke-test)
- `../docs/PROJECT_OVERVIEW.md` - What we’re building and why
- `../docs/guides/DATA_AND_MODELS.md` - Where data/models come from (repo is intentionally lean)
- `../docs/guides/QUICK_START.md` - Quick start workflow
- `../docs/guides/TRAINING_GUIDE.md` - Detailed training guide
- `../docs/guides/CALIBRATION_AND_RANKING_GUIDE.md` - Calibration and ranking (frequency sampling, differentiable Spearman, learned calibration)
- `../docs/guides/EPHEMERAL_TRAINING.md` - Training on ephemeral environments (RunPod)
- `../docs/results/EXPERIMENTS.md` - Experiment history and results

Evaluation (Jabberwocky, MAE, Spearman):
- `just eval-en` / `just eval-en-spearman` (en models with calibration); `../scripts/evaluate_model.py --model ... --data ... --calibration <path>`
- Calibration improves MAE and Jabberwocky; Spearman may stay similar (see DATA_AND_MODELS).

## Archive
Old documentation and logs are archived in `archive/` directory.
