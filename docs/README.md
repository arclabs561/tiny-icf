# Documentation

This directory contains organized documentation for the tiny-icf project.

## Structure

- `guides/` - Training guides, quick starts, and how-to documentation
- `results/` - Experiment results, analysis, and progress reports
- `design/` - Design decisions, implementation plans, and product thinking
- `integrations/` - Integration guides (RunPod, Lyceum, Aim, MCP, APIs)

## Essential Documentation

Start here (canonical paths):
- `../README.md` - Main entrypoint (install, quick smoke-test, CLI)
- `PROJECT_OVERVIEW.md` - The “why” and use cases
- `guides/QUICK_START.md` - First real workflow
- `guides/TRAINING_GUIDE.md` - Training strategies and variants
- `guides/DATA_AND_MODELS.md` - Where to get data/models (repo is intentionally lean)
- `guides/EPHEMERAL_TRAINING.md` - Training on ephemeral environments (RunPod)
- `results/EXPERIMENTS.md` - Experiment log / results

Jabberwocky Protocol is implemented in:
- `../scripts/evaluate_model.py` (recommended for trained checkpoints)
- `../tests/test_jabberwocky.py` (pytest-facing contract tests)
