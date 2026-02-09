# Essential docs and repo layout

The root `README.md` is the entrypoint; everything else is organized under `docs/`.

## Essential documentation (canonical paths)

- `../README.md`: install + CLI + quick smoke-test
- `../docs/PROJECT_OVERVIEW.md`: motivation and use cases
- `../docs/guides/DATA_AND_MODELS.md`: where to get data/models (repo is intentionally lean)
- `../docs/guides/QUICK_START.md`: first workflow
- `../docs/guides/TRAINING_GUIDE.md`: training strategies
- `../docs/guides/EPHEMERAL_TRAINING.md`: RunPod / ephemeral training
- `../docs/results/EXPERIMENTS.md`: experiment log / results

Jabberwocky Protocol is implemented in:
- `../scripts/evaluate_model.py` (recommended for trained checkpoints)
- `../tests/test_jabberwocky.py` (pytest contract)

## Repo layout (high-level)

- `../src/tiny_icf/`: library code (models, loss, eval, CLI entrypoints)
- `../tests/`: test suite (fast, CPU-only by default)
- `../scripts/`: training/eval utilities (intentionally “scratchier”)
- `../docs/`: design notes, guides, and results
