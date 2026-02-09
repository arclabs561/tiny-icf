# UV Workspace Usage

This project uses `uv` for dependency management and script execution.

## Running Scripts

All scripts in the `scripts/` directory use `#!/usr/bin/env -S uv run` shebangs, which means they automatically use the uv workspace environment.

### Direct Execution

Scripts can be executed directly if they're executable:

```bash
./scripts/train_best_practices.py --data data/word_frequency.csv --epochs 100
```

### Explicit uv run

You can also use `uv run` explicitly:

```bash
uv run scripts/train_best_practices.py --data data/word_frequency.csv --epochs 100
```

### Module Execution

For modules in `src/tiny_icf/`, use:

```bash
uv run python -m tiny_icf.train --data data/word_frequency.csv
```

## Benefits

- **Automatic environment**: Scripts automatically use the correct Python environment
- **Dependency isolation**: All dependencies are managed by uv
- **Reproducible**: `uv.lock` ensures consistent environments
- **Fast**: uv is faster than pip for dependency resolution

## Workspace Detection

Since this is a uv workspace (has `uv.lock` and `pyproject.toml`), scripts automatically:
- Use the workspace's virtual environment
- Have access to all project dependencies
- Use the correct Python version

## Scripts Updated

All training, evaluation, and analysis scripts have been updated to use `uv run`:
- `train_best_practices.py`
- `train_adaptive.py`
- `benchmark_training.py`
- `compare_loss_configs.py`
- `comprehensive_eval.py`
- `compare_models.py`
- `quick_test_improvements.py`
- And all other Python scripts in `scripts/`

