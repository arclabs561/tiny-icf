#!/usr/bin/env python3
"""
Non-interactive RunPod batch training entry point.
Delegates to tiny_icf.train_lightning. Run from project root:
  uv run python scripts/train_batch.py ...
  or: python scripts/train_batch.py ...  (with venv activated)
"""

import sys
from pathlib import Path

# Ensure package is importable when run from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_icf.train_lightning import main

if __name__ == "__main__":
    sys.exit(main())
