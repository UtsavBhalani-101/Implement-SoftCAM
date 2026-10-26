"""Kaggle entry point for the short one-epoch RSNA experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    sys.argv = [
        str(ROOT / "main.py"),
        "-config", str(ROOT / "configs" / "kaggle_short.yaml"),
        "-paths", str(ROOT / "configs" / "kaggle_short_paths.yaml"),
        "-print_config",
    ]
    main()
