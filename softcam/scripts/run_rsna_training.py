"""Launch a configured RSNA training job on the current machine or Kaggle.

This wrapper keeps the official ``main.py`` training path intact while adding
two safeguards useful for GPU experiments: explicit config/paths arguments and
an optional CUDA requirement check.
"""

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--paths", required=True, help="Dataset paths YAML")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. This is a GPU training job; run it in a "
            "CUDA-enabled Kaggle/Colab runtime or omit --require-cuda only "
            "for a deliberate CPU smoke run."
        )

    from main import main as train_main

    argv = [
        str(ROOT / "main.py"),
        "-config", str(Path(args.config).resolve()),
        "-paths", str(Path(args.paths).resolve()),
    ]
    if args.print_config:
        argv.append("-print_config")
    sys.argv = argv
    train_main()


if __name__ == "__main__":
    main()
