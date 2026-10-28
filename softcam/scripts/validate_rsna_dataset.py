"""Validate converted RSNA PNGs through the repository's RSNADataset loader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import RSNADataset


def to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", default="png_images")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    cfg = SimpleNamespace(
        base=SimpleNamespace(test=False),
        data=SimpleNamespace(
            input_size=512,
            target="bin_target",
            fname="filename",
        ),
        dset=SimpleNamespace(
            root=str(args.root),
            data_dir=args.data_dir,
            train_csv=str(args.csv),
            val_csv=str(args.csv),
            test_csv=str(args.csv),
        ),
    )

    dataset = RSNADataset(cfg, transform=to_tensor)
    sample_size = min(args.sample_size, len(dataset))
    if sample_size == 0:
        raise SystemExit("CSV contains no rows")

    for index in range(sample_size):
        image, label = dataset[index]
        if tuple(image.shape) != (3, 512, 512):
            raise SystemExit(
                f"Unexpected tensor shape at index {index}: {tuple(image.shape)}"
            )
        if not torch.isfinite(image).all():
            raise SystemExit(f"Non-finite tensor at index {index}")
        if int(label) not in (0, 1):
            raise SystemExit(f"Unexpected label at index {index}: {label}")

    print(f"RSNA dataset validation passed: {sample_size} samples")
    print(f"tensor shape: (3, 512, 512), labels: 0/1, csv: {args.csv}")


if __name__ == "__main__":
    main()
