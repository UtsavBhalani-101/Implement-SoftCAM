"""Run a tiny CPU end-to-end RSNA experiment.

This deliberately uses ResNet-18, 64x64 images, a few samples, and a few
optimization steps. Its purpose is to validate the complete pipeline locally:
CSV/PNG loading -> training -> checkpoint -> evidence maps -> metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import RSNADataset
from evaluation.explanation_metrics import (
    activation_precision,
    activation_sensitivity,
    top_k_localization_precision,
)
from explainability.softcam_maps import predicted_class_maps
from modules.models import FullyConvModel
from modules.regularization import evidence_penalty


def tensor_transform(image):
    image = image.resize((64, 64), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def make_config(root, data_dir, csv_path):
    return SimpleNamespace(
        base=SimpleNamespace(test=False),
        data=SimpleNamespace(input_size=64, target="bin_target", fname="filename"),
        dset=SimpleNamespace(
            root=str(root),
            data_dir=str(data_dir),
            train_csv=str(csv_path),
            val_csv=str(csv_path),
            test_csv=str(csv_path),
        ),
    )


def save_maps(maps, filenames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for saliency, filename in zip(maps[:, 0], filenames):
        image = (saliency.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(image, mode="L").save(output_dir / filename)


def make_masks(frame, filenames):
    masks = torch.zeros(len(filenames), 64, 64, dtype=torch.bool)
    for index, filename in enumerate(filenames):
        row = frame[frame.filename == filename].iloc[0]
        if pd.notna(row.x) and pd.notna(row.y) and pd.notna(row.width) and pd.notna(row.height):
            x1 = max(0, min(63, round(float(row.x) / 1024 * 64)))
            y1 = max(0, min(63, round(float(row.y) / 1024 * 64)))
            x2 = max(x1 + 1, min(64, round((float(row.x) + float(row.width)) / 1024 * 64)))
            y2 = max(y1 + 1, min(64, round((float(row.y) + float(row.height)) / 1024 * 64)))
            masks[index, y1:y2, x1:x2] = True
    return masks


def train_model(model, loader, softcam, steps, output_dir):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for step, (images, labels) in enumerate(loader):
        if step >= steps:
            break
        if softcam:
            logits, maps = model(images)
            loss = criterion(logits, labels) + evidence_penalty(maps, 1e-4, 1e-4)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    checkpoint = output_dir / ("softcam_resnet18.pt" if softcam else "baseline_resnet18.pt")
    torch.save(model.state_dict(), checkpoint)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", default="png_images")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/local_micro_experiment"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    available = frame[
        frame.filename.map(lambda name: (args.root / args.data_dir / name).exists())
    ].head(args.max_samples).copy()
    if available.empty:
        raise SystemExit("No CSV rows have matching PNG files")

    args.output.mkdir(parents=True, exist_ok=True)
    subset_csv = args.output / "subset.csv"
    available.to_csv(subset_csv, index=False)
    cfg = make_config(args.root, args.data_dir, subset_csv)
    dataset = RSNADataset(cfg, transform=tensor_transform)
    loader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)

    baseline = resnet18(weights=None)
    baseline.fc = nn.Linear(baseline.fc.in_features, 2)
    softcam = FullyConvModel(
        SimpleNamespace(train=SimpleNamespace(network="resnet18")),
        resnet18(weights=None),
        num_classes=2,
    )
    baseline_losses = train_model(baseline, loader, False, args.steps, args.output)
    softcam_losses = train_model(softcam, loader, True, args.steps, args.output)

    images, labels = next(iter(loader))
    with torch.no_grad():
        _, predicted, maps = predicted_class_maps(softcam, images, (64, 64))
    filenames = list(available.filename.iloc[: len(images)])
    masks = make_masks(available, filenames)
    metrics = {
        "activation_precision": float(activation_precision(maps, masks)),
        "activation_sensitivity": float(activation_sensitivity(maps, masks)),
        "top1_localization_precision": float(
            top_k_localization_precision(maps, masks, k=1, patch_size=16)
        ),
    }
    save_maps(maps, [Path(name).stem + "_softcam.png" for name in filenames], args.output / "evidence_maps")
    report = {
        "num_samples": len(dataset),
        "steps": args.steps,
        "input_shape": [3, 64, 64],
        "baseline_losses": baseline_losses,
        "softcam_losses": softcam_losses,
        "predicted_classes": predicted.tolist(),
        "metrics": metrics,
    }
    (args.output / "micro_experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Micro experiment passed; artifacts saved to {args.output}")


if __name__ == "__main__":
    main()
