"""Phase 3 test for CSV-to-image loading and preprocessing."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import csv
import sys
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import RSNADataset


class RSNADatasetTests(unittest.TestCase):
    def test_csv_image_label_and_transform(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "png_images"
            image_dir.mkdir()
            filename = "sample.png"
            Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(
                image_dir / filename
            )

            csv_path = root / "train.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["filename", "bin_target"]
                )
                writer.writeheader()
                writer.writerow({"filename": filename, "bin_target": 1})

            cfg = SimpleNamespace(
                base=SimpleNamespace(test=False),
                data=SimpleNamespace(
                    input_size=8,
                    target="bin_target",
                    fname="filename",
                ),
                dset=SimpleNamespace(
                    root=str(root),
                    data_dir="png_images",
                    train_csv=str(csv_path),
                    val_csv=str(csv_path),
                    test_csv=str(csv_path),
                ),
            )

            def to_tensor(image):
                array = np.asarray(image, dtype=np.float32) / 255.0
                return torch.from_numpy(array).permute(2, 0, 1)

            dataset = RSNADataset(cfg, transform=to_tensor)
            image, label = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(label, 1)
            self.assertTrue(torch.isfinite(image).all().item())


if __name__ == "__main__":
    unittest.main()
