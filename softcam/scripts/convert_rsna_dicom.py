"""Convert RSNA chest X-ray DICOM files to PNG files expected by SoftCAM.

Example:
    python scripts/convert_rsna_dicom.py \
        --input /path/to/stage_2_train_images \
        --output /path/to/rsna-softcam/png_images
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut


def convert_one(source: Path, destination: Path, size: int | None = None) -> None:
    ds = pydicom.dcmread(source)
    pixels = ds.pixel_array

    # Apply the DICOM modality transform before VOI/windowing, as recommended
    # by pydicom. RSNA files are single-frame grayscale images.
    pixels = apply_modality_lut(pixels, ds)
    try:
        pixels = apply_voi_lut(pixels, ds)
    except (AttributeError, ValueError, TypeError):
        # Some files do not contain VOI/window metadata.
        pass

    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {pixels.shape} from {source}")

    low, high = np.percentile(pixels, (1, 99))
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    pixels = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    if size:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def csv_stems(csv_paths: list[Path]) -> set[str]:
    names = set()
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names.add(Path(row["filename"]).stem)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, action="append", default=[],
                        help="CSV split to convert; repeat for train/val/test")
    parser.add_argument("--size", type=int, default=512,
                        help="Resize output to square size; use 0 to preserve size")
    args = parser.parse_args()

    sources = sorted(args.input.rglob("*.dcm"))
    if not sources:
        raise SystemExit(f"No .dcm files found under {args.input}")
    if args.csv:
        wanted = csv_stems(args.csv)
        sources = [source for source in sources if source.stem in wanted]
        print(f"selected {len(sources)} DICOM files from {len(wanted)} CSV entries")

    failures = 0
    for index, source in enumerate(sources, start=1):
        destination = args.output / f"{source.stem}.png"
        try:
            convert_one(source, destination, args.size or None)
        except Exception as exc:  # keep processing and report failed files
            failures += 1
            print(f"FAILED {source}: {exc}")
        if index == 1 or index % 1000 == 0 or index == len(sources):
            print(f"processed {index}/{len(sources)}")

    if failures:
        raise SystemExit(f"Conversion finished with {failures} failures")
    print(f"Converted {len(sources)} DICOM files to {args.output}")


if __name__ == "__main__":
    main()
