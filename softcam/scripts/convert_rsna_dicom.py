"""Convert RSNA chest X-ray DICOM files to PNG files expected by SoftCAM.

Example:
    python scripts/convert_rsna_dicom.py \
        --input /path/to/stage_2_train_images \
        --output /path/to/rsna-softcam/png_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut


def convert_one(source: Path, destination: Path) -> None:
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = sorted(args.input.rglob("*.dcm"))
    if not sources:
        raise SystemExit(f"No .dcm files found under {args.input}")

    failures = 0
    for index, source in enumerate(sources, start=1):
        destination = args.output / f"{source.stem}.png"
        try:
            convert_one(source, destination)
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
