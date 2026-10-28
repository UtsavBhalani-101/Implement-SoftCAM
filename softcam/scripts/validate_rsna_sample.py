"""Convert and validate a small RSNA DICOM sample locally.

This is the Phase 2 gate. It intentionally processes only the rows selected
from one repository CSV, then verifies the generated PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from convert_rsna_dicom import convert_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    if not args.dicom_dir.exists():
        raise SystemExit(f"DICOM directory does not exist: {args.dicom_dir}")
    dicom_files = list(args.dicom_dir.rglob("*.dcm"))
    if not dicom_files:
        raise SystemExit(f"No .dcm files found under: {args.dicom_dir}")

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:args.sample_size]

    source_by_stem = {path.stem: path for path in dicom_files}
    report = {
        "csv": str(args.csv),
        "requested": len(rows),
        "converted": [],
        "missing_dicom": [],
        "failed": {},
        "size": args.size,
    }

    for row in rows:
        filename = row["filename"]
        stem = Path(filename).stem
        source = source_by_stem.get(stem)
        destination = args.output / filename
        if source is None:
            report["missing_dicom"].append(filename)
            continue
        try:
            convert_one(source, destination, args.size)
            with Image.open(destination) as image:
                image.load()
                if image.size != (args.size, args.size):
                    raise ValueError(f"unexpected size {image.size}")
                report["converted"].append({
                    "filename": filename,
                    "label": int(row["bin_target"]),
                    "size": list(image.size),
                    "mode": image.mode,
                })
        except Exception as exc:
            report["failed"][filename] = str(exc)

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "sample_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["missing_dicom"] or report["failed"]:
        raise SystemExit("RSNA sample validation failed")
    print(f"RSNA sample validation passed: {len(report['converted'])} images")


if __name__ == "__main__":
    main()
