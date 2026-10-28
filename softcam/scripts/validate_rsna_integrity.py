"""Validate that repository CSV splits match the RSNA DICOM collection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_split(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/rsna_integrity_report.json"))
    args = parser.parse_args()

    if not args.dicom_dir.exists():
        raise SystemExit(f"DICOM directory does not exist: {args.dicom_dir}")
    dicom_stems = {path.stem for path in args.dicom_dir.rglob("*.dcm")}
    if not dicom_stems:
        raise SystemExit(f"No DICOM files found under: {args.dicom_dir}")

    report = {
        "dicom_count": len(dicom_stems),
        "splits": {},
        "duplicate_split_entries": [],
        "unreferenced_dicom_count": 0,
    }
    seen = {}
    referenced = set()
    for csv_path in args.csv:
        rows = read_split(csv_path)
        missing = []
        labels = Counter()
        for row in rows:
            stem = Path(row["filename"]).stem
            referenced.add(stem)
            if stem not in dicom_stems:
                missing.append(row["filename"])
            labels[row.get("bin_target", "unknown")] += 1
            if stem in seen and seen[stem] != str(csv_path):
                report["duplicate_split_entries"].append({
                    "filename": row["filename"],
                    "first": seen[stem],
                    "second": str(csv_path),
                })
            seen[stem] = str(csv_path)
        report["splits"][str(csv_path)] = {
            "rows": len(rows),
            "missing_dicom": missing,
            "label_counts": dict(labels),
        }

    report["unreferenced_dicom_count"] = len(dicom_stems - referenced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    missing = sum(len(split["missing_dicom"]) for split in report["splits"].values())
    if missing or report["duplicate_split_entries"]:
        raise SystemExit("RSNA integrity validation failed")
    print("RSNA integrity validation passed")


if __name__ == "__main__":
    main()
