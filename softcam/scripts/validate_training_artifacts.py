"""Check that a training job produced the artifacts needed for evaluation."""

import argparse
import json
from pathlib import Path


CHECKPOINTS = [
    "final_weights.pt",
    "best_validation_weights_acc.pt",
    "best_validation_weights_auc.pt",
    "best_validation_weights_loss.pt",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Training output directory")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    output = Path(args.output)
    summary = output / "summarize.csv"
    tensorboard = output / "events.out.tfevents"
    event_files = list(output.glob("events.out.tfevents.*")) if output.exists() else []
    present = {name: (output / name).is_file() for name in CHECKPOINTS}
    report = {
        "output": str(output),
        "exists": output.is_dir(),
        "summary_csv": summary.is_file(),
        "tensorboard_event_files": len(event_files),
        "checkpoints": present,
        "complete": output.is_dir() and summary.is_file() and present["final_weights.pt"],
    }

    print(json.dumps(report, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["complete"]:
        raise SystemExit("Training artifacts are incomplete")
    print("Training artifact validation passed")


if __name__ == "__main__":
    main()
