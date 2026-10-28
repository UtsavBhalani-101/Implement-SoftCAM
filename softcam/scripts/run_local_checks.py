"""Run the CPU-only Phase 0/1 validation suite."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

subprocess.run([sys.executable, str(ROOT / "scripts" / "cpu_smoke_test.py")], check=True)
subprocess.run([
    sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"
], cwd=ROOT, check=True)

print("Phase 0 and Phase 1 checks passed")
