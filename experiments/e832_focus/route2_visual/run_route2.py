#!/usr/bin/env python3
"""One CPU Route-2 command: fresh data, tests, PILOT_ONLY smoke, GPU preflight."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/e832_focus/route2"
DATA = ART / "data"
BUNDLE = ART / "gpu_bundle"


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(HERE / "data_generator.py"), "--out", str(DATA)
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
        "-p", "test_*.py", "-v",
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "visual_mechanism.py"), "--smoke",
        "--out", str(ART / "cpu_smoke.json"),
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "gpu_preflight.py"), "--data", str(DATA),
        "--out", str(BUNDLE),
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "gpu_run.py"),
        "--manifest", str(BUNDLE / "manifest.json"), "--data", str(DATA),
        "--out", str(ART / "gpu_run"),
    ], check=True)
    print("ROUTE2_COMPLETE: fresh data ready; PILOT_ONLY CPU smoke; formal result DATA_READY_BLOCKED_GPU")


if __name__ == "__main__":
    main()
