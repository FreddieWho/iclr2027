#!/usr/bin/env python3
"""Verify dataset paths against configs/data_manifest.yaml without modifying data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config: dict[str, Any] = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    package_root = args.manifest.resolve().parents[1]
    root = args.data_root or package_root / config.get("root", "data/raw")

    results: list[dict[str, Any]] = []
    required_failures = 0
    for name, spec in config["datasets"].items():
        base = root / spec["path"]
        markers = spec.get("markers", [])
        marker_status = {m: (base / m).exists() for m in markers}
        present = base.exists() and all(marker_status.values())
        required = bool(spec.get("required", False))
        if required and not present:
            required_failures += 1
        results.append(
            {
                "dataset": name,
                "required": required,
                "present": present,
                "path": str(base),
                "markers": marker_status,
                "source": spec.get("source"),
            }
        )

    width = max(len(r["dataset"]) for r in results)
    for r in results:
        status = "OK" if r["present"] else ("MISSING-REQUIRED" if r["required"] else "missing-optional")
        print(f"{r['dataset']:<{width}}  {status:<16}  {r['path']}")

    payload = {"root": str(root), "required_failures": required_failures, "datasets": results}
    out = args.json_out or package_root / "data" / "checksums" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVerification report: {out}")
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
