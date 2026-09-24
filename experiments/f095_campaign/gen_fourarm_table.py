#!/usr/bin/env python3
"""Regenerate four-arm table cells from per-instance rows (no hand inversion).

Reads artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv and emits
per seed/arm: J, atomic_pass, AB_correct, AB_err. Cross-checks against
U1_SUMMARY.json. Column contract: paper table uses explicit _err values.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "artifacts" / "next_novelty" / "u1_factorial" / "U1_PERQUARTET.csv"
SUM = ROOT / "artifacts" / "next_novelty" / "u1_factorial" / "U1_SUMMARY.json"

ARM_ORDER = ("raw_clean", "raw_flipmine", "relfeat", "relflip")
SEEDS = ("11", "23", "47")


def main():
    rows = list(csv.DictReader(PQ.open()))
    summary = json.loads(SUM.read_text())
    out = {}
    for seed in SEEDS:
        for arm in ARM_ORDER:
            rs = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            n = len(rs)
            ab = sum(int(r["correct_AB"]) for r in rs) / n
            ap = sum(int(r["correct_A"]) and int(r["correct_B"]) for r in rs) / n
            j = sum(int(r["J"]) for r in rs) / n
            ref = summary["seeds"]["s" + seed]["arms"][arm]
            assert abs(ab - ref["AB_correct"]) < 2e-4, (seed, arm, ab, ref["AB_correct"])
            assert abs(ap - ref["atomic_pass"]) < 2e-4, (seed, arm, ap, ref["atomic_pass"])
            assert abs(j - ref["J"]) < 2e-4, (seed, arm, j, ref["J"])
            out[(seed, arm)] = {"n": n, "n_parent": len({r["parent"] for r in rs}), "J": round(j, 4),
                                "atomic_pass": round(ap, 4),
                                "AB_correct": round(ab, 4),
                                "AB_err": round(1 - ab, 4)}
    for arm in ARM_ORDER:
        tag = {"raw_clean": "raw", "raw_flipmine": "raw+flip",
               "relfeat": "relational", "relflip": "relational+flip"}[arm]
        j = "/".join(f"{out[(s, arm)]['J']:.3f}".lstrip("0") for s in SEEDS)
        ap = "/".join(f"{out[(s, arm)]['atomic_pass']:.3f}".lstrip("0") for s in SEEDS)
        e = "/".join(f"{out[(s, arm)]['AB_err']:.3f}".lstrip("0") for s in SEEDS)
        print(f"& {tag} & {j} & {ap} & {e} \\\\")
    print("DENOMINATORS:", json.dumps({f"s{seed}/{arm}": {"n_quartet": v["n"], "n_parent": v["n_parent"]} for (seed, arm), v in out.items()}, sort_keys=True))
    print("OK: all cells regenerated from per-instance rows, match U1_SUMMARY.json")


if __name__ == "__main__":
    main()
