"""Offline audit of published history summaries; no network or model calls.
Inputs were manually transcribed from the pinned GitHub source in README.md.
Utility is a fraction in [0,1]; exported effects and intervals are percentage points.
"""
import csv
import hashlib
import json
import random
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED, REPS = 2026091303, 20000
ARMS = ["direct", "staged2", "rewrite"]
VENDORED_SOURCE = ROOT / "source_v4_confirmation_rows_compact_v2_20260914.jsonl"
SOURCE_PATH = VENDORED_SOURCE
SOURCE_SHA256 = "9a5502b2e00f4f9fcaf2daffeaa820f28e0eeba25cca4eef7d03fc4c3e18497e"

def verify_transcription(rows, source_path=SOURCE_PATH):
    src = Path(source_path)
    global SOURCE_PATH
    SOURCE_PATH = src
    return _verify_transcription(rows)


def _verify_transcription(rows):
    raw = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Pinned source SHA-256 mismatch: {digest}")

    source_rows = {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        obj = json.loads(line)
        if obj.get("record_type") != "p2_history_utility":
            continue
        key = (obj["reader_slot"], obj["history_id"])
        if key in source_rows:
            raise ValueError(f"Duplicate source key at line {line_no}: {key}")
        source_rows[key] = obj

    csv_rows = {(r["reader"], r["history_id"]): r for r in rows}
    if len(rows) != 48 or len(csv_rows) != 48 or set(csv_rows) != set(source_rows):
        raise ValueError(
            f"Source/input row mismatch: input={len(csv_rows)}, source={len(source_rows)}"
        )

    comparisons = 0
    for key, row in csv_rows.items():
        source = source_rows[key]
        for i, arm in enumerate(ARMS):
            utility = source["U"][i]
            cell = row[f"U_{arm}"]
            if (utility is None and cell != "") or (
                utility is not None and float(cell) != utility
            ):
                raise ValueError(f"Utility transcription mismatch: {key}, {arm}")
            if int(row[f"L_{arm}"]) != source["L_native"][i]:
                raise ValueError(f"Length transcription mismatch: {key}, {arm}")
            comparisons += 1
            comparisons += 1
    return comparisons, digest

def quantile(values, p):
    v = sorted(values)
    z = (len(v)-1)*p
    i,j = math.floor(z), math.ceil(z)
    return v[i] if i == j else v[i]+(z-i)*(v[j]-v[i])

def bootstrap(values):
    n = len(values)
    if not n:
        raise ValueError("No valid pairs.")
    rng = random.Random(SEED)
    means = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(REPS)]
    return sum(values)/n, quantile(means,.05), quantile(means,.95)

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Offline paired reanalysis (no network/model calls)')
    ap.add_argument('--source', default=str(VENDORED_SOURCE),
                    help='Pinned source JSONL; default is the vendored in-package copy')
    args = ap.parse_args(argv)
    global SOURCE_PATH
    SOURCE_PATH = Path(args.source)
    with (ROOT/"history_inputs.csv").open(encoding="utf-8",newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 48 or len({(r["reader"],r["history_id"]) for r in rows}) != 48:
        raise ValueError("Expected 48 distinct reader-history records.")
    comparisons, source_digest = verify_transcription(rows, args.source)
    print(
        f"Source transcription: PASS ({len(rows)} rows, {comparisons} fields, "
        f"sha256={source_digest})"
    )
    for r in rows:
        for arm in ARMS:
            key = "U_"+arm
            r[key] = float(r[key]) if r[key] else None
            if r[key] is not None and not 0 <= r[key] <= 1:
                raise ValueError("Utility out of range.")
    result=[]
    for reader in ["R1","R2"]:
        rr=[r for r in rows if r["reader"]==reader]
        for mode in ["three_arm_complete","pair_specific"]:
            for name,a,b in [("SD","staged2","direct"),("RD","rewrite","direct"),
                              ("SR","staged2","rewrite")]:
                required=ARMS if mode=="three_arm_complete" else [a,b]
                usable=[r for r in rr if all(r["U_"+x] is not None for x in required)]
                deltas=[r["U_"+a]-r["U_"+b] for r in usable]
                m,lo,hi=bootstrap(deltas)
                result.append(dict(reader=reader,mode=mode,contrast=name,n=len(deltas),
                    mean_pp=100*m,ci90_low_pp=100*lo,ci90_high_pp=100*hi,
                    bootstrap_seed=SEED,bootstrap_replicates=REPS))
    with (ROOT/"paired_reanalysis.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]))
        w.writeheader();w.writerows(result)
    for r in result:
        print("{reader} {mode} {contrast}: n={n}, {mean_pp:.3f} pp "
              "[{ci90_low_pp:.3f}, {ci90_high_pp:.3f}]".format(**r))

if __name__ == "__main__":
    main()
