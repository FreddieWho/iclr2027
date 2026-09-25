#!/usr/bin/env python3
"""Execute the frozen Figure-1 selection rule using archived predictions only.

No effect estimate is computed. The bank is historical dev512, seed 11.
The displayed examples are selected by parent id then archived qid, with no
outcome-based tie break beyond the precommitted eligibility rule.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "artifacts/p123_upgrade/bank/bank_dev512.npz"
MANIFEST = ROOT / "artifacts/p123_upgrade/bank/bank_dev512.manifest.json"
PQ = ROOT / "artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv"
POOL = ROOT / "artifacts/discovery_campaign/scenes/eval_202/scenes.npz"
OUT = ROOT / "artifacts/e832_focus/route5_figure"
FIGDIR = ROOT / "paper/figures"
RECEIPT = ROOT / "reports/e832_focus/FIGURE1_SELECTION_RECEIPT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    out = {}
    with PQ.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), row["arm"], int(row["qid"]))
            out[key] = row
    return out


def state(row):
    return (int(row["correct_A"]), int(row["correct_B"]), int(row["correct_AB"]))


def eligible_endpoint(rows):
    out = []
    for qid in sorted({k[2] for k in rows if k[0] == 11 and k[1] == "raw_flipmine"}):
        c = rows.get((11, "raw_clean", qid))
        f = rows.get((11, "raw_flipmine", qid))
        if not c or not f:
            continue
        if state(c) == (1, 1, 0) and int(f["correct_AB"]) == 1 and (
            int(f["correct_A"]) == 0 or int(f["correct_B"]) == 0
        ):
            out.append((int(c["parent"]), qid, "endpoint_only"))
    return sorted(out)


def eligible_full(rows):
    out = []
    for qid in sorted({k[2] for k in rows if k[0] == 11 and k[1] == "relflip"}):
        c = rows.get((11, "relfeat", qid))
        f = rows.get((11, "relflip", qid))
        if not c or not f:
            continue
        if state(c) == (1, 1, 0) and state(f) == (1, 1, 1):
            out.append((int(c["parent"]), qid, "full_repair"))
    return sorted(out)


def quartet_context(qid):
    b = np.load(BANK, allow_pickle=True)
    meta = json.loads(str(b["Qmeta"]))
    rows = [(i, m) for i, m in enumerate(meta) if int(m["qid"]) == qid]
    if len(rows) != 2:
        raise ValueError(f"qid {qid} has {len(rows)} paths, expected 2")
    (i1, m1), (i2, m2) = rows
    qx, qe = np.asarray(b["Qx"], float), np.asarray(b["Qe"], float)
    a, bb, ab = qx[i1], qx[i2], qx[i1] + qe[i1]
    pool = np.load(POOL, allow_pickle=True)
    positions = np.asarray(pool["positions"], float)
    parent = int(m1["parent"])
    if parent >= len(positions):
        raise ValueError("parent id outside eval_202 position pool")
    p = positions[parent]
    if int(m2["parent"]) != parent or not np.allclose(p, a + bb - ab, atol=1e-6, rtol=0):
        raise ValueError("Figure 1 base scene does not match the archived additive quartet")
    la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
    lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
    labels = {
        "P": int(m1["y0"]),
        "A": int(la),
        "B": int(lb),
        "AB": int(m1["yAB"]),
    }
    return {"P": p, "A": a, "B": bb, "AB": ab}, labels, m1, m2


def predicted_label(label, correct):
    """Recover a binary class label from an archived correctness indicator."""
    label, correct = int(label), int(correct)
    if label not in (0, 1) or correct not in (0, 1):
        raise ValueError("Figure 1 requires binary labels and correctness indicators")
    return label if correct else 1 - label


def draw(ax, x, label, pred=None, title="", baseline_pred=None):
    x = np.asarray(x, float).reshape(4, 2)
    # Actual source-task colors: first segment red, second blue.
    for j, color in enumerate(("#c43c35", "#2878b5")):
        ax.plot(x[2 * j:2 * j + 2, 0], x[2 * j:2 * j + 2, 1],
                color=color, lw=2.6, solid_capstyle="round")
    ax.scatter(x[:, 0], x[:, 1], s=13, c=["#c43c35", "#c43c35", "#2878b5", "#2878b5"])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    txt = f"y={label}" if pred is None else f"y={label}  pred={int(baseline_pred)}→{int(pred)}"
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))


def render(selected, rows):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), constrained_layout=True)
    receipt_cases = []
    for r, (parent, qid, kind) in enumerate(selected):
        states, labels, m1, m2 = quartet_context(qid)
        base_arm = "raw_clean" if kind == "endpoint_only" else "relfeat"
        flip_arm = "raw_flipmine" if kind == "endpoint_only" else "relflip"
        base = rows[(11, base_arm, qid)]
        flip = rows[(11, flip_arm, qid)]
        case = {"kind": kind, "parent": parent, "qid": qid,
                "baseline_arm": base_arm, "flip_arm": flip_arm,
                "baseline_state": state(base), "flip_state": state(flip),
                "state_encoding": "A/B/AB correctness indicators, not class predictions",
                "displayed_predictions": {},
                "labels": labels, "source": "bank_dev512", "seed": 11,
                "effect_estimate_used": False}
        receipt_cases.append(case)
        for c, name in enumerate(("P", "A", "B", "AB")):
            pred = None
            baseline_pred = None
            if name in ("A", "B", "AB"):
                col = {"A": "correct_A", "B": "correct_B", "AB": "correct_AB"}[name]
                pred = predicted_label(labels[name], flip[col])
                baseline_pred = predicted_label(labels[name], base[col])
                case["displayed_predictions"][name] = {"baseline": baseline_pred, "flip": pred}
            draw(axes[r, c], states[name], labels[name], pred,
                 f"{kind}: {name}", baseline_pred=baseline_pred)
        axes[r, 0].set_ylabel("baseline→flip", fontsize=8)
    fig.suptitle("Archived repair-flow examples (illustration only)", fontsize=10)
    fig.savefig(FIGDIR / "fig1_real_quartets.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig1_real_quartets.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return receipt_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    endpoint = eligible_endpoint(rows)
    full = eligible_full(rows)
    selected = []
    if endpoint:
        selected.append(endpoint[0])
    if full:
        selected.append(full[0])
    cases = render(selected, rows) if len(selected) == 2 else []
    receipt = {
        "status": "SELECTED" if len(cases) == 2 else "INCOMPLETE",
        "schema": "figure1_binary_predictions_v2",
        "renderer_sha256": sha(Path(__file__)),
        "selection_rule": {
            "bank": "artifacts/p123_upgrade/bank/bank_dev512.npz",
            "prediction_file": "artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv",
            "seed": 11,
            "endpoint": "raw_clean 110 -> raw_flipmine AB=1 and at least one atomic=0",
            "full": "relfeat 110 -> relflip 111",
            "tie_break": "smallest parent id, then archived qid",
            "no_seed_switch": True,
            "no_effect_estimate": True,
        },
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (BANK, MANIFEST, PQ, POOL)},
        "eligible_counts": {"endpoint_only": len(endpoint), "full_repair": len(full)},
        "selected": cases,
        "empty_reason": None if len(cases) == 2 else "One or both precommitted eligible sets were empty; no fallback bank or seed was used.",
    }
    receipt["figure_hashes"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (FIGDIR / "fig1_real_quartets.pdf", FIGDIR / "fig1_real_quartets.png")
    } if len(cases) == 2 else {}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
