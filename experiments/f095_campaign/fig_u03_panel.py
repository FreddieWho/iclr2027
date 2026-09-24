#!/usr/bin/env python3
"""FigU03: three-part decomposition of joint correctness (dev512).

Stacked bars per arm/seed from artifacts/f095_campaign/U03/U03_SUMMARY.json:
  1 - J = ordering_failure + global_incompatibility + operating_point_gap.
Provenance: S must equal U2 H and J must equal U1 J_t0 cell-by-cell
(independent-code cross-check); plotted values are the certificate fields.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
U3 = json.load(open(ROOT / "artifacts/f095_campaign/U03/U03_SUMMARY.json"))
U1 = json.load(open(ROOT / "artifacts/next_novelty/u1_factorial/U1_SUMMARY.json"))
U2 = json.load(open(ROOT / "artifacts/next_novelty/u2_ordering/U2_SUMMARY.json"))

ARMS = (("raw_clean", "raw"), ("raw_flipmine", "raw+flip"),
        ("relfeat", "rel"), ("relflip", "rel+flip"))
SEEDS = ("s11", "s23", "s47")

for s in SEEDS:
    for key, _ in ARMS:
        c = U3["seeds"][s][key]
        assert abs(c["S"] - U2["seeds"][s]["arms"][key]["H"]) < 5e-5, (s, key)
        assert abs(c["J_t0"] - U1["seeds"][s]["arms"][key]["J"]) < 5e-5, (s, key)
        # stored fields are rounded to 4dp; allow rounding slack
        assert abs((1 - c["J_t0"]) - (c["ordering_failure"] +
                                       c["global_incompatibility"] +
                                       c["operating_point_gap"])) < 5e-4, (s, key)

fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=True)
for ax, s in zip(axes, SEEDS):
    bot = np.zeros(len(ARMS))
    segs = [("J_t0", "joint correct J", "#2ca02c"),
            ("operating_point_gap", "working point", "#ffbb78"),
            ("global_incompatibility", "no common threshold", "#ff7f0e"),
            ("ordering_failure", "unorderable", "#d62728")]
    x = np.arange(len(ARMS))
    for field, label, color in segs:
        v = np.array([U3["seeds"][s][key][field] for key, _ in ARMS])
        ax.bar(x, v, bottom=bot, label=label, color=color, width=0.62)
        bot += v
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in ARMS], fontsize=9)
    ax.set_title(s, fontsize=10)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("fraction of E quartets", fontsize=10)
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Where joint correctness is lost: per-quartet ordering, "
             "global-threshold incompatibility, working point (dev, E bank)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(ROOT / "paper/figures/figU03_decomp.pdf")
fig.savefig(ROOT / "paper/figures/figU03_decomp.png", dpi=150)
print("wrote paper/figures/figU03_decomp.pdf")
