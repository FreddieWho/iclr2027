#!/usr/bin/env python3
"""Fig4: four-arm joint consistency (dev + confirmation banks) + full-repair/migration
decomposition on H (baseline-110). Data only from archived frozen JSONs.

Sources:
- dev bank (n_q=237): raw/relfeat arms  artifacts/next_novelty/relfeat/RELFEAT.json
                     raw+flip J          artifacts/next_novelty/p3/P3_FINAL.json (J_repair)
                     relflip dev J + H decomposition  artifacts/next_novelty/relflip/RELFLIP_MIG.json
- confirm bank (n_q=509, bank confirm1007):
                     all four arms       artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json
                     H decomposition     same file, migration block
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = ["s11", "s23", "s47"]
X = np.arange(4)
W = 0.16

# --- load dev bank ---
rf = json.load(open("artifacts/next_novelty/relfeat/RELFEAT.json"))
p3 = json.load(open("artifacts/next_novelty/p3/P3_FINAL.json"))
rl = json.load(open("artifacts/next_novelty/relflip/RELFLIP_MIG.json"))
dev_J = {
    "raw":            [rf[f"{s}_raw"]["J"] for s in SEEDS],
    "raw+flip":       [p3[s]["metrics"]["J_repair"][0] for s in SEEDS],
    "relational":     [rf[f"{s}_relfeat"]["J"] for s in SEEDS],
    "relational+flip":[rl[s]["J_relflip"] for s in SEEDS],
}
dev_flow = {  # on H baseline-110
    "raw+flip":       [(p3[s]["metrics"]["R_full"][0], p3[s]["metrics"]["M_migrate"][0]) for s in SEEDS],
    "relational+flip":[(rl[s]["R_full"][0], rl[s]["M"][0]) for s in SEEDS],
}

# --- load confirmation bank ---
rc = json.load(open("artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json"))
arms = rc["arms"]
conf_J = {
    "raw":            [arms["raw_clean"][s]["J"] for s in SEEDS],
    "raw+flip":       [arms["raw_flipmine"][s]["J"] for s in SEEDS],
    "relational":     [arms["relfeat"][s]["J"] for s in SEEDS],
    "relational+flip":[arms["relflip"][s]["J"] for s in SEEDS],
}
conf_flow = [(rc["migration"][s]["R_full"][0], rc["migration"][s]["M"][0]) for s in SEEDS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

# Panel A: four-arm J, dev (solid) and confirmation (hatched), grouped by seed
for i, s in enumerate(SEEDS):
    devv = [dev_J[k][i] for k in dev_J]
    conv = [conf_J[k][i] for k in conf_J]
    ax1.bar(X + (i - 1) * W - W / 2, devv, W, color="tab:blue", alpha=0.35 + 0.3 * i,
            hatch=None, edgecolor="k", linewidth=0.4, label=f"{s} dev" if i == 0 else None)
    ax1.bar(X + (i - 1) * W + W / 2, conv, W, color="tab:orange", alpha=0.35 + 0.3 * i,
            hatch="//", edgecolor="k", linewidth=0.4, label=f"{s} confirm" if i == 0 else None)
# legend seed shading
from matplotlib.patches import Patch
handles = [Patch(facecolor="tab:blue", alpha=0.35, edgecolor="k", label="dev bank (n=237)"),
           Patch(facecolor="tab:orange", alpha=0.35, edgecolor="k", hatch="//", label="confirm bank (n=509, confirm1007)")]
for i, s in enumerate(SEEDS):
    handles.append(plt.Line2D([], [], marker="s", ls="", color="0.2", alpha=0.35 + 0.3 * i, label=s))
ax1.legend(handles=handles, fontsize=7.5, loc="upper left", ncol=2)
ax1.set_xticks(X)
ax1.set_xticklabels(["raw", "raw\n+flip", "rel.\ninput", "rel.\n+flip"], fontsize=8.5)
ax1.set_ylabel("joint consistency $J=P(\\mathrm{A\\ correct, B\\ correct, AB\\ correct})$")
ax1.set_title("(a) Four arms, matched banks, paired quartets", fontsize=9.5)
ax1.set_ylim(0, 0.56)
ax1.axhline(0.5, color="0.6", lw=0.7, ls="--")
ax1.text(3.45, 0.505, "$J=0.5$", fontsize=7, color="0.4", ha="right")

# Panel B: full-repair / migration decomposition on H
X2 = np.arange(len(SEEDS))
Wb = 0.26
for j, (key, dat) in enumerate(dev_flow.items()):
    full = [d[0] for d in dat]
    mig = [d[1] for d in dat]
    off = (j - 0.5) * Wb
    ax2.bar(X2 + off, full, Wb, color="tab:green", alpha=0.85 if key == "relational+flip" else 0.45,
            edgecolor="k", linewidth=0.4, label="full repair $R_{\\mathrm{full}}$" if j == 0 else None)
    ax2.bar(X2 + off, mig, Wb, bottom=full, color="tab:red", alpha=0.85 if key == "relational+flip" else 0.45,
            edgecolor="k", linewidth=0.4, label="error migration $M$" if j == 0 else None)
    for i, (f, m) in enumerate(zip(full, mig)):
        ax2.text(X2[i] + off, f + m + 0.015, f"{f+m:.2f}", ha="center", fontsize=6.5, color="0.25")
# relflip confirmation flow as hollow markers
for i, (f, m) in enumerate(conf_flow):
    ax2.plot(X2[i] + 1.5 * Wb, f, marker="v", color="tab:green", mec="k", ms=5)
    ax2.plot(X2[i] + 1.5 * Wb, f + m, marker="^", color="tab:red", mec="k", ms=5)
ax2.plot([], [], marker="v", ls="", color="tab:green", mec="k", label="rel.+flip confirm $R_{\\mathrm{full}}$")
ax2.plot([], [], marker="^", ls="", color="tab:red", mec="k", label="rel.+flip confirm $R_{\\mathrm{end}}$")
ax2.set_xticks(X2)
ax2.set_xticklabels(SEEDS, fontsize=9)
ax2.set_ylabel("share of baseline-110 ($H$: A,B correct, AB wrong)")
ax2.set_title("(b) Endpoint repair decomposes on $H$ ($R_{\\mathrm{endpoint}}=R_{\\mathrm{full}}+M$)", fontsize=9.5)
ax2.legend(fontsize=7.5, loc="upper right")
ax2.set_ylim(0, 0.9)

fig.tight_layout()
fig.savefig("paper/figures/fig4_fourarm.pdf")
fig.savefig("paper/figures/fig4_fourarm.png", dpi=150)
print("wrote paper/figures/fig4_fourarm.pdf")
