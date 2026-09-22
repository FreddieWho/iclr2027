#!/usr/bin/env python3
"""Fig4: four-arm joint consistency (dev + confirmation banks) + full-repair/migration
decomposition on H (baseline-110) + J/H ordering panel. Data only from archived
frozen JSONs and U1/U2 re-derivations (bit-verified against the archives).

Sources:
- dev bank (n_q=237): four-arm J + H decomposition
                     artifacts/next_novelty/u1_factorial/U1_SUMMARY.json
                     (asserted bit-equal to RELFEAT.json / p3/P3_FINAL.json /
                      relflip/RELFLIP_MIG.json before plotting)
- dev J/H ordering:  artifacts/next_novelty/u2_ordering/U2_SUMMARY.json
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
U1ARM = {"raw": "raw_clean", "raw+flip": "raw_flipmine",
         "relational": "relfeat", "relational+flip": "relflip"}

# --- load dev bank from U1 re-derivation ---
u1 = json.load(open("artifacts/next_novelty/u1_factorial/U1_SUMMARY.json"))
u2 = json.load(open("artifacts/next_novelty/u2_ordering/U2_SUMMARY.json"))

# --- provenance assertions: U1 must reproduce the frozen archives ---
rf = json.load(open("artifacts/next_novelty/relfeat/RELFEAT.json"))
p3 = json.load(open("artifacts/next_novelty/p3/P3_FINAL.json"))
rl = json.load(open("artifacts/next_novelty/relflip/RELFLIP_MIG.json"))
for s in SEEDS:
    a = u1["seeds"][s]["arms"]
    assert a["raw_clean"]["J"] == rf[f"{s}_raw"]["J"], s
    assert a["raw_flipmine"]["J"] == p3[s]["metrics"]["J_repair"][0], s
    assert a["relfeat"]["J"] == rf[f"{s}_relfeat"]["J"], s
    assert a["relflip"]["J"] == rl[s]["J_relflip"], s
    h = u1["seeds"][s]["repair_vs_rawclean_H"]["relflip"]
    assert h["R_full"][0] == rl[s]["R_full"][0], s
    assert h["M_migrate"][0] == rl[s]["M"][0], s

dev_J = {k: [u1["seeds"][s]["arms"][v]["J"] for s in SEEDS]
         for k, v in U1ARM.items()}
dev_H = {k: [u2["seeds"][s]["arms"][v]["H"] for s in SEEDS]
         for k, v in U1ARM.items()}
dev_flow = {}
for key, uarm in (("raw+flip", "raw_flipmine"),
                  ("relational+flip", "relflip")):
    dev_flow[key] = [(u1["seeds"][s]["repair_vs_rawclean_H"][uarm]["R_full"][0],
                      u1["seeds"][s]["repair_vs_rawclean_H"][uarm]["M_migrate"][0])
                     for s in SEEDS]

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2),
                                    gridspec_kw={"width_ratios": [1.35, 1, 1.1]})

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

# Panel C: J (fixed threshold) vs H (oracle separability) on dev
order = ["raw", "raw+flip", "relational", "relational+flip"]
for i, s in enumerate(SEEDS):
    jj = [dev_J[k][i] for k in order]
    hh = [dev_H[k][i] for k in order]
    ax3.plot(order, jj, marker="o", ms=4, lw=1.2, color=f"C{i}",
             label=f"{s} $J$ (fixed thr.)")
    ax3.plot(order, hh, marker="s", ms=4, lw=1.2, ls="--", color=f"C{i}",
             label=f"{s} $H$ (oracle sep.)")
ax3.set_ylabel("rate on dev bank (n=237)")
ax3.set_title("(c) Fixed-threshold $J$ vs oracle separability $H$", fontsize=9.5)
ax3.legend(fontsize=7, loc="upper left", ncol=2)
ax3.set_ylim(0, 1.0)
ax3.tick_params(axis="x", labelsize=8)

fig.tight_layout()
fig.savefig("paper/figures/fig4_fourarm.pdf")
fig.savefig("paper/figures/fig4_fourarm.png", dpi=150)
print("wrote paper/figures/fig4_fourarm.pdf")
