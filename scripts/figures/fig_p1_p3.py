#!/usr/bin/env python3
"""Fig2 (P1 dual curves) + Fig3 (P3 migration bars). Data only from frozen JSONs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "artifacts/next_novelty/p1b_res/P1B.json"
d = json.load(open(R))
v = d["s11_clean"]
xs = [0.125, 0.375, 0.625, 0.875]
fig, ax = plt.subplots(figsize=(7, 4.2))
for key, lab, mk in (("current_by_bin", "static error", "o"),
                     ("preserve_by_bin", "preserve error", "s"),
                     ("flip_by_bin", "flip update error", "^")):
    ys = [b["err"] for b in v[key]]
    lo = [b["ci"][0] for b in v[key]]
    hi = [b["ci"][1] for b in v[key]]
    ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - np.array(ys)],
                marker=mk, capsize=3, label=lab)
ax.set_xlabel("start-confidence quartile (low → high)")
ax.set_ylabel("error rate")
ax.set_title("Confidence predicts static correctness, reverses under true change (s11 clean, dev)")
ax.legend()
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig("paper/figures/fig2_p1_dual.pdf")
fig.savefig("paper/figures/fig2_p1_dual.png", dpi=150)

p3 = json.load(open("artifacts/next_novelty/p3/P3_FINAL.json"))
seeds = ["s11", "s23", "s47"]
labels = ["R_endpoint", "R_full", "M_migrate"]
vals = {L: [p3[s]["metrics"][L][0] for s in seeds] for L in labels}
cis = {L: [p3[s]["metrics"][L][1] for s in seeds] for L in labels}
x = np.arange(3)
w = 0.25
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, L in enumerate(labels):
    y = vals[L]
    lo = [c[0] for c in cis[L]]
    hi = [c[1] for c in cis[L]]
    ax.bar(x + (i - 1) * w, y, w, yerr=[np.array(y) - lo, np.array(hi) - np.array(y)],
           capsize=3, label=L)
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel("rate on baseline-110 quartets")
ax.set_title("Endpoint repair vs full repair vs migration (dev bank)")
ax.legend()
fig.tight_layout()
fig.savefig("paper/figures/fig3_p3_migration.pdf")
fig.savefig("paper/figures/fig3_p3_migration.png", dpi=150)

# J bars: clean / flipmine / relfeat / relflip (dev)
rel = json.load(open("artifacts/next_novelty/relfeat/RELFEAT.json"))
rfx = json.load(open("artifacts/next_novelty/relflip/RELFLIP_CONFIRM.json")) if False else None
J = {}
for s in seeds:
    J[s] = {"clean": p3[s]["metrics"]["J_clean"][0],
            "flipmine": p3[s]["metrics"]["J_repair"][0],
            "relfeat": rel["s%s_relfeat" % s[1:]]["J"] if "s%s_relfeat" % s[1:] in rel else None}
print("J:", J)
print("FIGURES DONE")
