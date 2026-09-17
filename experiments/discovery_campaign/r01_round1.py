#!/usr/bin/env python3
"""R01 round 1: same-marginal, same-energy, centroid-balanced dependence shift.

Bank: n=4 active nodes (A,B,C,D endpoints), antithetic +/- pairs.
All laws share ONE kept-mode bank (both arms oracle-valid), fixed once.
Q_ref: uniform. Q_coherent: team-block template (fixed, response-blind).
Q_source_*: chosen on SOURCE scenes only, applied unchanged on TARGET scenes.
Q_whitebox: per-target worst kept mode (reference only, not a cert bound).

Two tables: label-preserving arms (robustness) and label-flipped arms
(oracle relabeled; model-follow rate). Clean accuracy also reported.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

PACK = Path(__file__).resolve().parents[2] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.perturbations import (balanced_sign_patterns, antithetic_fields,
                                weighted_sign_law,
                                normalize_weights, worst_case_weights,
                                greedy_mode_cover)
from core.relations import segment_relation
from common import load_model, preprocess


def oracle_batch(pos: np.ndarray, min_margin: float):
    """pos [N,4,2] -> labels [N] (-1 invalid), margins [N]."""
    n = len(pos)
    labels = np.full(n, -1)
    margins = np.zeros(n)
    for i in range(n):
        r = segment_relation(pos[i])
        margins[i] = r["margin"]
        if r["margin"] >= min_margin and not r["ambiguous"]:
            labels[i] = r["label"]
    return labels, margins


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    p.add_argument("--min-margin", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bank", type=str, default="4")
    p.add_argument("--src-slice", type=str, default="")
    p.add_argument("--tgt-slice", type=str, default="")
    a = p.parse_args()
    if a.output.exists():
        p.error("--output must be a new directory")
    a.output.mkdir(parents=True, exist_ok=False)

    src = np.load(a.source / "scenes.npz"); tgt = np.load(a.target / "scenes.npz")
    def apply_slice(d, s):
        if not s:
            return d["positions"].astype(float), d["labels"].astype(int)
        lo, hi = [int(x) for x in s.split(":")]
        return d["positions"][lo:hi].astype(float), d["labels"][lo:hi].astype(int)
    Xs, ys = apply_slice(src, a.src_slice)
    Xt, yt = apply_slice(tgt, a.tgt_slice)
    n_src, n_tgt = len(Xs), len(Xt)

    if a.bank == "4":
        patterns = balanced_sign_patterns((4,), max_patterns=64, seed=a.seed)
        coherence = np.abs(patterns[:, 0] + patterns[:, 1]) + np.abs(patterns[:, 2] + patterns[:, 3])

        def scatter(fields):
            return fields
    elif a.bank == "8":
        patterns = balanced_sign_patterns((8,), max_patterns=64, seed=a.seed)
        coherence = (np.abs(patterns[:, 0] + patterns[:, 2]) + np.abs(patterns[:, 1] + patterns[:, 3])
                     + np.abs(patterns[:, 4] + patterns[:, 6]) + np.abs(patterns[:, 5] + patterns[:, 7]))

        def scatter(fields):
            return fields.reshape(fields.shape[0], 2, 4, 2)
    else:
        p.error("--bank must be '4' or '8'")
    n_modes = len(patterns)
    coh_w = (coherence == coherence.max()).astype(float)
    coh_w /= coh_w.sum()
    directions = {"vx": np.array([1., 0.]), "vy": np.array([0., 1.]),
                  "vdiag": np.array([1., 1.]) / np.sqrt(2)}
    if a.bank == "8":
        directions = {"coord": np.array([1.0])}  # scalar per-coordinate signs
    loaded = [(str(m), *load_model(m)) for m in a.models]

    # Clean accuracy on target (all models, same samples).
    clean = {}
    for name, model, stats in loaded:
        with torch.no_grad():
            prob = torch.sigmoid(model(preprocess(Xt, stats))).numpy()
        clean[name] = float((((prob > .5).astype(int)) == yt).mean())

    law_defs = {"Q_ref": "uniform over kept modes; no model/source access",
                "Q_coherent": "team-block template (fixed, response-blind); no model access",
                "Q_source_greedy": "greedy failure cover (k=2) on SOURCE scenes; black-box predictions",
                "Q_source_dro": "entropic DRO (reg=1) on SOURCE pair losses",
                "Q_whitebox": "per-TARGET-scene worst kept mode; reference only"}
    (a.output / "law_definition.json").write_text(json.dumps(
        {"laws": law_defs, "bank": {"bank": a.bank, "n_modes": n_modes,
          "patterns": patterns.tolist(), "coherence": coherence.tolist(),
          "coherent_weights": coh_w.tolist()},
         "directions": {k: v.tolist() for k, v in directions.items()},
         "epsilons": a.epsilons, "min_margin": a.min_margin,
         "source": str(a.source), "target": str(a.target),
         "n_source": n_src, "n_target": n_tgt}, indent=2) + "\n")

    rows, keep_info, marginal_evidence = [], {}, []
    for dname, v in directions.items():
        for eps in a.epsilons:
            fields = scatter(antithetic_fields(patterns, eps, v))  # [modes,2,4,2]
            # Oracle validity for both arms, all scenes (source+target share the bank rule).
            keep_mask, cov = None, {}
            for tag, X, y in (("source", Xs, ys), ("target", Xt, yt)):
                pert = X[None, None, :, :, :] + fields[:, :, None, :, :]  # [M,2,N,4,2]
                M, _, N = pert.shape[:3]
                lab = np.full((M, 2, N), -1)
                for m_i in range(M):
                    for arm in range(2):
                        lab[m_i, arm], _ = oracle_batch(pert[m_i, arm], a.min_margin)
                valid = lab >= 0
                # Bank rule (fixed once, same for all laws): a mode is kept iff EACH arm
                # is oracle-valid on >=10 pool scenes. Per-scene invalid arms are later
                # excluded from denominators (recorded, not silently dropped).
                arm_counts = valid.sum(axis=2)  # [modes,2]
                keep = arm_counts.min(axis=1) >= 10
                cov[tag] = {"valid_arm_fraction": float(valid.mean()),
                            "orig_label_kept_fraction": float((lab == y[None, None, :]).mean())}
                keep_mask = keep if keep_mask is None else (keep_mask & keep)
            kept_idx = np.where(keep_mask)[0]
            keep_info[f"{dname}/eps={eps}"] = {
                "n_modes_kept": int(keep_mask.sum()), "kept_idx": kept_idx.tolist(),
                "coverage": cov}
            if keep_mask.sum() == 0:
                continue
            fields_k = fields[kept_idx]
            Mk = len(kept_idx)
            # Per-scene, per-mode, per-arm oracle labels + model failures on TARGET.
            pert_t = Xt[None, None, :, :, :] + fields_k[:, :, None, :, :]
            lab_t = np.full((Mk, 2, n_tgt), -1)
            for m_i in range(Mk):
                for arm in range(2):
                    lab_t[m_i, arm], _ = oracle_batch(pert_t[m_i, arm], a.min_margin)
            for name, model, stats in loaded:
                with torch.no_grad():
                    base_prob = torch.sigmoid(model(preprocess(Xt, stats))).numpy()
                base_pred = (base_prob > .5).astype(int)
                # pair loss on source for Q_source weights (source scenes, same kept bank).
                pert_s = Xs[None, None, :, :, :] + fields_k[:, :, None, :, :]
                lab_s = np.full((Mk, 2, n_src), -1)
                for m_i in range(Mk):
                    for arm in range(2):
                        lab_s[m_i, arm], _ = oracle_batch(pert_s[m_i, arm], a.min_margin)
                with torch.no_grad():
                    src_fail = np.zeros((Mk, 2, n_src))
                    for m_i in range(Mk):
                        for arm in range(2):
                            pr = (torch.sigmoid(model(preprocess(pert_s[m_i, arm], stats)))
                                  .numpy() > .5).astype(int)
                            ok = lab_s[m_i, arm] >= 0
                            src_fail[m_i, arm][ok] = (pr[ok] != lab_s[m_i, arm][ok]).astype(float)
                pair_loss_src = src_fail.mean(axis=(1, 2))
                # success matrix for greedy cover: mode fails scene if either arm fails (valid arms).
                succ = (src_fail.sum(axis=1) > 0)
                greedy_idx, _ = greedy_mode_cover(succ, k=min(2, Mk))
                w_greedy = np.zeros(Mk); w_greedy[greedy_idx] = 1.0 / max(1, len(greedy_idx))
                w_dro = worst_case_weights(pair_loss_src, regularization=1.0)
                w_ref = np.full(Mk, 1.0 / Mk)
                w_coh = coh_w[kept_idx]; w_coh = w_coh / w_coh.sum()
                # Target pair losses (valid arms only; invalid arms excluded from denominator).
                with torch.no_grad():
                    tgt_fail = np.zeros((Mk, 2, n_tgt)); tgt_ok = np.zeros((Mk, 2, n_tgt), bool)
                    tgt_flip = np.zeros((Mk, 2, n_tgt), bool)  # oracle label != original
                    for m_i in range(Mk):
                        for arm in range(2):
                            pr = (torch.sigmoid(model(preprocess(pert_t[m_i, arm], stats)))
                                  .numpy() > .5).astype(int)
                            ok = lab_t[m_i, arm] >= 0
                            tgt_ok[m_i, arm] = ok
                            tgt_fail[m_i, arm][ok] = (pr[ok] != lab_t[m_i, arm][ok]).astype(float)
                            tgt_flip[m_i, arm][ok] = (lab_t[m_i, arm][ok] != yt[ok])
                pres = ~tgt_flip & tgt_ok   # label-preserving, valid
                flip = tgt_flip & tgt_ok    # label-flipped, valid
                laws = {"Q_ref": w_ref, "Q_coherent": w_coh,
                        "Q_source_greedy": w_greedy, "Q_source_dro": w_dro}
                for lname, w in laws.items():
                    w = normalize_weights(w, Mk)
                    ev = weighted_sign_law(patterns[kept_idx], w)
                    marginal_evidence.append(float(np.abs(ev["plus_probability"] - .5).max()))
                    for tab, mask in (("preserving", pres), ("flipped", flip)):
                        num = (w[:, None, None] * tgt_fail * mask).sum()
                        den = (w[:, None, None] * mask).sum()
                        rows.append({"model": Path(name).name, "direction": dname,
                                     "epsilon": eps, "law": lname, "table": tab,
                                     "risk": float(num / den) if den > 0 else float("nan"),
                                     "n_valid_arms": int(mask.sum()),
                                     "denominator_weighted": float(den)})
                # Whitebox per-scene worst kept mode (valid arms, all arms table).
                pl = np.where(tgt_ok, tgt_fail, np.nan)
                with np.errstate(all="ignore"):
                    worst = np.nanmax(pl.reshape(Mk, -1), axis=0)
                okmask = ~np.isnan(worst)
                rows.append({"model": Path(name).name, "direction": dname, "epsilon": eps,
                             "law": "Q_whitebox", "table": "all_valid_arms",
                             "risk": float(np.nanmean(worst)),
                             "n_valid_arms": int(okmask.sum() * 2),
                             "denominator_weighted": float(okmask.sum())})
    import csv
    with open(a.output / "risk_by_scene.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(a.output / "same_marginal_example.npz",
                        patterns=patterns, plus_deviation_max=np.max(marginal_evidence))
    (a.output / "R01_result.json").write_text(json.dumps(
        {"clean_target_acc": clean,
         "max_marginal_plus_deviation": float(np.max(marginal_evidence)),
         "keep_info": keep_info,
         "n_rows": len(rows),
         "notes": ["All laws share one kept-mode bank fixed before law comparison.",
                   "Invalid (ambiguous) arms excluded per-scene; denominators recorded.",
                   "Q_source_* chosen on source scenes only; reported on target scenes."]},
        indent=2) + "\n")
    print(json.dumps({"clean": clean, "max_plus_dev": float(np.max(marginal_evidence)),
                      "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
