#!/usr/bin/env python3
"""N04 round1: atomics correct, combos? (eval only, oracle-built combos)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import atomic_edits, oracle_at
from common import load_model, preprocess
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_parents", type=int, default=256)
    p.add_argument("--model", type=Path, default=ART / "r04b_s11" / "clean")
    a = p.parse_args()
    rng = np.random.default_rng(4)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(a.model)
    model.eval()

    def pred(x):
        with torch.no_grad():
            lg = model(preprocess(x[None].astype(np.float32), stats)).numpy()[0]
        return int(lg > 0), float(lg)

    atom_ok = atom_n = 0
    combo_cond_err = combo_cond_n = 0
    combo_all_err = combo_all_n = 0
    emerg_n = emerg_miss = emerg_cond_n = emerg_cond_miss = 0
    order = rng.permutation(len(X))[:a.n_parents]
    for pi in order:
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        edits = atomic_edits(x, rng)
        # atomic accuracy
        results = []
        for e, fam in edits:
            try:
                y, _, _ = oracle_at(x + e)
            except ValueError:
                continue
            p1, _ = pred(x + e)
            atom_n += 1
            atom_ok += (p1 == y)
            results.append((e, fam, y))
        # combos: both atomics keep label per oracle, combo changes it
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                (ea, fa, ya), (eb, fb, yb) = results[i], results[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, _, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                pc, _ = pred(x + eb + ea)
                combo_all_n += 1
                combo_all_err += (pc != yc)
                if yc != y0:
                    emerg_n += 1
                    emerg_miss += (pc != yc)
                pa, _ = pred(x + ea)
                pb, _ = pred(x + eb)
                if pa == ya and pb == yb:
                    combo_cond_n += 1
                    combo_cond_err += (pc != yc)
                    if yc != y0:
                        emerg_cond_n += 1
                        emerg_cond_miss += (pc != yc)
    res = {"atomic_acc": atom_ok / atom_n if atom_n else None,
           "combo_err_uncond": combo_all_err / combo_all_n if combo_all_n else None,
           "combo_n": combo_all_n,
           "combo_err_given_both_atomics_correct": combo_cond_err / combo_cond_n if combo_cond_n else None,
           "combo_cond_n": combo_cond_n,
           "emergent_n": emerg_n,
           "emergent_miss_rate": emerg_miss / emerg_n if emerg_n else None,
           "emergent_cond_n": emerg_cond_n,
           "emergent_cond_miss_rate": emerg_cond_miss / emerg_cond_n if emerg_cond_n else None}
    # displacement-matched control: single-edit flips with ||e|| ~ emergent combo ||ea+eb||
    emerg_costs = []
    for pi in order:
        x = X[pi].astype(float)
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        edits = atomic_edits(x, rng)
        results = []
        for e, fam in edits:
            try:
                y, _, _ = oracle_at(x + e)
            except ValueError:
                continue
            results.append((e, y))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                (ea, ya), (eb, yb) = results[i], results[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, _, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc != y0:
                    emerg_costs.append(float(np.linalg.norm(ea + eb)))
    # single-edit flips matched to emergent cost distribution
    matched_miss = matched_n = 0
    for pi in order:
        x = X[pi].astype(float)
        for e, _ in candidates_for_scene(x, rng, n_random=32):
            e = np.asarray(e, dtype=float)
            c = float(np.linalg.norm(e))
            if not emerg_costs or min(abs(c - ec) for ec in emerg_costs) > 0.02:
                continue
            try:
                y0, _, _ = oracle_at(x)
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 == y0 or m1 < 0.005:
                continue
            p1, _ = pred(x + e)
            matched_n += 1
            matched_miss += (p1 != y1)
            break
    res["emergent_cost_median"] = float(np.median(emerg_costs)) if emerg_costs else None
    res["matched_single_n"] = matched_n
    res["matched_single_miss"] = matched_miss / matched_n if matched_n else None
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print(f"SAW: atomic_acc={res['atomic_acc']} combo_cond_err={res['combo_err_given_both_atomics_correct']} (n={combo_cond_n}) emerg_n={emerg_n} emerg_miss={res['emergent_miss_rate']} emerg_cond_miss={res['emergent_cond_miss_rate']} (n={emerg_cond_n}) matched_single_miss={res['matched_single_miss']} (n={res['matched_single_n']})")
    print("NEXT: if cond gap exists -> turn-type-cover curriculum trial; "
          "if gap vanishes under displacement control -> drop N04")
    print("CLAIM: knows parts, fails their composition")


if __name__ == "__main__":
    main()
