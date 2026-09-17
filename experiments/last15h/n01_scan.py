#!/usr/bin/env python3
"""N01 round1: locate true vs model turns on 256 linear paths (clean model)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import scan_linear, locate_oracle_turns, locate_model_turns, match_turns
from common import load_model, preprocess
from r02_search import candidates_for_scene

ART = ROOT / "artifacts" / "discovery_campaign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_paths", type=int, default=256)
    p.add_argument("--model", type=Path,
                   default=ART / "r04b_s11" / "clean")
    a = p.parse_args()
    rng = np.random.default_rng(11)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    model, stats = load_model(a.model)
    model.eval()

    def predict(xs):
        with torch.no_grad():
            lg = model(preprocess(xs, stats)).numpy()
        return lg, (lg > 0).astype(int)

    # stratified edits: 6 per family group
    fams = {}
    paths, single, multi, noturn = 0, 0, 0, 0
    matches = []
    order = rng.permutation(len(X))[:a.n_paths]
    for pi in order:
        x = X[pi].astype(float)
        cands = candidates_for_scene(x, rng, n_random=16)
        groups = {}
        for e, f in cands:
            groups.setdefault(f.split("_")[0], []).append((e, f))
        for g, lst in groups.items():
            kk = rng.choice(len(lst), size=min(3, len(lst)), replace=False)
            for k in kk:
                e, f = lst[k]
                e = np.asarray(e, dtype=float)
                sc = scan_linear(x, e, predict, n_scan=17)
                paths += 1
                ot = locate_oracle_turns(x, e, sc)
                if len(ot) == 0:
                    noturn += 1
                    continue
                if len(ot) > 1:
                    multi += 1
                    continue
                single += 1
                mt = locate_model_turns(x, e, sc, predict)
                m = match_turns(ot, mt)[0]
                m.update({"parent_id": int(pi), "family": f,
                          "cost": float(np.linalg.norm(e))})
                matches.append(m)
    st = {}
    for m in matches:
        st[m["status"]] = st.get(m["status"], 0) + 1
    errs = [abs(m["error"]) for m in matches if m["error"] is not None]
    res = {"n_paths": paths, "single_turn": single, "multi_turn": multi,
           "no_turn": noturn, "status_counts": st,
           "mean_abs_err": float(np.mean(errs)) if errs else None,
           "median_abs_err": float(np.median(errs)) if errs else None,
           "matches": matches}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print(f"SAW: {paths} paths, {single} single-turn; status={st}; "
          f"mean|terr|={res['mean_abs_err']}")
    print("NEXT: if systematic early/late bias -> two-sided bracket training; "
          "if mostly missing -> check endpoint-correct subset first")
    print("CLAIM: static accuracy does not guarantee turn location; "
          "quantifies how far off the decision boundary sits")


if __name__ == "__main__":
    main()
