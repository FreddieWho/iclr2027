#!/usr/bin/env python3
"""B02: probe-capacity check on CACHED frozen features (no re-extraction).

Loads artifacts/bridge/<run>/feats.npz, fits stronger probes, reports the
same conditional metric. Distinguishes representation failure vs readout
failure. CPU minutes.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def metrics(pred, yte, kte):
    P = {k: pred[kte == k] for k in ("base", "A", "B", "AB")}
    Y = {k: yte[kte == k] for k in ("base", "A", "B", "AB")}
    both = (P["A"] == Y["A"]) & (P["B"] == Y["B"])
    n = int(both.sum())
    cm = float((P["AB"][both] != Y["AB"][both]).mean()) if n else None
    return {"atomic": float(np.mean([P["A"] == Y["A"], P["B"] == Y["B"]])),
            "cond_n": n, "cond_miss": cm}


def main():
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    d = np.load(ROOT / "artifacts" / "bridge" / a.run / "feats.npz")
    Ztr, ytr, Zte, yte, kte = d["Ztr"], d["ytr"], d["Zte"], d["yte"], d["kte"]
    sc = StandardScaler().fit(Ztr)
    S_tr, S_te = sc.transform(Ztr), sc.transform(Zte)
    out = {}
    lr = LogisticRegressionCV(Cs=8, max_iter=5000).fit(S_tr, ytr)
    out["logregCV"] = metrics(lr.predict(S_te), yte, kte)
    for h in ((128,), (256, 128)):
        mlp = MLPClassifier(hidden_layer_sizes=h, max_iter=2000,
                            random_state=0).fit(S_tr, ytr)
        out[f"mlp{h}"] = metrics(mlp.predict(S_te), yte, kte)
        out[f"mlp{h}"]["train_acc"] = float(mlp.score(S_tr, ytr))
    print(json.dumps({"run": str(a.run), **out}, indent=1))


if __name__ == "__main__":
    main()
