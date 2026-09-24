#!/usr/bin/env python3
"""U06 base (local CPU): relation-supervision transfer to an image student.

Student: ImageNet-pretrained ResNet18 (the D04-validated net), 64px.
Teacher signal (route option b, preferred): pair-relation targets computed
from train-scene geometry (6 distances + 4 radii, exact by construction) --
train-only privileged supervision; inference is pixel-only.
Arms, same student / same images / same endpoint labels:
  A static_flip : endpoint BCE only (D04-stageB equivalent, rerun in-path)
  B general_aux : + MSE head -> 8-dim raw coords (equal-amount generic geometry)
  C rel_distill : + MSE head -> 10-dim relation features (task-organized)
  Cshuf         : C with permuted relation targets, s803 only (sanity)
Legacy lambda_aux=0.5 (not gradient-scale matching). Seeds 803/805/806.
Eval: stage-A/B protocol (J/atomic/E-CCM + R_full/M vs arm-A B110).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(8)
from d04_vision_train import build_data, prep, IMNET_MEAN, IMNET_STD  # noqa: E402
from n08_visual import render  # noqa: E402

U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
LAM_AUX = 0.5


def rel10(X44):
    X = np.asarray(X44, float).reshape(-1, 4, 2)
    i, j = np.triu_indices(4, k=1)
    pw = np.linalg.norm(X[:, i] - X[:, j], axis=-1)
    cen = X.mean(axis=1, keepdims=True)
    return np.concatenate([pw, np.linalg.norm(X - cen, axis=-1)], axis=-1)


H_ENDPOINTS = np.array([[0, 1, 2, 3], [1, 0, 2, 3],
                        [0, 1, 3, 2], [1, 0, 3, 2]])


def target_orbit(coords, kind="rel10"):
    """One consistent red/blue endpoint permutation per entire structure."""
    x = np.asarray(coords, float).reshape(-1, 4, 2)
    return np.stack([rel10(x[:, p]) if kind == "rel10" else
                     x[:, p].reshape(len(x), 8) for p in H_ENDPOINTS], axis=1)


def normalize_orbit(orbit, stats=None):
    """Train-only orbit-pooled statistics commute with the endpoint action."""
    if stats is None:
        stats = (orbit.mean(axis=(0, 1)), orbit.std(axis=(0, 1)).clip(1e-8))
    return (orbit - stats[0]) / stats[1], stats


def matched_mse(pred, orbit, reduction="mean"):
    losses = ((pred[:, None, :] - orbit) ** 2).mean(dim=-1).min(dim=1).values
    return losses.mean() if reduction == "mean" else losses


def shuffled_indices(n, seed):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed + 300_000))


class DistillNet(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.base = resnet18(weights="IMAGENET1K_V1")
        self.base.fc = nn.Identity()
        self.head = nn.Linear(512, 1)
        # All rel10 arms, including BCE-only, allocate the same auxiliary head.
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed() + 200_000)
            self.aux = nn.Linear(512, aux_dim) if aux_dim else None

    def forward(self, x):
        z = self.base(x)
        out = (self.head(z).squeeze(-1),)
        if self.aux is not None:
            out += (self.aux(z),)
        return out[0] if len(out) == 1 else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=803)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--arms", type=str, default="A,B,C")
    p.add_argument("--res", type=int, default=64)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    R, Y, attempts, C = build_data(a.seed, return_coords=True)
    Xb = prep(R, a.res)
    yb = torch.from_numpy(np.array(Y, dtype=np.float32))
    orbit10, stats10 = normalize_orbit(target_orbit(C))
    orbit08, stats08 = normalize_orbit(target_orbit(C, "coords"))
    R10 = orbit10[:, 0]
    C08 = (C.reshape(len(C), 8) - C.reshape(len(C), 8).mean(0)) / \
        (C.reshape(len(C), 8).std(0) + 1e-8)
    T10 = torch.from_numpy(R10.astype(np.float32))
    T08 = torch.from_numpy(C08.astype(np.float32))
    O10 = torch.from_numpy(orbit10.astype(np.float32))
    O08 = torch.from_numpy(orbit08.astype(np.float32))
    if {"Cshuf", "Dmatched_shuf"} & set(a.arms.split(",")):
        sh = shuffled_indices(len(T10), a.seed)
        T10S = T10[sh]
        O10S = O10[sh]
    else:
        T10S = None
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]
    res = {"seed": a.seed, "res": a.res, "recipe": "Adam3e-4-20ep-b128-IMAGENETnorm-lam0.5",
           "n_train": len(R), "teacher": "oracle rel10 (exact); generic C08 control",
           "rerun": True}
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    for mode in [m.strip() for m in a.arms.split(",")]:
        torch.manual_seed(a.seed)
        aux_dim = {"A": 10, "B": 8, "C": 10, "Cshuf": 10,
                   "Cmatched": 10, "Dmatched_shuf": 10, "Bmatched": 8}[mode]
        net = DistillNet(aux_dim)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        T = {"A": None, "B": T08, "C": T10, "Cshuf": T10S,
             "Cmatched": O10, "Bmatched": O08,
             "Dmatched_shuf": O10S if T10S is not None else None}[mode]
        net.train()
        batch_rng = torch.Generator().manual_seed(a.seed + 100_000)
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xb), generator=batch_rng)
            for i in range(0, len(Xb), 128):
                b = perm[i:i + 128]
                opt.zero_grad()
                if T is None:
                    loss = bce(net(Xb[b])[0], yb[b])
                else:
                    pe, pa = net(Xb[b])
                    loss = bce(pe, yb[b]) + LAM_AUX * (
                        matched_mse(pa, T[b]) if T.ndim == 3 else mse(pa, T[b]))
                loss.backward(); opt.step()
        net.eval()
        (a.out / mode).mkdir(exist_ok=True)
        torch.save({"state": net.state_dict(), "seed": a.seed, "mode": mode},
                   a.out / mode / "model.pt")
        Ps, Ls = [], []
        with torch.no_grad():
            for qi, (q, m) in enumerate(QEV):
                x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
                y0, ya, yb_, yc = (int(v) for v in m)
                sts = [(x, y0), (x + ea, ya), (x + eb, yb_), (x + ea + eb, yc)]
                imgs = [render(np.asarray(s, float),
                               np.random.default_rng(50000 + qi)) for s, _ in sts]
                r = net(prep(imgs, a.res))
                pl = ((r[0] if isinstance(r, tuple) else r).numpy() > 0).astype(int)
                Ps.append(pl); Ls.append([yl for _, yl in sts])
        P, L = np.array(Ps), np.array(Ls)
        okA, okB, okAB = P[:, 1] == L[:, 1], P[:, 2] == L[:, 2], P[:, 3] == L[:, 3]
        E = (L[:, 1] == L[:, 2]) & (L[:, 2] == L[:, 0]) & (L[:, 3] != L[:, 0])
        res[mode] = {
            "static_acc": round(float((P[:, 0] == L[:, 0]).mean()), 4),
            "atomic_acc": round(float((okA & okB).mean()), 4),
            "AB_acc": round(float(okAB.mean()), 4),
            "J": round(float((okA & okB & okAB).mean()), 4),
            "E_n": int(E.sum()),
            "E_J": round(float((okA & okB & okAB)[E].mean()), 4) if E.sum() else None,
            "E_CCM_denom": int(((okA & okB)[E]).sum()),
        }
        res[mode]["_okAB_E"] = [int(v) for v in okAB[E]]
        res[mode]["_okAEB_E"] = [int(v) for v in (okA & okB & okAB)[E]]
        res[mode]["_okAE_E"] = [int(v) for v in (okA & okB)[E]]
        print(f"SAW {mode}: J={res[mode]['J']} E_J={res[mode]['E_J']}", flush=True)
    if "A" in res and isinstance(res["A"], dict):
        aAE = np.array(res["A"]["_okAE_E"])
        aAB = np.array(res["A"]["_okAB_E"])
        H = (aAE == 1) & (aAB == 0)
        res["B110_n"] = int(H.sum())
        for mode in [m.strip() for m in a.arms.split(",") if m.strip() != "A"]:
            eAB = np.array(res[mode]["_okAB_E"])
            eAE = np.array(res[mode]["_okAE_E"])
            eAEB = np.array(res[mode]["_okAEB_E"])
            res[mode]["R_endpoint"] = round(float(eAB[H].mean()), 4) if H.sum() else None
            res[mode]["R_full"] = round(float(eAEB[H].mean()), 4) if H.sum() else None
            res[mode]["M"] = round(float(((eAB == 1) & (eAE == 0))[H].mean()), 4) \
                if H.sum() else None
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE u06_relation_distill")


if __name__ == "__main__":
    main()
