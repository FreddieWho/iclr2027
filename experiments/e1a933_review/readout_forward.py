#!/usr/bin/env python3
"""O04 forward intervention: a *preregistered* prediction of which segment
(S / J / J*) the change-supervision intervention moves, executed on a new task.

New task  : U10 T1 (point-in-triangle, 4 points) and T2 (segment-disk, 3 points).
New models: the frozen U10 T1/T2 backbones, 4 encoder classes x 3 seeds each
            (raw_clean, raw_flip, six_clean, six_flip).  "New" is relative to
            the O04 readout reanalysis, which used the source segment-crossing
            bank and its 4 arms; the T1/T2 backbones and banks already exist in
            the f095 campaign and were NOT trained here.
Intervention I_flip: keep the frozen trunk and the frozen feature
            standardization; change only the head's supervision (clean states
            only vs clean + single-edit states at 50:50 weight).  Same head
            family, same initialisation, same L2/width candidates, same
            dev-selection budget in both regimes.

Stages
  --stage prereg : write PREREGISTRATION.json + PREREGISTRATION.seal.json and
                   stop.  Refuses if the preregistration already exists.
  --stage run    : verify the preregistration (payload hash + script hash),
                   then execute and write results.  Refuses if results.json
                   exists.
  --stage verify : re-check preregistration integrity and the recorded
                   prereg-before-results ordering; compute nothing.

Old artifacts are read-only; every output goes to a fresh directory.
"""
import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "6"
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from coord_mlp import CoordMLP  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402

SEEDS = (11, 23, 47)
CLASSES = (("raw", "clean"), ("raw", "flip"), ("six", "clean"), ("six", "flip"))
FAMILIES = ("logistic", "mlp")
REGIMES = ("static", "flip")
L2_CANDIDATES = (1e-4, 1e-3, 1e-2)
MLP_WIDTHS = (16, 64)
MLP_EPOCHS = 300
TASKS = {"T1": dict(npts=4, in_raw=8, in_six=6), "T2": dict(npts=3, in_raw=6, in_six=3)}
BOOT_SEED = 904
BOOT_N = 2000

U10 = ROOT / "artifacts" / "f095_campaign" / "U10"
BANKDIR = ROOT / "artifacts" / "p123_upgrade" / "bank"
SRC_BANK = BANKDIR / "bank_dev512.npz"
SRC_TRAIN = ROOT / "artifacts" / "discovery_campaign" / "scenes" / "train_101" / "scenes.npz"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8"))


def backbone_sha(model) -> str:
    """Hash of the frozen trunk (identical convention to the O04 reanalysis)."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        if k.startswith("net.") or k.startswith("feat_head."):
            h.update(sd[k].cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# preregistered prediction (frozen text; payload hash is recorded in the seal)
# --------------------------------------------------------------------------

PREDICTION_TEXT = """\
干预 I_flip（本预测的对象）：冻结主干 h 与冻结的特征标准化/whitening 不变，只改 head 的训练监督。
regime=static 只用 clean 单状态（head-fit parent 的 512 场景子集）；regime=flip 在同一 clean 子集
之外加入该任务 train 的单编辑（single-edit）状态，两组各占 50% 权重。head 家族、初始化、L2/宽度
候选、dev 选择预算在两种 regime 之间完全一致。所以 I_flip 是"变化监督"干预：它不改变主干，
不改变评估 quartet，也不读任何评估标签。

对每个 (task, encoder class, seed, head family) 定义 ΔX = X(regime=flip) − X(regime=static)，
X 为评估 quartet 上的 parent-equal 配对均值（同一 parent 内先平均，再对 parent 平均）。
S = S(f_head∘h)：最终标量 score 在单个 quartet 上把三个状态分开的比例（局部可分性）。
J = 固定阈值 0 下的联合正确率。J* = 用评估真值选出的最优公共阈值下的联合正确率，
是诊断上界，不是可部署准确率。

预测（primary pipeline：冻结标准化、convex L2-logistic head、3 seed 均值）：

P1【主要影响哪一段 = J】全部 8 个 (task × encoder class) 单元上，|ΔJ| 严格大于 |ΔS| 与 |ΔJ*|。
   即 I_flip 是"J 段干预"：主要改变固定阈值下的联合正确率，不改变 score 的局部可分性 S，
   也不改变 oracle 阈值上界 J*。
P2【oracle 上界不是获益来源】全部 8 个单元上 |ΔJ*| < |ΔJ|。
P3【S 不该动】4 个 six 单元（T1/T2 × six_clean/six_flip）上，ΔS 的 3 seed 均值 ≤ 0：
   该干预不提高关系型表示的局部可分性。
P4【次要、方向性风险】4 个 six 单元上，ΔJ 的 3 seed 均值 > 0：联合段变好。
P5【次要：分段归属随 encoder 类别改变（MLP head）】在 MLP head 家族上，4 个 six 单元归属 J
   （|ΔJ| 最大），4 个 raw 单元归属 S（|ΔS| 最大），8 个单元全部成立。

判据（可判定，无自由数值阈值）：每个 Pi 在"全部单元成立"时记为 SUPPORTED，否则 NOT_SUPPORTED；
无论结果如何都逐单元、逐 seed 报告数值与 parent bootstrap 区间，不删任何单元。
hit/miss：primary verdict = P1 ∧ P2。P3/P4/P5 单独报告，不用于回填 primary。

证伪含义：若任一单元 |ΔS| 最大，则该单元上 I_flip 是 S 段干预，O04 的"S/J/J* 分解能前瞻指出
干预作用在哪一段"在该单元不成立。若 |ΔJ*| 最大，则获益主要是 oracle 阈值位置移动，同样证伪。
"""

PREDICTION = {
    "intervention": "I_flip_head_supervision",
    "metric_definitions": {
        "S": "S(f_head∘h): fraction of eval quartets whose three single-state scores are locally separable",
        "J": "joint correctness at the fixed threshold 0",
        "J_star": "joint correctness at the evaluation-truth oracle common threshold (diagnostic upper bound, NOT deployable accuracy)",
        "delta": "parent-equal paired mean of (regime=flip) minus (regime=static)",
    },
    "primary_pipeline": {
        "feature_space": "standardized",
        "head_family": "logistic",
        "aggregation": "3-seed mean of parent-equal paired deltas",
    },
    "secondary_pipelines": ["whitened feature space (logistic only)", "mlp head family"],
    "predictions": [
        {"id": "P1", "claim": "attribution of I_flip is the J segment in all 8 (task x encoder-class) cells",
         "rule": "|dJ| > |dS| and |dJ| > |dJ_star| in every cell", "cells": 8,
         "segment_expected": "J"},
        {"id": "P2", "claim": "the oracle bound is not where the gain comes from",
         "rule": "|dJ_star| < |dJ| in every cell", "cells": 8, "segment_expected": "not J_star"},
        {"id": "P3", "claim": "I_flip does not increase separability on relational (six) encoders",
         "rule": "3-seed mean dS <= 0 in every six cell", "cells": 4, "segment_expected": "S unchanged"},
        {"id": "P4", "claim": "the joint segment improves on relational (six) encoders",
         "rule": "3-seed mean dJ > 0 in every six cell", "cells": 4, "segment_expected": "J up"},
        {"id": "P5", "claim": "under an MLP head the attributed segment follows the encoder class",
         "rule": "|dJ| max in every six cell and |dS| max in every raw cell", "cells": 8,
         "segment_expected": "six->J, raw->S"},
    ],
    "decision_rule": "a prediction is SUPPORTED only if it holds in all of its cells; counts are reported either way",
    "primary_verdict": "P1 AND P2",
    "prior_evidence_source_bank": {
        "source": "artifacts/e1a933_review/readout_refit_20260924/summary.json (head regime flip minus static, parent-equal, 3-seed mean)",
        "note": ("the source segment-crossing bank showed this pattern; the new-task test is an "
                 "extrapolation of it, not a restatement"),
        "logistic": {"raw_clean": {"dS": -0.0032, "dJ": -0.0375, "dJ_star": -0.0147},
                     "raw_flipmine": {"dS": 0.0037, "dJ": 0.0204, "dJ_star": -0.0060},
                     "relfeat": {"dS": -0.0046, "dJ": 0.0797, "dJ_star": 0.0126},
                     "relflip": {"dS": 0.0, "dJ": 0.0381, "dJ_star": -0.0013}},
        "mlp": {"raw_clean": {"dS": -0.0566, "dJ": -0.0084, "dJ_star": -0.0016},
                "raw_flipmine": {"dS": -0.0338, "dJ": 0.0036, "dJ_star": -0.0127},
                "relfeat": {"dS": -0.0105, "dJ": 0.0724, "dJ_star": -0.0124},
                "relflip": {"dS": 0.0016, "dJ": 0.0678, "dJ_star": 0.0078}},
        "source_pattern": ("logistic: argmax segment is J in 4/4 classes, |dJ*| < |dJ| in 4/4; "
                           "mlp: argmax is S for the raw classes and J for the relational classes"),
    },
}


# --------------------------------------------------------------------------
# data / feature plumbing
# --------------------------------------------------------------------------

def pairwise_dists(X, npts):
    X = np.asarray(X, float).reshape(-1, npts, 2)
    i, j = np.triu_indices(npts, k=1)
    return np.linalg.norm(X[:, i] - X[:, j], axis=-1)


def featurize(arch, X, mu, sd, npts):
    F = pairwise_dists(X, npts) if arch == "six" else np.asarray(X, np.float32).reshape(len(X), -1)
    return torch.from_numpy(((np.asarray(F, np.float32) - mu) / sd).astype(np.float32))


@torch.no_grad()
def frozen_features(model, X):
    model.eval()
    out = []
    for s in range(0, len(X), 2048):
        _, z = model(X[s:s + 2048], return_feat=True)
        out.append(z)
    return torch.cat(out).numpy().astype(np.float64)


def load_task(task):
    cfg = TASKS[task]
    d = np.load(U10 / f"{task}_train" / "scenes.npz")
    X = d["positions"].astype(np.float64)
    y = d["labels"].astype(np.float64)
    fz = np.load(U10 / f"{task}_flips.npz", allow_pickle=True)
    fmeta = fz["meta"]
    F = np.stack([fz[f"edit_{k}"] for k in range(len(fmeta))]).astype(np.float64)
    fii = np.array([int(r[0]) for r in fmeta])
    fy = np.array([int(r[1]) for r in fmeta], dtype=np.float64)
    xf = X[fii] + F
    b = np.load(BANKDIR / f"bank_u10{task}.npz", allow_pickle=True)
    qmeta = json.loads(str(b["Qmeta"]))
    groups = {}
    for i, row in enumerate(qmeta):
        groups.setdefault(row["qid"], {})[row["ptype"]] = (i, row)
    xe, labels, parents, qids = [], [], [], []
    for qid, ps in sorted(groups.items()):
        if set(ps) != {"A", "B"}:
            raise ValueError("incomplete quartet")
        ia, ma = ps["A"]
        ib, mb = ps["B"]
        np.testing.assert_allclose(b["Qx"][ia] + b["Qe"][ia], b["Qx"][ib] + b["Qe"][ib], atol=1e-6)
        xe.extend([b["Qx"][ia], b["Qx"][ib], b["Qx"][ia] + b["Qe"][ia]])
        labels.append([ma["yA"], ma["yB"], ma["yAB"]])
        parents.append(ma["parent"])
        qids.append(qid)
    rng = np.random.default_rng(4404)
    perm = rng.permutation(len(X))
    fit = np.zeros(len(X), bool)
    fit[perm[:int(0.8 * len(X))]] = True
    return dict(cfg=cfg, X=X, y=y, xf=xf, fii=fii, fy=fy, fit=fit,
                xe=np.array(xe), labels=np.array(labels, float),
                parents=np.array(parents), qids=np.array(qids),
                src_files=[U10 / f"{task}_train" / "scenes.npz",
                           U10 / f"{task}_flips.npz",
                           BANKDIR / f"bank_u10{task}.npz"])


def eval_flags(scores, labels, t=0.0):
    lo = np.where(labels == 0, scores, -np.inf).max(1)
    hi = np.where(labels == 1, scores, np.inf).min(1)
    cert = threshold_certificate(scores, labels, t)
    jt = cert.diagnostic_optimal_threshold
    return dict(S=(lo < hi).astype(float),
                J=((lo <= t) & (t < hi)).astype(float),
                J_star=((lo <= jt) & (jt < hi)).astype(float) if jt is not None
                else np.zeros(len(lo))), cert


def parent_equal(delta, parents):
    return float(np.mean([delta[parents == p].mean() for p in np.unique(parents)]))


def paired(a, b, parents):
    delta = np.asarray(b) - np.asarray(a)
    vals = np.array([delta[parents == p].mean() for p in np.unique(parents)])
    rng = np.random.default_rng(BOOT_SEED)
    boots = vals[rng.integers(len(vals), size=(BOOT_N, len(vals)))].mean(1)
    return dict(quartet_mean=float(delta.mean()), parent_equal_mean=float(vals.mean()),
                parent_cluster_ci95=np.quantile(boots, [.025, .975]).tolist(),
                n_parent=len(vals))


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------

def objective(theta, z, y, w, lam):
    score = z @ theta[:-1] + theta[-1]
    value = np.dot(w, np.logaddexp(0, score) - y * score) + lam * np.dot(theta[:-1], theta[:-1]) / 2
    err = w * (expit(score) - y)
    return value, np.r_[z.T @ err + lam * theta[:-1], err.sum()]


def convex(z, y, w, lam):
    trace = []

    def fun(t):
        v, g = objective(t, z, y, w, lam)
        trace.append(float(v))
        return v, g

    r = minimize(fun, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B",
                 options={"maxiter": 1000, "gtol": 1e-7, "ftol": 1e-12, "maxls": 40})
    return r.x, dict(success=bool(r.success), message=str(r.message), nit=int(r.nit),
                     nfev=int(r.nfev), grad_inf=float(np.max(np.abs(r.jac))),
                     objective=float(r.fun), l2=lam, trace=trace)


def bce(score, y, w):
    return float(np.dot(w, np.logaddexp(0, score) - y * score))


def mlp_fit(z, y, w, zv, yv, wv, width, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1], width), torch.nn.ReLU(),
                              torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    zz, yy, ww = [torch.as_tensor(a, dtype=torch.float32) for a in (z, y, w)]
    zvv = torch.as_tensor(zv, dtype=torch.float32)
    best, state, best_ep, trace = float("inf"), None, 0, []
    for epoch in range(MLP_EPOCHS):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            net(zz).flatten(), yy, reduction="none") * ww).sum()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                v = bce(net(zvv).flatten().numpy(), yv, wv)
            trace.append([epoch + 1, float(loss.detach()), v])
            if v < best:
                best, best_ep = v, epoch + 1
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    return net, dict(dev_bce=best, best_epoch=best_ep, width=width, trace=trace,
                     convergence="nonconvex; no global convergence claim")


def dev_threshold(dv, yv, wv):
    ts = np.unique(np.r_[np.nextafter(dv.min(), -np.inf), dv])
    acc = np.array([np.dot(wv, (dv > t) == yv) for t in ts])
    return float(ts[int(np.argmax(acc))])


def whitening_matrix(zfit):
    zc = zfit - zfit.mean(0)
    cov = zc.T @ zc / len(zc)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-3 * max(evals.mean(), 1e-12))
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def prereg_payload(output: Path):
    payload = {
        "lane": "O04",
        "package": "docs/e1a933_review",
        "kind": "prospective intervention prediction (preregistration)",
        "written_utc": utcnow(),
        "written_epoch": time.time(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha(Path(__file__).resolve()),
        "command": sys.argv,
        "new_task": {
            "T1": "U10 point-in-triangle (4 points), eval bank 2724 E-quartets / 322 parents",
            "T2": "U10 segment-disk (3 points), eval bank 3443 E-quartets / 369 parents",
            "new_means": ("new for the O04 readout decomposition; the U10 backbones and banks "
                          "were produced by the earlier f095 campaign and were NOT trained here"),
        },
        "new_models": {
            "classes": ["raw_clean", "raw_flip", "six_clean", "six_flip"],
            "seeds": list(SEEDS),
            "checkpoints": [str((U10 / f"{t}_{a}_s{s}" / r / "model.pt").relative_to(ROOT))
                            for t in TASKS for a, r in CLASSES for s in SEEDS],
        },
        "intervention": PREDICTION["intervention"],
        "metric_definitions": PREDICTION["metric_definitions"],
        "primary_pipeline": PREDICTION["primary_pipeline"],
        "secondary_pipelines": PREDICTION["secondary_pipelines"],
        "predictions": PREDICTION["predictions"],
        "prior_evidence_source_bank": PREDICTION["prior_evidence_source_bank"],
        "decision_rule": PREDICTION["decision_rule"],
        "primary_verdict": PREDICTION["primary_verdict"],
        "falsification": ("any cell whose largest |delta| is S falsifies the J-attribution there; "
                          "a largest |delta| of J* means the gain is an oracle-threshold shift"),
        "prediction_text": PREDICTION_TEXT,
        "payload_sha256": canonical_sha(PREDICTION),
        "inputs_hashed": {str(p.relative_to(ROOT)): sha(p) for p in (
            SRC_TRAIN, SRC_BANK, *[U10 / f"{t}_{sp}" / "scenes.npz" for t in TASKS for sp in ("train", "eval")],
            *[U10 / f"{t}_flips.npz" for t in TASKS],
            *[BANKDIR / f"bank_u10{t}.npz" for t in TASKS])},
        "backbone_hashes": {str((U10 / f"{t}_{a}_s{s}" / r / "model.pt").relative_to(ROOT)):
                            sha(U10 / f"{t}_{a}_s{s}" / r / "model.pt")
                            for t in TASKS for a, r in CLASSES for s in SEEDS},
        "absence_assertions": {str((output / name).relative_to(ROOT)): "absent at prereg time"
                               for name in ("results.json", "solver_logs", "scores")},
        "declared_unread_before_prereg": [
            "artifacts/f095_campaign/U10/U10_T1_EVAL.json",
            "artifacts/f095_campaign/U10/U10_T2_EVAL.json",
            "artifacts/f095_campaign/U10/U10_ORBIT_EVAL.json",
        ],
        "disclosure": ("The prediction was derived from the source-bank O04 readout reanalysis "
                       "(reports/e1a933_review/O04_READOUT_REPORT.md). Only the top-level key list of "
                       "U10_T1_EVAL.json was read before this file was written; no U10 evaluation "
                       "value was inspected."),
        "threads": {"OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6",
                    "MKL_NUM_THREADS": "6", "torch": 6},
    }
    return payload


def stage_prereg(output: Path):
    if (output / "PREREGISTRATION.json").exists():
        raise SystemExit(f"preregistration already exists: {output/'PREREGISTRATION.json'}")
    output.mkdir(parents=True, exist_ok=True)
    payload = prereg_payload(output)
    path = output / "PREREGISTRATION.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    seal = {"prereg_sha256": sha(path), "written_utc": payload["written_utc"],
            "written_epoch": payload["written_epoch"], "payload_sha256": payload["payload_sha256"],
            "script_sha256": payload["script_sha256"],
            "note": "seal of the preregistration file bytes; results must be written after this"}
    (output / "PREREGISTRATION.seal.json").write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PREREG_WRITTEN {path} utc={payload['written_utc']} "
          f"payload_sha256={payload['payload_sha256'][:16]} seal={seal['prereg_sha256'][:16]}", flush=True)


def verify_prereg(output: Path) -> dict:
    path = output / "PREREGISTRATION.json"
    seal_path = output / "PREREGISTRATION.seal.json"
    if not path.exists() or not seal_path.exists():
        raise SystemExit("preregistration missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    checks = {
        "file_sha_matches_seal": sha(path) == seal["prereg_sha256"],
        "payload_sha_matches_seal": payload["payload_sha256"] == seal["payload_sha256"],
        "payload_sha_recomputes": canonical_sha(PREDICTION) == payload["payload_sha256"],
        "script_unchanged": sha(Path(__file__).resolve()) == payload["script_sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"preregistration integrity failure: {checks}")
    return payload


def stage_verify(output: Path):
    payload = verify_prereg(output)
    res = output / "results.json"
    report = {"prereg_written_utc": payload["written_utc"], "integrity": "ok"}
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        report["results_written_utc"] = r["run"]["results_written_utc"]
        report["ordering_ok"] = r["run"]["results_written_epoch"] > payload["written_epoch"]
        report["results_json_sha256"] = sha(res)
        report["prereg_mtime"] = path_mtime(output / "PREREGISTRATION.json")
        report["results_mtime"] = path_mtime(res)
    else:
        report["results_written_utc"] = None
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def path_mtime(p: Path) -> float:
    return p.stat().st_mtime


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run(output: Path):
    prereg = verify_prereg(output)
    for name in ("results.json",):
        if (output / name).exists():
            raise SystemExit(f"refusing to overwrite {output/name}")
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"prereg written_utc={prereg['written_utc']} payload_sha256={prereg['payload_sha256']}")
    (output / "scores").mkdir(exist_ok=True)
    (output / "features").mkdir(exist_ok=True)
    (output / "solver_logs").mkdir(exist_ok=True)

    rows, paired_out, preds = {}, {}, {}
    sample_hashes = {}
    for task in TASKS:
        D = load_task(task)
        nq, npar = len(D["labels"]), len(np.unique(D["parents"]))
        sample_hashes[task] = dict(
            eval_states=sha_bytes(np.ascontiguousarray(D["xe"], dtype=np.float32).tobytes()),
            eval_labels=sha_bytes(np.ascontiguousarray(D["labels"]).tobytes()),
            eval_parents=sha_bytes(np.ascontiguousarray(D["parents"]).tobytes()),
            train_scenes=sha_bytes(np.ascontiguousarray(D["X"], dtype=np.float32).tobytes()),
            flip_states=sha_bytes(np.ascontiguousarray(D["xf"], dtype=np.float32).tobytes()),
            n_quartets=nq, n_parents=npar)
        log(f"[{task}] quartets={nq} parents={npar} train_scenes={len(D['X'])} flips={len(D['xf'])} "
            f"fit_parents={int(D['fit'].sum())} dev_parents={int((~D['fit']).sum())}")
        fit, fii = D["fit"], D["fii"]
        for seed in SEEDS:
            for arch, regime in CLASSES:
                tag = f"{task}_{arch}_{regime}_s{seed}"
                mp = U10 / f"{task}_{arch}_s{seed}" / regime / "model.pt"
                ck = torch.load(mp, map_location="cpu", weights_only=False)
                model = CoordMLP(64, 32, in_dim=D["cfg"]["in_raw" if arch == "raw" else "in_six"])
                model.load_state_dict(ck["state"])
                for p in model.parameters():
                    p.requires_grad = False
                bsha = backbone_sha(model)
                mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
                npts = D["cfg"]["npts"]
                zc = frozen_features(model, featurize(arch, D["X"], mu, sd, npts))
                zf = frozen_features(model, featurize(arch, D["xf"], mu, sd, npts))
                ze = frozen_features(model, featurize(arch, D["xe"], mu, sd, npts))
                # frozen normalization fitted on clean head-fit parents only
                mz = zc[fit].mean(0)
                sz = np.maximum(zc[fit].std(0), 1e-6)
                W = whitening_matrix(zc[fit])
                zs = {"standardized": dict(c=(zc - mz) / sz, f=(zf - mz) / sz, e=(ze - mz) / sz),
                      "whitened": dict(c=(zc - mz) @ W, f=(zf - mz) @ W, e=(ze - mz) @ W)}
                np.savez_compressed(output / "features" / f"{tag}.npz",
                                    z_clean=zc.astype(np.float32), z_flip=zf.astype(np.float32),
                                    z_eval=ze.astype(np.float32), mu=mz, sd=sz, whitening=W)
                flags_store, score_store, logs = {}, {}, {}
                for space, Z in zs.items():
                    zc_s, zf_s, ze_s = Z["c"], Z["f"], Z["e"]
                    for hregime in REGIMES:
                        if hregime == "static":
                            z = zc_s[fit]; yy = D["y"][fit]
                            zv = zc_s[~fit]; yv = D["y"][~fit]
                            w = np.ones(len(z)) / len(z)
                            wv = np.ones(len(zv)) / len(zv)
                        else:
                            z = np.r_[zc_s[fit], zf_s[fit[fii]]]
                            yy = np.r_[D["y"][fit], D["fy"][fit[fii]]]
                            zv = np.r_[zc_s[~fit], zf_s[~fit[fii]]]
                            yv = np.r_[D["y"][~fit], D["fy"][~fit[fii]]]
                            w = np.r_[np.ones(fit.sum()) / fit.sum() / 2,
                                      np.ones(fit[fii].sum()) / fit[fii].sum() / 2]
                            wv = np.r_[np.ones((~fit).sum()) / (~fit).sum() / 2,
                                       np.ones((~fit)[fii].sum()) / (~fit)[fii].sum() / 2]
                        for family in (("logistic",) if space == "whitened" else FAMILIES):
                            name = f"{space}|{family}|{hregime}"
                            cands = []
                            if family == "logistic":
                                for lam in L2_CANDIDATES:
                                    theta, lg = convex(z, yy, w, lam)
                                    lg["dev_bce"] = bce(zv @ theta[:-1] + theta[-1], yv, wv)
                                    cands.append((lg["dev_bce"], theta, lg))
                                _, theta, chosen = min(cands, key=lambda v: v[0])
                                sc = (ze_s @ theta[:-1] + theta[-1]).reshape(-1, 3)
                                dv = zv @ theta[:-1] + theta[-1]
                            else:
                                for width in MLP_WIDTHS:
                                    net, lg = mlp_fit(z, yy, w, zv, yv, wv, width, 1000 + SEEDS.index(seed))
                                    cands.append((lg["dev_bce"], net, lg))
                                _, net, chosen = min(cands, key=lambda v: v[0])
                                with torch.no_grad():
                                    sc = net(torch.as_tensor(ze_s, dtype=torch.float32)).flatten().numpy().reshape(-1, 3)
                                    dv = net(torch.as_tensor(zv, dtype=torch.float32)).flatten().numpy()
                            ff, cert = eval_flags(sc, D["labels"])
                            dt = dev_threshold(dv, yv, wv)
                            j_dev = ((sc > dt) == D["labels"]).all(1)
                            flags_store[name] = ff
                            score_store[name] = sc
                            logs[name] = [{k: v for k, v in c[2].items() if k != "trace"} | {"trace_len": len(c[2].get("trace", []))}
                                          for c in cands]
                            rows[f"{tag}|{name}"] = dict(
                                S=cert.S_local_separable, J_star=cert.J_star_global_oracle,
                                J=cert.J_at_threshold, J_head_dev_threshold=float(j_dev.mean()),
                                head_dev_threshold=dt, oracle_threshold=cert.diagnostic_optimal_threshold,
                                ordering_failure=cert.ordering_failure,
                                global_incompatibility=cert.global_incompatibility,
                                operating_point_gap=cert.operating_point_gap,
                                n_quartets=len(D["labels"]), n_parents=int(len(np.unique(D["parents"]))),
                                selected={k: v for k, v in chosen.items() if k != "trace"},
                                backbone_sha256=bsha, checkpoint=str(mp.relative_to(ROOT)),
                                checkpoint_sha256=sha(mp))
                (output / "solver_logs" / f"{tag}.json").write_text(
                    json.dumps(logs, indent=2) + "\n", encoding="utf-8")
                np.savez_compressed(output / "scores" / f"{tag}.npz",
                                    labels=D["labels"], parents=D["parents"], qids=D["qids"],
                                    **{f"score|{k}": v for k, v in score_store.items()},
                                    **{f"flag|{k}|{m}": flags_store[k][m]
                                       for k in flags_store for m in ("S", "J", "J_star")})
                for space, family in (("standardized", "logistic"), ("standardized", "mlp"),
                                      ("whitened", "logistic")):
                    st = flags_store[f"{space}|{family}|static"]
                    fl = flags_store[f"{space}|{family}|flip"]
                    key = f"{tag}|{space}|{family}"
                    paired_out[key] = {m: paired(st[m], fl[m], D["parents"])
                                       for m in ("S", "J", "J_star")}
                log(f"[{task}/{arch}_{regime}_s{seed}] trunk={bsha[:12]} "
                    f"J(static/std/log)={rows[f'{tag}|standardized|logistic|static']['J']:.4f} "
                    f"J(flip/std/log)={rows[f'{tag}|standardized|logistic|flip']['J']:.4f}")

    # ---------------- prediction evaluation ----------------
    def cell_delta(task, arch, regime, space, family):
        out = {}
        for m in ("S", "J", "J_star"):
            per_seed = []
            for seed in SEEDS:
                key = f"{task}_{arch}_{regime}_s{seed}|{space}|{family}"
                per_seed.append(paired_out[key][m]["parent_equal_mean"])
            out[m] = dict(per_seed=per_seed, seed_mean=float(np.mean(per_seed)))
        out["argmax_segment"] = max(("S", "J", "J_star"), key=lambda m: abs(out[m]["seed_mean"]))
        return out

    def evaluate(space, family):
        cells = {}
        for task in TASKS:
            for arch, regime in CLASSES:
                cls = f"{arch}_{regime}"
                cells[f"{task}/{cls}"] = cell_delta(task, arch, regime, space, family)
        six = [k for k in cells if k.endswith("six_clean") or k.endswith("six_flip")]
        raw = [k for k in cells if k.endswith("raw_clean") or k.endswith("raw_flip")]
        res = {"cells": cells, "predictions": {}}
        p1 = [k for k, v in cells.items()
              if abs(v["J"]["seed_mean"]) > abs(v["S"]["seed_mean"])
              and abs(v["J"]["seed_mean"]) > abs(v["J_star"]["seed_mean"])]
        p2 = [k for k, v in cells.items()
              if abs(v["J_star"]["seed_mean"]) < abs(v["J"]["seed_mean"])]
        p3 = [k for k in six if cells[k]["S"]["seed_mean"] <= 0]
        p4 = [k for k in six if cells[k]["J"]["seed_mean"] > 0]
        p5 = [k for k in six if cells[k]["argmax_segment"] == "J"] + \
             [k for k in raw if cells[k]["argmax_segment"] == "S"]
        res["predictions"] = {
            "P1": dict(held=len(p1), total=len(cells), supported=len(p1) == len(cells), cells_held=sorted(p1)),
            "P2": dict(held=len(p2), total=len(cells), supported=len(p2) == len(cells), cells_held=sorted(p2)),
            "P3": dict(held=len(p3), total=len(six), supported=len(p3) == len(six), cells_held=sorted(p3)),
            "P4": dict(held=len(p4), total=len(six), supported=len(p4) == len(six), cells_held=sorted(p4)),
            "P5": dict(held=len(p5), total=len(six) + len(raw), supported=len(p5) == len(six) + len(raw),
                       cells_held=sorted(p5)),
        }
        res["argmax_segments"] = {k: v["argmax_segment"] for k, v in cells.items()}
        return res

    preds["primary"] = evaluate("standardized", "logistic")
    preds["secondary_mlp"] = evaluate("standardized", "mlp")
    preds["secondary_whitened"] = evaluate("whitened", "logistic")
    hit = (preds["primary"]["predictions"]["P1"]["supported"]
           and preds["primary"]["predictions"]["P2"]["supported"])
    preds["hit_or_miss"] = {
        "primary_verdict": "P1 AND P2 (standardized + convex logistic head)",
        "result": "HIT" if hit else "MISS",
        "P1_supported": preds["primary"]["predictions"]["P1"]["supported"],
        "P2_supported": preds["primary"]["predictions"]["P2"]["supported"],
    }

    results = {
        "run": dict(
            lane="O04", kind="forward intervention on a new task with a preregistered prediction",
            results_written_utc=utcnow(), results_written_epoch=time.time(),
            prereg_written_utc=prereg["written_utc"], prereg_written_epoch=prereg["written_epoch"],
            prereg_payload_sha256=prereg["payload_sha256"], prereg_file_sha256=sha(output / "PREREGISTRATION.json"),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            command=sys.argv, script=str(Path(__file__).resolve().relative_to(ROOT)),
            script_sha256=sha(Path(__file__).resolve()),
            status="NEW_TASK_FORWARD_TEST; T1/T2 backbones and banks pre-exist (f095 campaign), not a fresh-bank confirmation",
            threads=prereg["threads"]),
        "input_hashes": prereg["inputs_hashed"],
        "backbone_hashes": prereg["backbone_hashes"],
        "sample_hashes": sample_hashes,
        "recipe": dict(
            feature_normalization="mean/std fitted on clean head-fit parents only, frozen to dev/eval; whitened variant uses the same fit covariance",
            l2_candidates=list(L2_CANDIDATES), mlp_widths=list(MLP_WIDTHS), mlp_epochs=MLP_EPOCHS,
            mlp_selection="head-dev BCE every 10 epochs; same budget in both head regimes",
            head_regimes="static = clean states; flip = clean + single-edit states, 50:50 weight; identical init (zero vector / fixed torch seed)",
            dev_split="deterministic 80/20 parent split of the new task's train scenes (rng 4404); flips follow their parent",
            thresholds="J at fixed 0.0; J_head_dev_threshold from head-dev single-state weighted accuracy; J_star is the evaluation-truth oracle bound"),
        "rows": rows,
        "paired": paired_out,
        "predictions": preds,
        "limitations": [
            "new task is new for the O04 readout decomposition, not a fresh-bank confirmation of the project's main claim",
            "T1/T2 backbones were trained in the f095 campaign and are re-used frozen; this script trains heads only",
            "J_star uses evaluation truth and is a diagnostic upper bound, never deployable accuracy",
            "S is a property of S(f_head∘h): it changes with the head, so it is not an intrinsic encoder property",
            "3 seeds share one task/bank and are not independent tasks",
            "paired bootstrap holds the sample-selected oracle threshold fixed for J_star and is descriptive",
            "typed U10 arms are excluded (their frozen trunk exposes no feature head in the released checkpoints)",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"results_written_utc={results['run']['results_written_utc']} "
        f"primary={preds['hit_or_miss']}")
    for name in ("P1", "P2", "P3", "P4", "P5"):
        p = preds["primary"]["predictions"][name]
        print(f"primary {name}: held {p['held']}/{p['total']} supported={p['supported']}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prereg", "run", "verify"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output if a.output.is_absolute() else ROOT / a.output
    if a.stage == "prereg":
        stage_prereg(out)
    elif a.stage == "verify":
        stage_verify(out)
    else:
        run(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
