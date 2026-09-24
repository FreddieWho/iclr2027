#!/usr/bin/env python3
"""O01 continuation: scratch vs warm-start recipe factorial on the O01 new pool.

Frozen design (written before execution):

* clean start = the O01 static-only (frac=0) candidate selected by static/single
  development BCE for each (size, arch, seed). Same bank, same architecture and
  same feature normalization as the O01 scratch campaign, so scratch and
  warm-start differ only in the training path, not in the data or the encoder.
* warm arms = copy of that clean state, fresh optimizer, 300 full-batch steps:
    plain                  base + retained flips
    protection             + soft teacher (clean-start logits) on old-correct base points
    hard_replay            + hard labels on the same points
    preserve_flip_balanced real same-label perturbations replace half of the flips
  Every warm arm forwards base + augmentation + replay each step, so the extra
  forward count is matched across methods at a given dose.
* dose = retained fraction of the mined flip pool (0/25/100), raw and six-distance
  inputs, training size N and 4N as two levels of ONE size factor.
* scratch is the training-path control (O01 plain campaign plus robust-config
  scratch runs); scratch arms have no pre-flip phase, so they are not given the
  protection/replay/balanced variants.
* all selection uses static/single development BCE only. The evaluation pool is
  never used to choose a learning rate, step count, seed or method.
* torch threads = 2 on purpose: bit-compatible with the O01 campaign artifacts.

Stages: preserve -> train -> summarize -> check (assertions, no new science).
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for _p in ("experiments/e1a933_review", "experiments/f095_campaign",
           "experiments/discovery_campaign", "docs/iclr2027_discovery_campaign_20260917"):
    sys.path.insert(0, str(ROOT / _p))
from d01_capacity import build_model  # noqa: E402
from d03_build_bank import oracle  # noqa: E402
from common import rel_features  # noqa: E402
from data_stats import parent_bootstrap  # noqa: E402

torch.set_num_threads(2)

O01 = ROOT / "artifacts/e1a933_review/data_o01"
OUT = ROOT / "artifacts/e1a933_review/data_o01_continuation"
SCENES = {s: ROOT / f"artifacts/f095_campaign/D01/scenes_{s}/scenes.npz" for s in ("N", "4N")}
SIZES = ["N", "4N"]
ARCHS = ["raw", "sixdist"]
SEEDS = [11, 23, 47]
FRACS = [0, 25, 100]
METHODS = ["plain", "protection", "hard_replay", "preserve_flip_balanced"]
LRS = [0.01, 0.003, 0.001]
STEPS = 300
SUBSET_SEED = 934
PRESERVE_SEED = 260924
# convergence-robust candidates: a different optimizer family and a long low-LR run
ROBUST = {
    "sgd_m09_lr0.01": dict(kind="sgd", lr=0.01, momentum=0.9, steps=300),
    "sgd_m09_lr0.003": dict(kind="sgd", lr=0.003, momentum=0.9, steps=300),
    "adam_lr0.0005_steps600": dict(kind="adam", lr=0.0005, steps=600),
}
_BANK = {}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def init_hash_o01(model):
    """Same convention as the O01 receipts (insertion order, raw values)."""
    return hashlib.sha256(b"".join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def ckpt_state_hash(ckpt, arch):
    model, _ = build_model(arch, ckpt["hidden"], ckpt["feat"])
    model.load_state_dict(ckpt["state"])
    return state_hash(model)


def source_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- data / bank
def build_preserve(size):
    """One real same-label perturbation per mined flip, capped by that flip's norm."""
    path = OUT / f"preserve_{size}.npz"
    if path.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SCENES[size])
    X, y = d["positions"].astype(np.float64), d["labels"]
    fl = np.load(O01 / f"{size}_flips.npz")
    parents, states = fl["parents"], fl["states"]
    rng = np.random.default_rng(PRESERVE_SEED)
    out, labels, flip_index, attempts = [], [], [], 0
    for k in range(len(states)):
        i = int(parents[k])
        cap = float(np.linalg.norm(states[k] - X[i]))
        for _ in range(1000):
            e = rng.normal(size=(4, 2))
            e *= cap / (np.linalg.norm(e) + 1e-12)
            z = X[i] + e
            attempts += 1
            r = oracle(z)
            if r is not None and r[0] == int(y[i]):
                out.append(z)
                labels.append(y[i])
                flip_index.append(k)
                break
        else:
            raise RuntimeError(f"no preserve within budget for {size} flip {k}")
    np.savez_compressed(path, states=np.asarray(out, np.float64), labels=np.asarray(labels, np.float64),
                        flip_index=np.asarray(flip_index, int), attempts=attempts, rng_seed=PRESERVE_SEED)


def bank(size, arch):
    key = (size, arch)
    if key in _BANK:
        return _BANK[key]
    d = np.load(SCENES[size])
    X = d["positions"]  # keep the stored dtype: the O01 campaign normalized the float64 inputs
    fl = np.load(O01 / f"{size}_flips.npz")
    pv = np.load(OUT / f"preserve_{size}.npz")
    dev = np.load(O01 / "dev.npz")
    ev = np.load(O01 / "eval.npz")

    def feats(a):
        a = np.asarray(a)
        return a.reshape(len(a), 8) if arch == "raw" else rel_features(a)[:, :6]

    f = feats(X)
    mu, sd = f.mean(0), f.std(0) + 1e-8

    def norm(a):
        return torch.tensor((feats(a) - mu) / sd, dtype=torch.float32)

    assert len(pv["states"]) == len(fl["labels"]), "preserve pool must align 1:1 with the flip pool"
    b = dict(
        X=X, y=d["labels"], mu=mu, sd=sd,
        xb=norm(X), yb=torch.tensor(d["labels"], dtype=torch.float32),
        xf=norm(fl["states"]), yf=torch.tensor(fl["labels"], dtype=torch.float32),
        flip_parents=fl["parents"],
        xp=norm(pv["states"]), yp=torch.tensor(pv["labels"], dtype=torch.float32),  # noqa: E501
        preserve_attempts=int(pv["attempts"]),
        dev=norm(np.concatenate([dev["positions"], dev["single_states"]])),
        devy=torch.tensor(np.concatenate([dev["labels"], dev["single_labels"]]), dtype=torch.float32),
        ev=norm(ev["states"].reshape(-1, 4, 2)), ev_labels=ev["labels"], ev_parents=ev["parents"],
        sub25=np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(fl["labels"]))[:len(fl["labels"]) // 4]),
        flips_sha=sha_file(O01 / f"{size}_flips.npz"), scenes_sha=sha_file(SCENES[size]),
        preserve_sha=sha_file(OUT / f"preserve_{size}.npz"), eval_sha=sha_file(O01 / "eval.npz"),
        dev_sha=sha_file(O01 / "dev.npz"),
    )
    _BANK[key] = b
    return b


def all_candidates():
    with (O01 / "O01_ALL_CANDIDATES.csv").open() as f:
        return list(csv.DictReader(f))


def clean_cell(size, arch, seed):
    rows = [r for r in all_candidates()
            if r["size"] == size and r["arch"] == arch and int(r["seed"]) == seed and int(r["frac"]) == 0]
    return min(rows, key=lambda r: float(r["dev_bce"]))


def load_clean(size, arch, seed):
    row = clean_cell(size, arch, seed)
    path = O01 / row["name"] / "model.pt"
    return row, path, torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- arm planning
def arm_plan(size, arch, seed, frac, method):
    b = bank(size, arch)
    n = len(b["yf"])
    dose = np.array([], int) if frac == 0 else (b["sub25"] if frac == 25 else np.arange(n))
    if method == "preserve_flip_balanced":
        if len(dose) == 0:
            flips, pres = np.array([], int), np.array([], int)
        else:
            rng = np.random.default_rng([seed, frac, 7])
            nf = len(dose) // 2
            flips = np.sort(rng.choice(dose, nf, replace=False))
            pres = np.sort(np.setdiff1d(dose, flips))
    else:
        flips, pres = np.sort(dose), np.array([], int)
    assert len(flips) + len(pres) == len(dose)
    return flips, pres


def order_hash(npz_path):
    """Hash of the training batch index lists only (independent of the optimizer/LR)."""
    d = np.load(npz_path)
    h = hashlib.sha256()
    for k in ("base", "flips", "preserves", "replay"):
        h.update(k.encode())
        h.update(d[k].tobytes())
    return h.hexdigest()


def make_opt(model, cfg):
    if cfg["kind"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])


def arm_name(start, size, arch, seed, frac, method, opt_key):
    return f"{start}_{size}_{arch}_s{seed}_f{frac}_{method}_{opt_key}"


def run_arm(cfg, outdir):
    """Train one arm; write model.pt / batch_order.npz / batch_order.json / evaluation.npz / receipt.json."""
    t0 = time.monotonic()
    b = bank(cfg["size"], cfg["arch"])
    dest = Path(outdir) / cfg["name"]
    dest.mkdir(parents=True, exist_ok=True)
    if cfg["start"] == "warm":
        clean_row, clean_path, clean = load_clean(cfg["size"], cfg["arch"], cfg["seed"])
        model, _ = build_model(cfg["arch"], 64, 32)
        model.load_state_dict(clean["state"])
        assert np.allclose(clean["mu"], b["mu"], atol=1e-6) and np.allclose(clean["sd"], b["sd"], atol=1e-6), \
            "clean checkpoint normalization must match the O01 bank features"
        clean_state, clean_sha = state_hash(model), sha_file(clean_path)
        with torch.no_grad():
            ref_logits = model(b["xb"])
            dev_clean_bce = float(nn.functional.binary_cross_entropy_with_logits(model(b["dev"]), b["devy"]))
        ref_soft = ref_logits.sigmoid()
        ref_ok = (ref_logits > 0) == b["yb"].bool()
    else:
        assert cfg["method"] == "plain", "scratch arms are the plain training-path control only"
        torch.manual_seed(cfg["seed"])
        model, _ = build_model(cfg["arch"], 64, 32)
        clean_state, clean_sha, ref_soft, ref_ok, dev_clean_bce = None, None, None, None, ""
    init_state, init_o01 = state_hash(model), init_hash_o01(model)

    flips, pres = arm_plan(cfg["size"], cfg["arch"], cfg["seed"], cfg["frac"], cfg["method"])
    opt_cfg = cfg["opt_cfg"]
    steps = opt_cfg["steps"]
    replay = [] if ref_ok is None else torch.nonzero(ref_ok).reshape(-1).tolist()
    order = dict(base=list(range(len(b["yb"]))), flips=flips.tolist(), preserves=pres.tolist(),
                 replay=replay, repeated_for_steps=steps, optimizer=cfg["opt_key"])
    np.savez_compressed(dest / "batch_order.npz", base=np.asarray(order["base"], np.int32),
                        flips=np.asarray(order["flips"], np.int32),
                        preserves=np.asarray(order["preserves"], np.int32),
                        replay=np.asarray(order["replay"], np.int32))
    order_sha = order_hash(dest / "batch_order.npz")
    (dest / "batch_order.json").write_text(json.dumps(
        dict(repeated_for_steps=steps, optimizer=cfg["opt_key"], batch_order_sha256=order_sha,
             batch_order_metadata_sha256=hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest(),
             hash_scope="index lists only (base/flips/preserves/replay); optimizer and steps are metadata",
             counts=dict(base=len(order["base"]), flips=len(order["flips"]),
                         preserves=len(order["preserves"]), replay=len(order["replay"]))), indent=2))

    xa = torch.cat([b["xf"][flips], b["xp"][pres]]) if len(flips) + len(pres) else None
    ya = torch.cat([b["yf"][flips], b["yp"][pres]]) if xa is not None else None
    coef = 1.0 if cfg["method"] in ("protection", "hard_replay") else 0.0
    if cfg["method"] == "protection":
        replay_target = ref_soft[ref_ok]
    else:
        replay_target = b["yb"][ref_ok] if ref_ok is not None else None
    opt = make_opt(model, opt_cfg)
    bce = nn.BCEWithLogitsLoss()
    curve = []
    for _ in range(steps):
        opt.zero_grad()
        loss = bce(model(b["xb"]), b["yb"])
        if xa is not None:
            loss = loss + bce(model(xa), ya)
        if ref_ok is not None:
            loss = loss + coef * bce(model(b["xb"][ref_ok]), replay_target)
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))

    with torch.no_grad():
        dev_bce = float(bce(model(b["dev"]), b["devy"]))
        train_error = float(((model(b["xb"]) > 0) != b["yb"].bool()).float().mean())
        scores = model(b["ev"]).numpy().reshape(-1, 3)
    correct = (scores > 0) == b["ev_labels"]
    final_state = state_hash(model)
    torch.save(dict(state=model.state_dict(), hidden=64, feat=32,
                    in_dim=8 if cfg["arch"] == "raw" else 6, mu=b["mu"], sd=b["sd"],
                    seed=cfg["seed"], arch=cfg["arch"], size=cfg["size"], start=cfg["start"],
                    method=cfg["method"], opt_key=cfg["opt_key"], steps=steps), dest / "model.pt")
    np.savez_compressed(dest / "evaluation.npz", scores=scores, labels=b["ev_labels"],
                        parents=b["ev_parents"], correct=correct)
    (dest / "loss.json").write_text(json.dumps(curve))
    forwards_per_step = len(order["base"]) + len(order["flips"]) + len(order["preserves"]) + len(order["replay"])
    row = dict(name=cfg["name"], start=cfg["start"], size=cfg["size"], arch=cfg["arch"], seed=cfg["seed"],
               frac=cfg["frac"], method=cfg["method"], opt_key=cfg["opt_key"], opt_kind=opt_cfg["kind"],
               lr=opt_cfg["lr"], momentum=opt_cfg.get("momentum", ""), steps=steps,
               dev_bce=dev_bce, dev_clean_bce=dev_clean_bce, train_error=train_error, final_loss=curve[-1],
               init_state_sha256=init_state, init_sha256_o01=init_o01, final_state_sha256=final_state,
               clean_state_sha256=clean_state or "", clean_checkpoint_sha256=clean_sha or "",
               clean_candidate=(clean_row["name"] if cfg["start"] == "warm" else ""),
               checkpoint_sha256=sha_file(dest / "model.pt"), batch_order_sha256=order_sha,
               flips_used=len(order["flips"]), preserves_used=len(order["preserves"]),
               replay_points=len(order["replay"]), forwards_per_step=forwards_per_step,
               total_forward_passes=steps * forwards_per_step, total_backward_passes=steps,
               train_sha256=b["scenes_sha"], flip_sha256=b["flips_sha"], preserve_sha256=b["preserve_sha"],
               dev_sha256=b["dev_sha"], eval_sha256=b["eval_sha"], wall_seconds=time.monotonic() - t0,
               J=float(correct.all(1).mean()), atomic_pass=float(correct[:, :2].all(1).mean()),
               AB_pass=float(correct[:, 2].mean()))
    (dest / "receipt.json").write_text(json.dumps(row, indent=2))
    return row


def iter_factorial():
    for size in SIZES:
        for arch in ARCHS:
            for seed in SEEDS:
                for frac in FRACS:
                    for method in METHODS:
                        for lr in LRS:
                            key = f"lr{lr}"
                            yield dict(name=arm_name("warm", size, arch, seed, frac, method, key),
                                       start="warm", size=size, arch=arch, seed=seed, frac=frac,
                                       method=method, opt_key=key,
                                       opt_cfg=dict(kind="adam", lr=lr, steps=STEPS))


def iter_robust():
    for start in ("warm", "scratch"):
        for size in SIZES:
            for arch in ARCHS:
                for seed in SEEDS:
                    for frac in FRACS:
                        for key, cfg in ROBUST.items():
                            yield dict(name=arm_name(start, size, arch, seed, frac, "plain", key),
                                       start=start, size=size, arch=arch, seed=seed, frac=frac,
                                       method="plain", opt_key=key, opt_cfg=dict(cfg))


def all_cfgs():
    return list(iter_factorial()) + list(iter_robust())


# ---------------------------------------------------------------- stages
def stage_preserve():
    for size in SIZES:
        build_preserve(size)
        d = np.load(OUT / f"preserve_{size}.npz")
        print(f"preserve {size}: {len(d['states'])} real preserve labels, "
              f"{int(d['attempts'])} oracle attempts", flush=True)


def stage_train(out, only=None):
    cfgs = all_cfgs()
    if only:
        cfgs = [c for c in cfgs if c["name"] == only]
        assert len(cfgs) == 1, f"--only matched {len(cfgs)} arms"
    done = 0
    for cfg in cfgs:
        if (Path(out) / cfg["name"] / "receipt.json").exists():
            continue
        row = run_arm(cfg, out)
        done += 1
        print(f"[{done}] {cfg['name']} dev_bce={row['dev_bce']:.4f} J={row['J']:.4f} "
              f"fwd={row['total_forward_passes']} {row['wall_seconds']:.1f}s", flush=True)
    (Path(out) / "TRAIN_RECEIPT.json").write_text(json.dumps(
        dict(stage="train", source_commit=source_commit(), argv=sys.argv,
             threads=torch.get_num_threads(), lr_grid=LRS, robust=ROBUST, steps=STEPS,
             arms_factorial=len(list(iter_factorial())), arms_robust=len(list(iter_robust())),
             selection="static/single development BCE only; eval pool never used for selection"), indent=2))


def load_rows(out):
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("*/receipt.json"))]


def ref_scores_clean(size, arch, seed):
    b = bank(size, arch)
    _, _, ck = load_clean(size, arch, seed)
    model, _ = build_model(arch, 64, 32)
    model.load_state_dict(ck["state"])
    model.eval()
    with torch.no_grad():
        return (model(b["ev"]).numpy().reshape(-1, 3) > 0) == b["ev_labels"]


def o01_selected(size, arch, seed, frac):
    """Selected O01 scratch plain arm for this cell and dose (read-only artifact)."""
    with (O01 / "O01_FACTORIAL.csv").open() as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if r["size"] == size and r["arch"] == arch and int(r["seed"]) == seed
           and int(r["frac"]) == frac]
    assert len(hit) == 1, hit
    r = hit[0]
    d = np.load(O01 / f"{r['name']}_scores.npz")
    assert np.array_equal(d["parents"], np.load(O01 / "eval.npz")["parents"])
    return dict(name=r["name"], ok=(d["scores"] > 0) == d["labels"], dev_bce=float(r["dev_bce"]),
                lr=float(r["lr"]), receipt=r)


def transitions(before, after):
    bit = np.array([4, 2, 1])
    a, b = before.astype(int) @ bit, after.astype(int) @ bit
    m = np.zeros((8, 8), int)
    np.add.at(m, (a, b), 1)
    return m


def arm_metrics(ok, ref_ok, parents):
    at, nt = ref_ok[:, :2], ok[:, :2]
    old_joint = ref_ok.all(1)
    unique = np.unique(parents)
    rng = np.random.default_rng(924)
    draws = rng.integers(0, len(unique), (2000, len(unique)))

    def cluster(num, den):
        n = np.array([num[parents == p].sum() for p in unique], float)
        d = np.array([den[parents == p].sum() for p in unique], float)
        ratio = n[draws].sum(1) / np.maximum(d[draws].sum(1), 1)
        return np.quantile(ratio, [.025, .975]).tolist()

    reg = at & ~nt
    lost = old_joint & ~ok.all(1)
    gained = ~old_joint & ok.all(1)
    return dict(
        J=float(ok.all(1).mean()), A=float(ok[:, 0].mean()), B=float(ok[:, 1].mean()),
        AB=float(ok[:, 2].mean()), atomic_pass=float(nt.all(1).mean()),
        n_quartets=int(len(ok)), n_parents=int(len(unique)),
        ref_atomic_correct_n=int(at.sum()), atomic_regressed_n=int(reg.sum()),
        atomic_regression=float(reg.sum() / max(at.sum(), 1)),
        atomic_regression_CI_parent_bootstrap=cluster(reg, at),
        ref_joint_correct_n=int(old_joint.sum()), joint_lost_n=int(lost.sum()),
        joint_lost=float(lost.sum() / max(old_joint.sum(), 1)),
        joint_gained_n=int(gained.sum()), joint_gained=float(gained.sum() / max((~old_joint).sum(), 1)),
        transition_8x8=transitions(ref_ok, ok).tolist(),
    )


def paired(ok_a, ok_b, parents, ref_a=None, ref_b=None):
    out = {"J_delta": parent_bootstrap(ok_a.all(1).astype(float) - ok_b.all(1).astype(float), parents)}
    if ref_a is not None and ref_b is not None:
        ra = (ref_a[:, :2] & ~ok_a[:, :2]).sum(1).astype(float)
        rb = (ref_b[:, :2] & ~ok_b[:, :2]).sum(1).astype(float)
        out["atomic_regression_delta"] = parent_bootstrap(ra - rb, parents)
    return out


def stage_summarize(out):
    out = Path(out)
    parents = bank("N", "raw")["ev_parents"]
    rows = load_rows(out)
    by_name = {r["name"]: r for r in rows}
    ok_cache = {}

    def ok_of(name):
        if name not in ok_cache:
            d = np.load(out / name / "evaluation.npz")
            assert np.array_equal(d["parents"], parents)
            ok_cache[name] = d["correct"]
        return ok_cache[name]

    warm_sel, robust_sel = {}, {}
    for r in rows:
        if r["opt_key"] in ROBUST:
            key = (r["start"], r["size"], r["arch"], r["seed"], r["frac"])
            if key not in robust_sel or r["dev_bce"] < robust_sel[key]["dev_bce"]:
                robust_sel[key] = r
        else:
            key = (r["size"], r["arch"], r["seed"], r["frac"], r["method"])
            if key not in warm_sel or r["dev_bce"] < warm_sel[key]["dev_bce"]:
                warm_sel[key] = r

    # disclosed selection guard for the robust family: dev BCE alone can prefer a
    # near-constant solution in the scratch/raw/dose-0 corner (train error 36-38%).
    # The guard only removes degenerate candidates; it never looks at evaluation J.
    guard_groups, guard_sel, guarded = {}, {}, {}
    for r in rows:
        if r["opt_key"] in ROBUST:
            guard_groups.setdefault((r["start"], r["size"], r["arch"], r["seed"], r["frac"]), []).append(r)
    for key, g in sorted(guard_groups.items()):
        sel = min(g, key=lambda r: r["dev_bce"])
        live = [r for r in g if r["train_error"] <= 0.20]
        gsel = min(live, key=lambda r: r["dev_bce"]) if live else sel
        guard_sel[key] = gsel
        guarded["|".join(map(str, key))] = dict(
            dev_selected=sel["name"], dev_selected_dev_bce=sel["dev_bce"], dev_selected_J=sel["J"],
            dev_selected_train_error=sel["train_error"], guarded_selected=gsel["name"],
            guarded_dev_bce=gsel["dev_bce"], guarded_J=gsel["J"], guard_changed_selection=bool(gsel is not sel))

    clean_ok = {(s, a, sd): ref_scores_clean(s, a, sd) for s in SIZES for a in ARCHS for sd in SEEDS}
    o01_ok = {(s, a, sd, fr): o01_selected(s, a, sd, fr)
              for s in SIZES for a in ARCHS for sd in SEEDS for fr in FRACS}

    def find_warm(size, arch, seed, frac, method, lr):
        return by_name.get(arm_name("warm", size, arch, seed, frac, method, f"lr{lr}"))

    # ---- per-arm table (selected and non-selected runs, with reference-relative metrics)
    table = []
    for r in rows:
        size, arch, seed, frac, method = r["size"], r["arch"], r["seed"], r["frac"], r["method"]
        refs = {}
        if r["opt_key"] in ROBUST:
            fam = "robust"
            noflip = by_name[arm_name(r["start"], size, arch, seed, 0, "plain", r["opt_key"])]
            refs["matched_noflip"] = ok_of(noflip["name"])
            if r["start"] == "warm":
                refs["clean_start"] = clean_ok[(size, arch, seed)]
            row = dict(r)
            row["family"] = fam
            row["selected_by_dev_bce"] = int(robust_sel[(r["start"], size, arch, seed, frac)]["name"] == r["name"])
        else:
            fam = "warm_factorial"
            noflip = find_warm(size, arch, seed, 0, "plain", r["lr"])
            assert noflip is not None, (size, arch, seed, r["lr"])
            refs["matched_noflip"] = clean_ok[(size, arch, seed)] if (frac == 0 and method == "plain") \
                else ok_of(noflip["name"])
            refs["clean_start"] = clean_ok[(size, arch, seed)]
            sel = warm_sel[(size, arch, seed, frac, method)]
            row = dict(r)
            row["family"] = fam
            row["selected_by_dev_bce"] = int(sel["name"] == r["name"])
            row["selected_lr"] = sel["lr"]
        for rname, ref in refs.items():
            for k, v in arm_metrics(ok_of(r["name"]), ref, parents).items():
                row[f"{rname}__{k}"] = json.dumps(v) if isinstance(v, list) else v
        table.append(row)
    fieldnames = []
    for row in table:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    # ---- paired contrasts, parent-cluster bootstrap over the 250 eligible evaluation parents
    contrasts = {}
    for size in SIZES:
        for seed in SEEDS:
            c = {}
            for arch in ARCHS:
                for frac in (25, 100):
                    a = warm_sel[(size, arch, seed, frac, "plain")]
                    b = warm_sel[(size, arch, seed, 0, "plain")]
                    c[f"warm_{arch}_plain_f{frac}_minus_f0"] = paired(
                        ok_of(a["name"]), ok_of(b["name"]), parents,
                        clean_ok[(size, arch, seed)], clean_ok[(size, arch, seed)])
                for method in METHODS[1:]:
                    for frac in (25, 100):
                        a = warm_sel[(size, arch, seed, frac, method)]
                        b = warm_sel[(size, arch, seed, frac, "plain")]
                        c[f"warm_{arch}_{method}_minus_plain_f{frac}"] = paired(
                            ok_of(a["name"]), ok_of(b["name"]), parents,
                            clean_ok[(size, arch, seed)], clean_ok[(size, arch, seed)])
            for frac in FRACS:
                six = warm_sel[(size, "sixdist", seed, frac, "plain")]
                raw = warm_sel[(size, "raw", seed, frac, "plain")]
                c[f"warm_six_minus_raw_f{frac}"] = paired(
                    ok_of(six["name"]), ok_of(raw["name"]), parents,
                    clean_ok[(size, "sixdist", seed)], clean_ok[(size, "raw", seed)])
                c[f"scratch_six_minus_raw_f{frac}"] = paired(
                    o01_ok[(size, "sixdist", seed, frac)]["ok"], o01_ok[(size, "raw", seed, frac)]["ok"], parents,
                    o01_ok[(size, "sixdist", seed, 0)]["ok"], o01_ok[(size, "raw", seed, 0)]["ok"])
            c["warm_six25_minus_raw100"] = paired(
                ok_of(warm_sel[(size, "sixdist", seed, 25, "plain")]["name"]),
                ok_of(warm_sel[(size, "raw", seed, 100, "plain")]["name"]), parents,
                clean_ok[(size, "sixdist", seed)], clean_ok[(size, "raw", seed)])
            c["scratch_six25_minus_raw100"] = paired(
                o01_ok[(size, "sixdist", seed, 25)]["ok"], o01_ok[(size, "raw", seed, 100)]["ok"], parents,
                o01_ok[(size, "sixdist", seed, 0)]["ok"], o01_ok[(size, "raw", seed, 0)]["ok"])
            for frac in (25, 100):
                v_warm = (ok_of(warm_sel[(size, "sixdist", seed, frac, "plain")]["name"]).all(1).astype(float)
                          - ok_of(warm_sel[(size, "sixdist", seed, 0, "plain")]["name"]).all(1)
                          - ok_of(warm_sel[(size, "raw", seed, frac, "plain")]["name"]).all(1)
                          + ok_of(warm_sel[(size, "raw", seed, 0, "plain")]["name"]).all(1))
                v_scratch = (o01_ok[(size, "sixdist", seed, frac)]["ok"].all(1).astype(float)
                             - o01_ok[(size, "sixdist", seed, 0)]["ok"].all(1)
                             - o01_ok[(size, "raw", seed, frac)]["ok"].all(1)
                             + o01_ok[(size, "raw", seed, 0)]["ok"].all(1))
                c[f"warm_I_f{frac}"] = parent_bootstrap(v_warm, parents)
                c[f"scratch_I_f{frac}"] = parent_bootstrap(v_scratch, parents)
            # scratch vs warm-start at the O01-selected scratch learning rate (same recipe, different path)
            for arch in ARCHS:
                for frac in FRACS:
                    ref = o01_ok[(size, arch, seed, frac)]
                    w = by_name[arm_name("warm", size, arch, seed, frac, "plain", f"lr{ref['lr']}")]
                    c[f"warm_minus_scratch_{arch}_f{frac}_lr{ref['lr']}"] = paired(
                        ok_of(w["name"]), ref["ok"], parents, clean_ok[(size, arch, seed)],
                        o01_ok[(size, arch, seed, 0)]["ok"])
            # equal-recipe (matched learning rate) comparisons: separate recipe from LR selection
            for lr in LRS:
                for frac in FRACS:
                    six = find_warm(size, "sixdist", seed, frac, "plain", lr)
                    raw = find_warm(size, "raw", seed, frac, "plain", lr)
                    c[f"equalLR{lr}_six_minus_raw_f{frac}"] = paired(
                        ok_of(six["name"]), ok_of(raw["name"]), parents,
                        clean_ok[(size, "sixdist", seed)], clean_ok[(size, "raw", seed)])
                c[f"equalLR{lr}_six25_minus_raw100"] = paired(
                    ok_of(find_warm(size, "sixdist", seed, 25, "plain", lr)["name"]),
                    ok_of(find_warm(size, "raw", seed, 100, "plain", lr)["name"]), parents,
                    clean_ok[(size, "sixdist", seed)], clean_ok[(size, "raw", seed)])
                for arch in ARCHS:
                    for method in METHODS[1:]:
                        for frac in (25, 100):
                            c[f"equalLR{lr}_warm_{arch}_{method}_minus_plain_f{frac}"] = paired(
                                ok_of(find_warm(size, arch, seed, frac, method, lr)["name"]),
                                ok_of(find_warm(size, arch, seed, frac, "plain", lr)["name"]), parents,
                                clean_ok[(size, arch, seed)], clean_ok[(size, arch, seed)])
            # convergence-robust candidates
            for start in ("warm", "scratch"):
                for frac in FRACS:
                    six = robust_sel[(start, size, "sixdist", seed, frac)]
                    raw = robust_sel[(start, size, "raw", seed, frac)]
                    c[f"robust_{start}_six_minus_raw_f{frac}"] = paired(ok_of(six["name"]), ok_of(raw["name"]),
                                                                       parents)
                c[f"robust_{start}_six25_minus_raw100"] = paired(
                    ok_of(robust_sel[(start, size, "sixdist", seed, 25)]["name"]),
                    ok_of(robust_sel[(start, size, "raw", seed, 100)]["name"]), parents)
                for frac in FRACS:
                    c[f"robust_guarded_{start}_six_minus_raw_f{frac}"] = paired(
                        ok_of(guard_sel[(start, size, "sixdist", seed, frac)]["name"]),
                        ok_of(guard_sel[(start, size, "raw", seed, frac)]["name"]), parents)
                c[f"robust_guarded_{start}_six25_minus_raw100"] = paired(
                    ok_of(guard_sel[(start, size, "sixdist", seed, 25)]["name"]),
                    ok_of(guard_sel[(start, size, "raw", seed, 100)]["name"]), parents)
            contrasts[f"{size}_s{seed}"] = c

    # ---- oracle cost ledgers (measured, not assumed)
    ledger = dict(statement=("the 25% subset is drawn AFTER mining the full pool, so mining/oracle cost is "
                             "identical for every dose; only retained training labels differ. This cannot be "
                             "used to claim 75% fewer oracle queries."),
                  flip_mining={}, dev_single_mining={}, eval_bank_mining={}, preserve_mining={}, training_side={})
    for size in SIZES:
        fl = np.load(O01 / f"{size}_flips.npz")
        pv = np.load(OUT / f"preserve_{size}.npz")
        ledger["flip_mining"][size] = dict(
            source=f"artifacts/e1a933_review/data_o01/{size}_flips.npz",
            candidate_queries=int(fl["oracle_calls"]), retained_flip_labels=int(len(fl["labels"])),
            parents=int(len(np.unique(fl["parents"]))))
        ledger["preserve_mining"][size] = dict(
            source=f"artifacts/e1a933_review/data_o01_continuation/preserve_{size}.npz",
            oracle_attempts=int(pv["attempts"]), retained_preserve_labels=int(len(pv["labels"])))
        ledger["training_side"][size] = {}
        for frac in FRACS:
            arm = warm_sel[(size, "raw", 11, frac, "plain")]
            ledger["training_side"][size][{0: "f0", 25: "f25", 100: "f100"}[frac]] = dict(
                example_arm=arm["name"], base_scenes=int(len(np.load(SCENES[size])["positions"])),
                retained_flip_labels_used=int(arm["flips_used"]), preserve_labels_used=int(arm["preserves_used"]),
                replay_points=int(arm["replay_points"]), forwards_per_step=int(arm["forwards_per_step"]),
                total_forward_passes=int(arm["total_forward_passes"]),
                total_backward_passes=int(arm["total_backward_passes"]),
                note="every warm arm performs the replay forward; its coefficient is zero for plain/balanced")
    dev = np.load(O01 / "dev.npz")
    man = json.loads((O01 / "bank_manifest.json").read_text())
    ledger["dev_single_mining"] = dict(source="artifacts/e1a933_review/data_o01/dev.npz",
                                       candidate_queries=int(dev["single_oracle_calls"]),
                                       retained_single_labels=int(len(dev["single_labels"])), parents=256)
    ledger["eval_bank_mining"] = dict(source="artifacts/e1a933_review/data_o01/bank_manifest.json",
                                      candidate_and_pair_queries=int(man["oracle_calls"]),
                                      quartets=int(len(np.load(O01 / "eval.npz")["labels"])),
                                      eligible_parents=250, drawn_evaluation_parents=512)
    ledger["oracle_queries_total"] = int(
        sum(ledger["flip_mining"][s]["candidate_queries"] for s in SIZES)
        + ledger["dev_single_mining"]["candidate_queries"]
        + ledger["eval_bank_mining"]["candidate_and_pair_queries"]
        + sum(ledger["preserve_mining"][s]["oracle_attempts"] for s in SIZES))

    with (out / "O01_CONTINUATION_FACTORIAL.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(table)
    with (out / "ARMS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # ---- failure / collapse disclosure: no run is dropped, all of them are counted here
    groups = {}
    for r in rows:
        fam = "robust" if r["opt_key"] in ROBUST else "warm_factorial"
        groups.setdefault(f"{fam}|{r['start']}|{r['arch']}|f{r['frac']}", []).append(r)
    collapse = dict(note=("every run is retained; collapsed or degenerate configurations are counted here and "
                          "are not used as evidence for or against any loss/regularizer"),
                    groups={k: dict(n_runs=len(v), n_J_under_0p05=sum(1 for r in v if r["J"] < 0.05),
                                    min_J=min(r["J"] for r in v),
                                    median_J=float(np.median([r["J"] for r in v])),
                                    max_J=max(r["J"] for r in v),
                                    n_train_error_over_0p05=sum(1 for r in v if r["train_error"] > 0.05),
                                    min_dev_bce=min(r["dev_bce"] for r in v),
                                    max_dev_bce=max(r["dev_bce"] for r in v))
                            for k, v in sorted(groups.items())})
    (out / "O01_CONTINUATION_PAIRED.json").write_text(json.dumps(contrasts, indent=2))
    (out / "ORACLE_LEDGER.json").write_text(json.dumps(ledger, indent=2))
    (out / "COLLAPSE_SUMMARY.json").write_text(json.dumps(collapse, indent=2))
    (out / "ROBUST_GUARDED_SELECTION.json").write_text(json.dumps(guarded, indent=2))
    summary = {}
    for tag, c in contrasts.items():
        for k, v in c.items():
            if "estimate" in v:
                summary[f"{tag}|{k}"] = dict(estimate=v["estimate"], ci95=v["ci95"])
            else:
                for sub, vv in v.items():
                    summary[f"{tag}|{k}|{sub}"] = dict(estimate=vv["estimate"], ci95=vv["ci95"])
    (out / "SUMMARY.json").write_text(json.dumps(
        dict(contrasts=summary, n_selected_warm=len(warm_sel), n_selected_robust=len(robust_sel),
             n_runs=len(rows)), indent=2))
    print("table rows", len(table), "arms", len(rows), flush=True)
    print("wrote", out / "O01_CONTINUATION_FACTORIAL.csv", out / "O01_CONTINUATION_PAIRED.json", flush=True)


def stage_check(out):
    """Determinism / contract assertions. Runs a few arms standalone; no new science."""
    out = Path(out)
    tmp = out / "check_single_runs"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = {r["name"]: r for r in load_rows(out)}
    cfgs = {c["name"]: c for c in all_cfgs()}
    checks = []

    def record(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))

    # 1. single run vs batch run: identical init state, batch list, final state
    for name in [arm_name("warm", "N", "raw", 11, 100, "plain", "lr0.003"),
                 arm_name("warm", "4N", "sixdist", 23, 25, "preserve_flip_balanced", "lr0.001"),
                 arm_name("warm", "N", "raw", 47, 100, "hard_replay", "lr0.01")]:
        single = run_arm(cfgs[name], tmp)
        batch = rows[name]
        record(f"single_vs_batch::{name}",
               single["init_state_sha256"] == batch["init_state_sha256"]
               and single["batch_order_sha256"] == batch["batch_order_sha256"]
               and single["final_state_sha256"] == batch["final_state_sha256"],
               dict(init_equal=single["init_state_sha256"] == batch["init_state_sha256"],
                    order_equal=single["batch_order_sha256"] == batch["batch_order_sha256"],
                    final_state_equal=single["final_state_sha256"] == batch["final_state_sha256"],
                    dev_bce_single=single["dev_bce"], dev_bce_batch=batch["dev_bce"]))

    # 2. same clean start for every arm of a cell; batch list depends only on (cell, dose, method)
    for size in SIZES:
        for arch in ARCHS:
            for seed in SEEDS:
                pre = f"warm_{size}_{arch}_s{seed}_"
                inits = {rows[n]["init_state_sha256"] for n in rows if n.startswith(pre)}
                cleans = {rows[n]["clean_state_sha256"] for n in rows if n.startswith(pre)}
                record(f"same_start::{size}_{arch}_s{seed}", len(inits) == 1 and len(cleans) == 1,
                       dict(distinct_init=len(inits), distinct_clean=len(cleans), init=sorted(inits)[0]))
                for frac in FRACS:
                    for method in METHODS:
                        names = [n for n in rows if n.startswith(f"{pre}f{frac}_{method}_lr")]
                        orders = {rows[n]["batch_order_sha256"] for n in names}
                        record(f"same_batch_list::{size}_{arch}_s{seed}_f{frac}_{method}",
                               len(orders) == 1 and len(names) == len(LRS),
                               dict(distinct_orders=len(orders), n_arms=len(names)))

    # 3. the clean checkpoint is exactly re-derivable from random init (bit-identical)
    for size in SIZES:
        for arch in ARCHS:
            for seed in SEEDS:
                row, path, ck = load_clean(size, arch, seed)
                b = bank(size, arch)
                torch.manual_seed(seed)
                m, _ = build_model(arch, 64, 32)
                opt = torch.optim.Adam(m.parameters(), lr=float(row["lr"]))
                bce = nn.BCEWithLogitsLoss()
                for _ in range(STEPS):
                    opt.zero_grad()
                    bce(m(b["xb"]), b["yb"]).backward()
                    opt.step()
                record(f"clean_rederived::{size}_{arch}_s{seed}",
                       state_hash(m) == ckpt_state_hash(ck, arch),
                       dict(o01_candidate=row["name"], checkpoint_sha256=sha_file(path)))

    # 4. this harness reproduces the O01 scratch campaign bit-for-bit at matched settings
    for size, arch, seed, frac, lr in [("N", "raw", 11, 100, 0.001), ("4N", "sixdist", 23, 25, 0.003)]:
        name = arm_name("scratch", size, arch, seed, frac, "plain", f"lr{lr}")
        row = run_arm(dict(name=name, start="scratch", size=size, arch=arch, seed=seed, frac=frac,
                           method="plain", opt_key=f"lr{lr}", opt_cfg=dict(kind="adam", lr=lr, steps=STEPS)), tmp)
        ref = torch.load(O01 / f"{size}_{arch}_s{seed}_f{frac}_lr{lr}" / "model.pt",
                         map_location="cpu", weights_only=False)
        record(f"o01_scratch_reproduced::{name}", row["final_state_sha256"] == ckpt_state_hash(ref, arch),
               dict(mine=row["final_state_sha256"], o01=ckpt_state_hash(ref, arch), dev_bce=row["dev_bce"]))

    # 5. balanced at dose 0 equals plain at dose 0 (empty augmentation, zero replay coefficient)
    for size in SIZES:
        for arch in ARCHS:
            for seed in SEEDS:
                for lr in LRS:
                    a = rows.get(arm_name("warm", size, arch, seed, 0, "plain", f"lr{lr}"))
                    bb = rows.get(arm_name("warm", size, arch, seed, 0, "preserve_flip_balanced", f"lr{lr}"))
                    if a and bb:
                        record(f"balanced_f0_eq_plain_f0::{size}_{arch}_s{seed}_lr{lr}",
                               a["final_state_sha256"] == bb["final_state_sha256"],
                               dict(equal=a["final_state_sha256"] == bb["final_state_sha256"]))

    report = dict(source_commit=source_commit(), threads=torch.get_num_threads(), checks=checks,
                  all_passed=all(c["passed"] for c in checks))
    (out / "CHECK_REPORT.json").write_text(json.dumps(report, indent=2))
    print(f"checks {sum(c['passed'] for c in checks)}/{len(checks)} passed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preserve", "train", "summarize", "check"], required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT = a.out_dir
    {"preserve": stage_preserve, "train": lambda: stage_train(OUT, a.only),
     "summarize": lambda: stage_summarize(OUT), "check": lambda: stage_check(OUT)}[a.stage]()
