#!/usr/bin/env python3
"""Source-task contrast for three L015 explanations. Not a new method.

Explanations kept separate:
  missing cross-segment interaction, extra midpoint/angle features, G8 sorting.

Claim denominator is a new parent bank. A quartet is the official pair in
experiments/p123_upgrade/build_bank.py: two label-preserving atomic edits
whose sum flips the label. leads_l014_l015_l006.mine_eval is not used.
dev512 is not read. Sealed pools are not read.

Train only on artifacts/f095_campaign/D01/scenes_N/scenes.npz.
Selection uses static+single dev BCE only. Test quartet J is not a selection input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "e1a933_review"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))

from coord_mlp import CoordMLP  # noqa: E402
from common import rel_features  # noqa: E402
from core.relations import make_relational_scenes  # noqa: E402
from d03_build_bank import candidates_for_scene, oracle as flip_oracle  # noqa: E402
from leads_l014_l015_l006 import G8, apply_perm, g8_features  # noqa: E402
from paths import atomic_edits, oracle_at  # noqa: E402
from u01_models import TypedPairMLP, count_params  # noqa: E402

ART = ROOT / "artifacts" / "e832_focus" / "structure"
TRAIN_PATH = ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_N" / "scenes.npz"
SCENE_SEED = 832025
EDIT_SEED = 832028
DEV_FLIP_SEED = 832027
TRAIN_FLIP_SEED = 5
DEV_PARENTS = 256
TEST_PARENTS = 4096
DEDUP_LINF = 1e-4
MIN_ELIGIBLE = 200
EXTRA_CHUNK = 2048
EXTRA_SEED0 = 832125
EPOCHS = 300
LRS = (0.01, 0.003, 0.001)
SEEDS = (11, 23, 47)
EXTRA_SEEDS = (71, 83)
THREADS = 4
BOOT_SEED = 924
BOOT_N = 2000
RHO_K = 16
MARGIN_SLICE = 0.02
OPT_FAIL_BCE = 0.60

torch.set_num_threads(THREADS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def is_official_quartet(y0, ya, yb, yab) -> bool:
    """build_bank.py pair rule: both edits keep the label, the sum flips it."""
    return bool(ya == y0 and yb == y0 and yab != y0)


def rich8_unsorted(x) -> np.ndarray:
    """Two lengths, four cross distances, midpoint distance, absolute sine.

    Fixed endpoint order. No sort. Same continuous quantities as g8_features
    before that function sorts the length pair and the four cross distances.
    """
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    length_ab = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1)
    length_cd = np.linalg.norm(x[:, 2] - x[:, 3], axis=-1)
    cross = np.stack([
        np.linalg.norm(x[:, 0] - x[:, 2], axis=-1),
        np.linalg.norm(x[:, 0] - x[:, 3], axis=-1),
        np.linalg.norm(x[:, 1] - x[:, 2], axis=-1),
        np.linalg.norm(x[:, 1] - x[:, 3], axis=-1),
    ], axis=1)
    mid = np.linalg.norm(0.5 * (x[:, 0] + x[:, 1]) - 0.5 * (x[:, 2] + x[:, 3]), axis=-1)
    u0 = x[:, 1] - x[:, 0]
    u1 = x[:, 3] - x[:, 2]
    sine = np.abs(u0[:, 0] * u1[:, 1] - u0[:, 1] * u1[:, 0])
    sine = sine / (np.linalg.norm(u0, axis=-1) * np.linalg.norm(u1, axis=-1) + 1e-8)
    return np.concatenate(
        [length_ab[:, None], length_cd[:, None], cross, mid[:, None], sine[:, None]], axis=1
    ).astype(np.float32)


def rich8_sorted(x) -> np.ndarray:
    """Existing G8 sort of the same eight quantities. Not a new feature set."""
    return g8_features(x).astype(np.float32)


def six_ordered(x) -> np.ndarray:
    """Six pairwise distances in triu endpoint order, not sorted."""
    return rel_features(x)[:, :6].astype(np.float32)


def min_linf(query, ref, q_block=128, r_block=512) -> np.ndarray:
    q = np.asarray(query, np.float64).reshape(len(query), -1)
    r = np.asarray(ref, np.float64).reshape(len(ref), -1)
    out = np.empty(len(q), np.float64)
    for i in range(0, len(q), q_block):
        qq = q[i:i + q_block]
        best = np.full(len(qq), np.inf)
        for j in range(0, len(r), r_block):
            diff = np.abs(qq[:, None, :] - r[j:j + r_block][None, :, :]).max(axis=-1)
            best = np.minimum(best, diff.min(axis=1))
        out[i:i + q_block] = best
    return out


def greedy_dedup(x, threshold):
    """Drop later rows within L-inf threshold of an earlier kept row."""
    flat = np.asarray(x, np.float64).reshape(len(x), -1)
    keep = []
    dropped = 0
    kept_arr = np.zeros((0, flat.shape[1]))
    for i in range(len(flat)):
        if len(kept_arr):
            d = np.abs(kept_arr - flat[i]).max(axis=1).min()
            if d < threshold:
                dropped += 1
                continue
        keep.append(i)
        kept_arr = flat[keep]
    return np.asarray(keep, np.int64), dropped


class RepairedRho(nn.Module):
    """TypedPairMLP with a nonlinearity after the segment sum.

    Segment roles and segment-swap symmetry are unchanged: rho sees
    psi(h_AB)+psi(h_CD) only. The interaction layer is not removed to
    match a parameter count.
    """

    def __init__(self, width=64, k=RHO_K):
        super().__init__()
        self.base = TypedPairMLP(width)
        hidden = self.base.rho.in_features
        self.base.rho = nn.Sequential(nn.Linear(hidden, k), nn.GELU(), nn.Linear(k, 1))

    def forward(self, x):
        return self.base(x).reshape(-1)


class SymmetricQ(nn.Module):
    """Fallback only. Shared q sees both segment vectors; outputs are averaged.

    Used if RepairedRho does not reduce train/dev BCE. Not a named method.
    """

    def __init__(self, width=64, k=48):
        super().__init__()
        base = TypedPairMLP(width)
        self.phi = base.phi
        self.psi = base.psi
        hidden = base.rho.in_features
        self.q = nn.Sequential(nn.Linear(2 * hidden, k), nn.GELU(), nn.Linear(k, 1))

    def forward(self, x):
        h = self.phi(x.reshape(-1, 4, 2))
        hab = self.psi(h[:, 0] + h[:, 1])
        hcd = self.psi(h[:, 2] + h[:, 3])
        first = self.q(torch.cat([hab, hcd], dim=-1))
        second = self.q(torch.cat([hcd, hab], dim=-1))
        return (0.5 * (first + second)).reshape(-1)


def raw_hidden_matching(target_params, feat=32, in_dim=8):
    """Smallest hidden width whose CoordMLP has at least target_params."""
    for hidden in range(64, 256):
        model = CoordMLP(hidden, feat, in_dim=in_dim)
        n = count_params(model)
        if n >= target_params:
            return hidden, n
    raise RuntimeError("no matching raw width")


def macs_per_example(model, in_dim):
    counted = [0]

    def hook(mod, inputs, _out):
        batch = inputs[0].reshape(-1, inputs[0].shape[-1]).shape[0]
        counted[0] += int(batch) * mod.in_features * mod.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, in_dim))
    for handle in handles:
        handle.remove()
    return counted[0] / 2


def build_model(arm, raw_hidden):
    if arm == "raw":
        model, in_dim = CoordMLP(64, 32, in_dim=8), 8
    elif arm == "raw_matched":
        model, in_dim = CoordMLP(raw_hidden, 32, in_dim=8), 8
    elif arm == "additive":
        model, in_dim = TypedPairMLP(64), 8
    elif arm == "repaired_rho":
        model, in_dim = RepairedRho(64, RHO_K), 8
    elif arm == "symmetric_q":
        model, in_dim = SymmetricQ(64, 48), 8
    elif arm == "sixdist":
        model, in_dim = CoordMLP(64, 32, in_dim=6), 6
    elif arm in ("rich8_unsorted", "rich8_sorted"):
        model, in_dim = CoordMLP(64, 32, in_dim=8), 8
    else:
        raise KeyError(arm)
    return model, in_dim


def featurize(arm, x, mu, sd):
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    if arm in ("raw", "raw_matched", "additive", "repaired_rho", "symmetric_q"):
        z = (x - mu.reshape(1, 1, 2)) / sd.reshape(1, 1, 2)
        return z.reshape(len(x), 8).astype(np.float32)
    if arm == "sixdist":
        f = six_ordered(x)
    elif arm == "rich8_unsorted":
        f = rich8_unsorted(x)
    elif arm == "rich8_sorted":
        f = rich8_sorted(x)
    else:
        raise KeyError(arm)
    return ((f - mu) / sd).astype(np.float32)


def fit_stats(arm, x):
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    if arm in ("raw", "raw_matched", "additive", "repaired_rho", "symmetric_q"):
        flat = x.reshape(-1, 2)
        return flat.mean(0), flat.std(0) + 1e-8, "shared_xy"
    if arm == "sixdist":
        f = six_ordered(x)
    elif arm == "rich8_unsorted":
        f = rich8_unsorted(x)
    else:
        f = rich8_sorted(x)
    return f.mean(0), f.std(0) + 1e-8, "train_feature"


def mine_flips(x, y, seed, cap=12):
    """O01 train-flip contract: candidates_for_scene, d03 oracle, cap 12.

    This is supervision, not the claim denominator.
    """
    rng = np.random.default_rng(seed)
    states, labels, parents = [], [], []
    calls = 0
    for i in range(len(x)):
        n = 0
        for edit, _fam in candidates_for_scene(x[i], rng):
            calls += 1
            judged = flip_oracle(x[i] + edit)
            if judged is not None and judged[0] != int(y[i]):
                states.append((x[i] + edit).astype(np.float32))
                labels.append(int(judged[0]))
                parents.append(i)
                n += 1
                if n == cap:
                    break
    if not states:
        return {
            "states": np.zeros((0, 4, 2), np.float32),
            "labels": np.zeros(0, np.int64),
            "parents": np.zeros(0, np.int64),
            "oracle_calls": calls,
        }
    return {
        "states": np.stack(states).astype(np.float32),
        "labels": np.asarray(labels, np.int64),
        "parents": np.asarray(parents, np.int64),
        "oracle_calls": calls,
    }


def mine_quartets(x, y, seed):
    """One row per unordered atomic pair. build_bank stored two directed paths."""
    rng = np.random.default_rng(seed)
    rows = []
    calls = 0
    per_parent = []
    for i in range(len(x)):
        parent = np.asarray(x[i], np.float64)
        try:
            y0, m0, a0 = oracle_at(parent)
            calls += 1
        except ValueError:
            per_parent.append(0)
            continue
        if int(y0) != int(y[i]):
            raise RuntimeError("generator label disagrees with oracle_at")
        atomic = []
        for edit, fam in atomic_edits(parent, rng):
            try:
                lab, margin, amb = oracle_at(parent + edit)
                calls += 1
            except ValueError:
                calls += 1
                continue
            atomic.append((np.asarray(edit, np.float64), fam, int(lab), float(margin), bool(amb)))
        n = 0
        for a in range(len(atomic)):
            for b in range(a + 1, len(atomic)):
                ea, fa, ya, ma, aa = atomic[a]
                eb, fb, yb, mb, ab = atomic[b]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yab, mab, aab = oracle_at(parent + ea + eb)
                    calls += 1
                except ValueError:
                    calls += 1
                    continue
                if not is_official_quartet(y0, ya, yb, int(yab)):
                    continue
                rows.append({
                    "parent": i,
                    "ea": ea.astype(np.float32),
                    "eb": eb.astype(np.float32),
                    "fam_a": fa,
                    "fam_b": fb,
                    "y0": int(y0),
                    "yab": int(yab),
                    "m0": float(m0),
                    "ma": float(ma),
                    "mb": float(mb),
                    "mab": float(mab),
                    "ambiguous_any": bool(a0 or aa or ab or aab),
                })
                n += 1
        per_parent.append(n)
    return rows, calls, np.asarray(per_parent, np.int64)


def parent_ci(values, parents, seed=BOOT_SEED, n_boot=BOOT_N):
    """Parent resample, then row mean of the concatenated draw. Seeds are not datasets.

    The bootstrap mean is the count-weighted mean of parent groups. That equals
    concatenating the resampled rows. Group elements of G8 are not inputs here.
    """
    values = np.asarray(values, np.float64)
    parents = np.asarray(parents)
    if len(values) == 0:
        return {"estimate": None, "parent_mean": None, "ci95": None, "n": 0, "n_parents": 0}
    uniq = np.unique(parents)
    sums = np.empty(len(uniq))
    counts = np.empty(len(uniq))
    for i, parent in enumerate(uniq):
        group = values[parents == parent]
        sums[i] = group.sum()
        counts[i] = group.size
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_counts = np.zeros((n_boot, len(uniq)), np.int16)
    for i in range(n_boot):
        boot_counts[i] = np.bincount(draw[i], minlength=len(uniq))
    boots = (boot_counts @ sums) / (boot_counts @ counts)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "parent_mean": float((sums / counts).mean()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(values)),
        "n_parents": int(len(uniq)),
    }


def generate_scenes(n, seed):
    pos, lab, mar = make_relational_scenes(n, seed=seed, min_margin=0.02)
    return pos.astype(np.float64), lab.astype(np.int64), mar.astype(np.float64)


def build_bank():
    ART.mkdir(parents=True, exist_ok=True)
    manifest_path = ART / "bank_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to rebuild locked bank {manifest_path}")
    train = np.load(TRAIN_PATH)
    xtr = train["positions"].astype(np.float64)
    ytr = train["labels"].astype(np.int64)
    pos, lab, mar = generate_scenes(DEV_PARENTS + TEST_PARENTS, SCENE_SEED)
    chunks = [{"seed": SCENE_SEED, "n": int(len(pos)), "role": "initial"}]
    dist = min_linf(pos, xtr)
    far = dist >= DEDUP_LINF
    pos, lab, mar, dist = pos[far], lab[far], mar[far], dist[far]
    dropped_train = int((~far).sum())
    keep, dropped_within = greedy_dedup(pos, DEDUP_LINF)
    pos, lab, mar, dist = pos[keep], lab[keep], mar[keep], dist[keep]
    extra_i = 0
    while len(pos) < DEV_PARENTS + TEST_PARENTS:
        extra_i += 1
        seed = EXTRA_SEED0 + extra_i - 1
        more, my, mm = generate_scenes(EXTRA_CHUNK, seed)
        chunks.append({"seed": seed, "n": int(len(more)), "role": "topup_before_split"})
        dmore = min_linf(more, xtr)
        far = dmore >= DEDUP_LINF
        dropped_train += int((~far).sum())
        more, my, mm, dmore = more[far], my[far], mm[far], dmore[far]
        if len(pos):
            near = min_linf(more, pos) < DEDUP_LINF
            dropped_within += int(near.sum())
            more, my, mm, dmore = more[~near], my[~near], mm[~near], dmore[~near]
        k2, d2 = greedy_dedup(more, DEDUP_LINF)
        dropped_within += d2
        pos = np.concatenate([pos, more[k2]])
        lab = np.concatenate([lab, my[k2]])
        mar = np.concatenate([mar, mm[k2]])
        dist = np.concatenate([dist, dmore[k2]])
        if extra_i > 8:
            raise RuntimeError("could not fill parent pool after dedup")
    dev_x, dev_y, dev_m = pos[:DEV_PARENTS], lab[:DEV_PARENTS], mar[:DEV_PARENTS]
    te_x, te_y, te_m, te_d = pos[DEV_PARENTS:DEV_PARENTS + TEST_PARENTS], lab[DEV_PARENTS:DEV_PARENTS + TEST_PARENTS], mar[DEV_PARENTS:DEV_PARENTS + TEST_PARENTS], dist[DEV_PARENTS:DEV_PARENTS + TEST_PARENTS]
    rows, calls, per_parent = mine_quartets(te_x, te_y, EDIT_SEED)
    eligible = int((per_parent > 0).sum())
    added = 0
    while eligible < MIN_ELIGIBLE:
        added += 1
        seed = EXTRA_SEED0 + 100 + added
        more, my, mm = generate_scenes(EXTRA_CHUNK, seed)
        chunks.append({"seed": seed, "n": int(len(more)), "role": "yield_topup_before_scores"})
        dmore = min_linf(more, xtr)
        far = dmore >= DEDUP_LINF
        dropped_train += int((~far).sum())
        more, my, mm, dmore = more[far], my[far], mm[far], dmore[far]
        held = np.concatenate([dev_x, te_x])
        near = min_linf(more, held) < DEDUP_LINF
        dropped_within += int(near.sum())
        more, my, mm, dmore = more[~near], my[~near], mm[~near], dmore[~near]
        k2, d2 = greedy_dedup(more, DEDUP_LINF)
        dropped_within += d2
        more, my, mm, dmore = more[k2], my[k2], mm[k2], dmore[k2]
        base = len(te_x)
        rows2, calls2, per2 = mine_quartets(more, my, EDIT_SEED + added)
        for row in rows2:
            row["parent"] += base
        rows.extend(rows2)
        calls += calls2
        te_x = np.concatenate([te_x, more])
        te_y = np.concatenate([te_y, my])
        te_m = np.concatenate([te_m, mm])
        te_d = np.concatenate([te_d, dmore])
        per_parent = np.concatenate([per_parent, per2])
        eligible = int((per_parent > 0).sum())
        print(json.dumps({"yield_topup": added, "drawn": int(len(te_x)), "quartets": len(rows), "eligible": eligible}), flush=True)
        if added > 6:
            break
    if not rows:
        raise RuntimeError("official quartet yield is zero")
    parents = np.array([r["parent"] for r in rows], np.int64)
    ea = np.stack([r["ea"] for r in rows])
    eb = np.stack([r["eb"] for r in rows])
    y0 = np.array([r["y0"] for r in rows], np.int64)
    yab = np.array([r["yab"] for r in rows], np.int64)
    margins = np.stack([np.array([r["m0"], r["ma"], r["mb"], r["mab"]]) for r in rows])
    fam_a = np.array([r["fam_a"] for r in rows])
    fam_b = np.array([r["fam_b"] for r in rows])
    ambiguous = np.array([r["ambiguous_any"] for r in rows])
    px = te_x[parents]
    states = np.stack([px, px + ea, px + eb, px + ea + eb]).astype(np.float32)
    # states axis 0 is P,A,B,AB. Distance of sampled parents, and of edited states, to train.
    state_dist = min_linf(states.reshape(-1, 4, 2), xtr)
    bank_path = ART / "bank_quartets.npz"
    np.savez_compressed(
        bank_path,
        parent_x=te_x.astype(np.float32),
        parent_y=te_y,
        parent_margin=te_m.astype(np.float32),
        parent_train_linf=te_d.astype(np.float32),
        quartets_per_parent=per_parent,
        parent_index=parents,
        ea=ea.astype(np.float32),
        eb=eb.astype(np.float32),
        y0=y0,
        yab=yab,
        margins=margins.astype(np.float32),
        fam_a=fam_a,
        fam_b=fam_b,
        ambiguous_any=ambiguous,
        states=states,
    )
    dev = mine_flips(dev_x, dev_y, DEV_FLIP_SEED)
    np.savez_compressed(
        ART / "dev_single.npz",
        positions=dev_x.astype(np.float32),
        labels=dev_y,
        margins=dev_m.astype(np.float32),
        single_states=dev["states"],
        single_labels=dev["labels"],
        single_parents=dev["parents"],
        oracle_calls=np.array([dev["oracle_calls"]]),
    )
    flips = mine_flips(xtr, ytr, TRAIN_FLIP_SEED)
    np.savez_compressed(ART / "train_flips.npz", **{k: v for k, v in flips.items()})
    within = min_linf(te_x, dev_x)
    manifest = {
        "claim_denominator": "unique unordered atomic pairs, not build_bank directed-path doubling and not mine_eval",
        "quartet_rule": "two label-preserving atomic_edits whose sum flips the label; ValueError skipped; ambiguous flag recorded, not used as a drop rule",
        "atomic_edits": "experiments/last15h/shared/paths.py:atomic_edits default radius 0.10",
        "generator": "core.relations.make_relational_scenes min_margin=0.02",
        "scene_seed_initial": SCENE_SEED,
        "edit_seed": EDIT_SEED,
        "chunks": chunks,
        "dedup": "raw coordinates, L-inf, not integer ids",
        "dedup_linf": DEDUP_LINF,
        "dropped_near_train": dropped_train,
        "dropped_within_pool": dropped_within,
        "train_path": str(TRAIN_PATH.relative_to(ROOT)),
        "train_sha256": sha256_file(TRAIN_PATH),
        "n_train": int(len(xtr)),
        "dev_parents": int(len(dev_x)),
        "test_parents_drawn": int(len(te_x)),
        "n_quartets": int(len(rows)),
        "n_eligible_parents": eligible,
        "yield_quartets_per_drawn": float(len(rows) / len(te_x)),
        "min_parent_train_linf": float(te_d.min()),
        "min_dev_train_linf": float(min_linf(dev_x, xtr).min()),
        "min_test_to_dev_linf": float(within.min()),
        "min_quartet_state_train_linf": float(state_dist.min()),
        "ambiguous_quartets": int(ambiguous.sum()),
        "low_margin_quartets_min_ma_mb_mab_lt_0.02": int((margins[:, 1:].min(1) < MARGIN_SLICE).sum()),
        "atomic_margin_quantiles": {
            "mA": [float(v) for v in np.quantile(margins[:, 1], [0.1, 0.5, 0.9])],
            "mB": [float(v) for v in np.quantile(margins[:, 2], [0.1, 0.5, 0.9])],
            "mAB": [float(v) for v in np.quantile(margins[:, 3], [0.1, 0.5, 0.9])],
        },
        "margin_slice_note": "subset min(mA,mB,mAB)>=0.02 is a pre-stated descriptor, not a replacement denominator",
        "oracle_calls_quartets": int(calls),
        "dev_flip_oracle_calls": int(dev["oracle_calls"]),
        "train_flip_oracle_calls": int(flips["oracle_calls"]),
        "n_train_flips": int(len(flips["labels"])),
        "n_dev_flips": int(len(dev["labels"])),
        "threads": THREADS,
        "sealed_pools": "not read",
        "dev512": "not read; historical only; not confirmation",
        "yield_probe_not_denominator": {
            "n": 200, "seed": SCENE_SEED, "quartets": 92, "eligible_parents": 33,
            "low_margin_quartets": 82,
            "use": "chose TEST_PARENTS=4096 before scores because 200 parents yielded 33 eligible",
        },
        "selection_rule": "argmin pooled static+single dev BCE over Adam lr 0.01/0.003/0.001; ties take the smaller lr; never test J",
        "additive_budget": "same 3 Adam lrs, then stop; do not add optimizers if the negative control fails",
        "optimizer_fail_rule": "if repaired_rho mean train BCE stays above 0.60 in a majority of seed x supervision cells, the first repair did not optimize; train symmetric_q and use it as the repaired arm. Dev BCE is not this trigger.",
        "continuation_rule": "after 3 seeds, add seeds 71 and 83 only for a pre-stated contrast whose seed-wise parent intervals do not all lie strictly on the same side of 0",
        "flow_baseline": "raw clean, same seed, dev-selected lr; within-arm clean-to-flip is secondary and labeled",
        "decision_threshold": "none; logit>0 is the training decision rule, not a tuned test threshold",
        "bank_sha256": sha256_file(bank_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "n_quartets", "n_eligible_parents", "test_parents_drawn", "min_parent_train_linf",
        "low_margin_quartets_min_ma_mb_mab_lt_0.02")}, indent=2), flush=True)


def receipt_path(arm, seed, sup, lr):
    return ART / "runs" / f"{arm}_s{seed}_{sup}_lr{lr}" / "receipt.json"


def train_cell(arm, seed, sup, lr, raw_hidden, xtr, ytr, flips, dev, force=False):
    dest = ART / "runs" / f"{arm}_s{seed}_{sup}_lr{lr}"
    rec = dest / "receipt.json"
    if rec.exists() and not force:
        return json.loads(rec.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    mu, sd, kind = fit_stats(arm, xtr)
    xc = torch.tensor(featurize(arm, xtr, mu, sd))
    yc = torch.tensor(ytr, dtype=torch.float32)
    if sup == "flip":
        xf = torch.tensor(featurize(arm, flips["states"], mu, sd))
        yf = torch.tensor(flips["labels"], dtype=torch.float32)
    else:
        xf = yf = None
    d_clean = featurize(arm, dev["positions"], mu, sd)
    d_flip = featurize(arm, dev["single_states"], mu, sd)
    xd = torch.tensor(np.concatenate([d_clean, d_flip]))
    yd = torch.tensor(np.concatenate([dev["labels"], dev["single_labels"]]), dtype=torch.float32)
    torch.manual_seed(seed)
    model, in_dim = build_model(arm, raw_hidden)
    init = sha256_bytes(b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values()))
    macs = macs_per_example(model, in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    curve = []
    t0 = time.time()
    n_forward = 0
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = bce(model(xc), yc)
        n_forward += len(xc)
        if xf is not None:
            loss = loss + bce(model(xf), yf)
            n_forward += len(xf)
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    seconds = time.time() - t0
    model.eval()
    with torch.no_grad():
        dev_bce = float(bce(model(xd), yd))
        dev_clean = float(bce(model(torch.tensor(d_clean)), torch.tensor(dev["labels"], dtype=torch.float32)))
        dev_flip = float(bce(model(torch.tensor(d_flip)), torch.tensor(dev["single_labels"], dtype=torch.float32)))
        train_error = float(((model(xc) > 0) != yc.bool()).float().mean())
    ckpt = dest / "model.pt"
    torch.save({
        "state": model.state_dict(), "arm": arm, "seed": seed, "supervision": sup, "lr": lr,
        "mu": mu, "sd": sd, "stats_kind": kind, "raw_hidden": raw_hidden, "in_dim": in_dim,
    }, ckpt)
    row = {
        "arm": arm, "seed": int(seed), "supervision": sup, "lr": lr,
        "dev_bce": dev_bce, "dev_bce_clean": dev_clean, "dev_bce_flip": dev_flip,
        "train_error": train_error, "final_loss": curve[-1], "epochs": EPOCHS,
        "n_parameters": count_params(model), "macs_per_example": macs,
        "train_state_forwards": n_forward,
        "train_forward_macs": macs * n_forward,
        "seconds": seconds, "threads": THREADS, "init_sha256": init,
        "checkpoint_sha256": sha256_file(ckpt), "stats_kind": kind,
        "n_dev_clean": int(len(dev["labels"])), "n_dev_flip": int(len(dev["single_labels"])),
        "n_train": int(len(ytr)), "n_train_flip": int(len(flips["labels"])) if sup == "flip" else 0,
    }
    rec.write_text(json.dumps(row, indent=2) + "\n")
    (dest / "loss.json").write_text(json.dumps(curve))
    print(json.dumps({k: row[k] for k in ("arm", "seed", "supervision", "lr", "dev_bce", "final_loss", "seconds")}), flush=True)
    return row


def load_train_dev():
    train = np.load(TRAIN_PATH)
    dev = np.load(ART / "dev_single.npz")
    flips = np.load(ART / "train_flips.npz")
    return train, dev, flips


def train_grid(arms, seeds):
    train, dev, flips = load_train_dev()
    xtr = train["positions"].astype(np.float32)
    ytr = train["labels"].astype(np.int64)
    repaired = RepairedRho(64, RHO_K)
    raw_hidden, raw_n = raw_hidden_matching(count_params(repaired))
    (ART / "capacity.json").write_text(json.dumps({
        "repaired_rho_params": count_params(repaired),
        "additive_params": count_params(TypedPairMLP(64)),
        "raw_params": count_params(CoordMLP(64, 32, in_dim=8)),
        "raw_matched_hidden": raw_hidden,
        "raw_matched_params": raw_n,
        "note": "raw_matched is a capacity control, not a new method; the interaction layer was not removed",
    }, indent=2) + "\n")
    rows = []
    for arm in arms:
        for seed in seeds:
            for sup in ("clean", "flip"):
                for lr in LRS:
                    rows.append(train_cell(arm, seed, sup, lr, raw_hidden, xtr, ytr, flips, dev))
    return rows


def select(arms, seeds, merge=False):
    chosen = []
    for arm in arms:
        for seed in seeds:
            for sup in ("clean", "flip"):
                cands = []
                for lr in LRS:
                    rec = receipt_path(arm, seed, sup, lr)
                    if not rec.exists():
                        raise FileNotFoundError(rec)
                    cands.append(json.loads(rec.read_text()))
                cands.sort(key=lambda r: (r["dev_bce"], r["lr"]))
                pick = dict(cands[0])
                pick["selection"] = "min pooled static+single dev BCE; smaller lr wins ties; test J not read"
                chosen.append(pick)
    path = ART / "selection.json"
    if merge and path.exists():
        old = json.loads(path.read_text())
        keyed = {(r["arm"], r["seed"], r["supervision"]): r for r in old}
        for row in chosen:
            keyed[(row["arm"], row["seed"], row["supervision"])] = row
        chosen = list(keyed.values())
    path.write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def rho_failed(seeds):
    """Train-loss failure only. A high dev BCE is not a reason to switch repairs."""
    bad = 0
    total = 0
    cells = []
    for seed in seeds:
        for sup in ("clean", "flip"):
            total += 1
            losses = []
            for lr in LRS:
                row = json.loads(receipt_path("repaired_rho", seed, sup, lr).read_text())
                train_mean = row["final_loss"] if sup == "clean" else row["final_loss"] / 2
                losses.append(train_mean)
            best = min(losses)
            failed = best > OPT_FAIL_BCE
            bad += int(failed)
            cells.append({"seed": int(seed), "supervision": sup, "best_mean_train_bce": best, "failed": failed})
    return bad * 2 >= total, {
        "bad_cells": bad,
        "cells": total,
        "detail": cells,
        "rule": "majority of cells have best mean train BCE > 0.60; flip loss is the sum of two means so it is halved here",
    }


def predict(model, arm, x, mu, sd):
    feat = torch.tensor(featurize(arm, x, mu, sd))
    out = []
    with torch.no_grad():
        for i in range(0, len(feat), 2048):
            out.append(model(feat[i:i + 2048]).numpy())
    return np.concatenate(out)


def score_quartets(model, arm, mu, sd, bank):
    states = bank["states"]  # (4, N, 4, 2)
    logits = np.stack([predict(model, arm, states[i], mu, sd) for i in range(4)])
    y0 = bank["y0"]
    yab = bank["yab"]
    labels = np.stack([y0, y0, y0, yab])
    ok = (logits > 0) == (labels == 1)
    return logits, ok


def load_selected_model(row, raw_hidden):
    ckpt = torch.load(ART / "runs" / f"{row['arm']}_s{row['seed']}_{row['supervision']}_lr{row['lr']}" / "model.pt", map_location="cpu", weights_only=False)
    model, _ = build_model(row["arm"], raw_hidden)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["mu"], ckpt["sd"]


def flow_from(base_ok, cand_ok, parents):
    """base_ok/cand_ok columns are A, B, AB."""
    b110 = base_ok[:, 0] & base_ok[:, 1] & ~base_ok[:, 2]
    b111 = base_ok.all(1)
    if b110.any():
        ab = cand_ok[b110, 2]
        full = cand_ok[b110].all(1)
        mig = cand_ok[b110, 2] & ~cand_ok[b110, :2].all(1)
        endpoint = parent_ci(ab.astype(float), parents[b110])
        rfull = parent_ci(full.astype(float), parents[b110])
        migration = parent_ci(mig.astype(float), parents[b110])
    else:
        endpoint = rfull = migration = {"estimate": None, "n": 0, "n_parents": 0, "ci95": None}
    if b111.any():
        reg = parent_ci((~cand_ok[b111].all(1)).astype(float), parents[b111])
    else:
        reg = {"estimate": None, "n": 0, "n_parents": 0, "ci95": None}
    return {
        "baseline110_n": int(b110.sum()),
        "baseline111_n": int(b111.sum()),
        "R_endpoint": endpoint,
        "R_full": rfull,
        "M": migration,
        "regression111": reg,
        "identity_residual": None if not b110.any() else float(ab.mean() - full.mean() - mig.mean()),
    }


def arm_metrics(ok, parents, margins):
    # ok columns P,A,B,AB
    j3 = ok[1] & ok[2] & ok[3]
    j4 = ok[0] & j3
    atomic = ok[1] & ok[2]
    hi = margins[:, 1:].min(1) >= MARGIN_SLICE
    out = {
        "J3": parent_ci(j3.astype(float), parents),
        "J4": parent_ci(j4.astype(float), parents),
        "A_accuracy": parent_ci(ok[1].astype(float), parents),
        "B_accuracy": parent_ci(ok[2].astype(float), parents),
        "AB_accuracy": parent_ci(ok[3].astype(float), parents),
        "P_accuracy": parent_ci(ok[0].astype(float), parents),
        "atomic_joint": parent_ci(atomic.astype(float), parents),
        "CCM_denominator": int(atomic.sum()),
        "CCM": float(j3[atomic].mean()) if atomic.any() else None,
        "J3_margin_ge_0.02": parent_ci(j3[hi].astype(float), parents[hi]) if hi.any() else None,
        "margin_slice_n": int(hi.sum()),
        "margin_slice_parents": int(len(np.unique(parents[hi]))) if hi.any() else 0,
    }
    return out, j3


def contrast_row(delta, parents):
    return parent_ci(delta.astype(float), parents)


def intervals_same_side(rows):
    signs = []
    for row in rows:
        lo, hi = row["ci95"]
        if lo > 0:
            signs.append(1)
        elif hi < 0:
            signs.append(-1)
        else:
            signs.append(0)
    return len(signs) > 0 and all(s == signs[0] and s != 0 for s in signs)


def evaluate(seeds, arms, tag):
    bank = np.load(ART / "bank_quartets.npz", allow_pickle=True)
    parents = bank["parent_index"]
    margins = bank["margins"]
    cap = json.loads((ART / "capacity.json").read_text())
    raw_hidden = cap["raw_matched_hidden"]
    selection = json.loads((ART / "selection.json").read_text())
    selected = [r for r in selection if r["seed"] in seeds and r["arm"] in arms]
    scores = {}
    t0 = time.time()
    n_states = int(np.prod(bank["states"].shape[:2]))
    for row in selected:
        model, mu, sd = load_selected_model(row, raw_hidden)
        infer_t = time.time()
        logits, ok = score_quartets(model, row["arm"], mu, sd, bank)
        infer_s = time.time() - infer_t
        key = (row["arm"], row["seed"], row["supervision"])
        scores[key] = ok
        metrics, _ = arm_metrics(ok, parents, margins)
        metrics.update({
            "arm": row["arm"], "seed": row["seed"], "supervision": row["supervision"], "lr": row["lr"],
            "n_parameters": row["n_parameters"], "train_state_forwards": row["train_state_forwards"],
            "train_forward_macs": row["train_forward_macs"], "train_seconds": row["seconds"],
            "inference_states": n_states, "inference_seconds": infer_s,
            "inference_forward_macs": row["macs_per_example"] * n_states,
            "macs_per_example": row["macs_per_example"],
            "checkpoint": f"artifacts/e832_focus/structure/runs/{row['arm']}_s{row['seed']}_{row['supervision']}_lr{row['lr']}/model.pt",
        })
        # construction check: one forward per state, not 8 group replicates
        swing = []
        sample = bank["states"][1, :64]
        base = predict(model, row["arm"], sample, mu, sd)
        for perm in G8[1:]:
            swing.append(np.max(np.abs(predict(model, row["arm"], apply_perm(sample, perm), mu, sd) - base)))
        metrics["g8_logit_swing_max_on_64_A_states"] = float(max(swing) if swing else 0.0)
        metrics["g8_swing_note"] = "diagnostic on 64 states; group elements are not replicates; zero swing is by construction when the model and features are strictly invariant"
        np.savez_compressed(ART / "runs" / f"{row['arm']}_s{row['seed']}_{row['supervision']}_lr{row['lr']}" / "quartet_correct.npz", ok=ok, logits=logits)
        scores[key + ("metrics",)] = metrics
    table = [scores[k + ("metrics",)] for k in scores if k[-1] != "metrics" and len(k) == 3]
    # fix key handling
    table = []
    okmap = {}
    for key, value in list(scores.items()):
        if len(key) == 3:
            okmap[key] = value
    # recompute table from saved metrics in the loop properly
    table = []
    for row in selected:
        key = (row["arm"], row["seed"], row["supervision"])
        table.append(scores[key + ("metrics",)])
    contrasts = []
    for seed in seeds:
        for sup in ("clean", "flip"):
            def grab(arm, _seed=seed, _sup=sup):
                return okmap[(arm, _seed, _sup)]
            pairs = [("repaired_rho_minus_additive", "repaired_rho", "additive")]
            if ("symmetric_q", seed, sup) in okmap:
                pairs.append(("symmetric_q_minus_additive", "symmetric_q", "additive"))
            pairs.extend([
                ("sorted_minus_unsorted", "rich8_sorted", "rich8_unsorted"),
                ("unsorted_minus_sixdist", "rich8_unsorted", "sixdist"),
                ("sorted_minus_sixdist", "rich8_sorted", "sixdist"),
                ("repaired_rho_minus_raw", "repaired_rho", "raw"),
                ("sixdist_minus_raw", "sixdist", "raw"),
            ])
            for name, left, right in pairs:
                if (left, seed, sup) not in okmap or (right, seed, sup) not in okmap:
                    continue
                jl = okmap[(left, seed, sup)][1] & okmap[(left, seed, sup)][2] & okmap[(left, seed, sup)][3]
                jr = okmap[(right, seed, sup)][1] & okmap[(right, seed, sup)][2] & okmap[(right, seed, sup)][3]
                contrasts.append({
                    "contrast": name, "seed": int(seed), "supervision": sup,
                    "delta_J3": contrast_row(jl.astype(float) - jr.astype(float), parents),
                    "path": "artifacts/e832_focus/structure/bank_quartets.npz",
                })
            if ("raw", seed, "clean") in okmap:
                raw_clean = okmap[("raw", seed, "clean")]
                base = np.stack([raw_clean[1], raw_clean[2], raw_clean[3]], axis=1)
            else:
                base = None
            for arm in arms:
                if (arm, seed, sup) not in okmap:
                    continue
                cand_ok = okmap[(arm, seed, sup)]
                cand = np.stack([cand_ok[1], cand_ok[2], cand_ok[3]], axis=1)
                if base is not None:
                    flow = flow_from(base, cand, parents)
                    flow.update({"source": "raw_clean_same_seed", "arm": arm, "seed": int(seed), "supervision": sup})
                    contrasts.append({"flow": flow})
                if sup == "flip" and (arm, seed, "clean") in okmap:
                    own = okmap[(arm, seed, "clean")]
                    own_base = np.stack([own[1], own[2], own[3]], axis=1)
                    own_flow = flow_from(own_base, cand, parents)
                    own_flow.update({"source": "within_arm_clean", "arm": arm, "seed": int(seed), "supervision": "flip"})
                    contrasts.append({"flow": own_flow})
    # representation x supervision interaction on paired quartets
    interactions = []
    for seed in seeds:
        for left, right, name in (
            ("repaired_rho", "additive", "I_repaired_minus_additive"),
            ("rich8_sorted", "rich8_unsorted", "I_sorted_minus_unsorted"),
            ("rich8_unsorted", "sixdist", "I_unsorted_minus_sixdist"),
        ):
            if any((a, seed, s) not in okmap for a in (left, right) for s in ("clean", "flip")):
                continue
            def j(arm, sup, _seed=seed):
                ok = okmap[(arm, _seed, sup)]
                return (ok[1] & ok[2] & ok[3]).astype(float)
            delta = (j(left, "flip") - j(left, "clean")) - (j(right, "flip") - j(right, "clean"))
            interactions.append({
                "name": name, "seed": int(seed),
                "I_row_points": contrast_row(delta, parents),
                "marginal_flip_minus_clean_left": contrast_row(j(left, "flip") - j(left, "clean"), parents),
                "marginal_flip_minus_clean_right": contrast_row(j(right, "flip") - j(right, "clean"), parents),
                "note": "I is a paired percentage-point interaction on the same E rows; not a new threshold",
            })
    payload = {
        "tag": tag,
        "threads": THREADS,
        "bank": "artifacts/e832_focus/structure/bank_quartets.npz",
        "bank_sha256": sha256_file(ART / "bank_quartets.npz"),
        "n_quartets": int(len(parents)),
        "n_parents": int(len(np.unique(parents))),
        "eval_seconds": time.time() - t0,
        "arms": table,
        "contrasts": [c for c in contrasts if "contrast" in c],
        "flows": [c["flow"] for c in contrasts if "flow" in c],
        "interactions": interactions,
        "decision_rule": "logit > 0",
        "group_elements_are_not_replicates": True,
    }
    out = ART / f"metrics_{tag}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(out), "n": payload["n_quartets"], "arms": len(table)}), flush=True)
    return payload


def close_contrasts(payload):
    names = ["sorted_minus_unsorted"]
    if any(c["contrast"] == "symmetric_q_minus_additive" for c in payload["contrasts"]):
        names.append("symmetric_q_minus_additive")
    else:
        names.append("repaired_rho_minus_additive")
    close = []
    for name in names:
        for sup in ("clean", "flip"):
            rows = [c["delta_J3"] for c in payload["contrasts"] if c["contrast"] == name and c["supervision"] == sup]
            if len(rows) >= 3 and not intervals_same_side(rows):
                close.append({"contrast": name, "supervision": sup, "n_seed_intervals": len(rows)})
    return close


def assert_feature_contract():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4, 2)).astype(np.float32)
    unsorted = rich8_unsorted(x)
    manual = np.concatenate([np.sort(unsorted[:, 0:2], 1), np.sort(unsorted[:, 2:6], 1), unsorted[:, 6:8]], 1)
    if not np.allclose(manual, rich8_sorted(x)):
        raise RuntimeError("rich8_sorted is not the existing G8 sort of rich8_unsorted")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["bank", "train", "select", "eval", "all"], required=True)
    a = p.parse_args()
    assert_feature_contract()
    if a.stage in ("bank", "all"):
        build_bank()
    if a.stage == "bank":
        return
    arms = ["raw", "raw_matched", "additive", "repaired_rho", "sixdist", "rich8_unsorted", "rich8_sorted"]
    if a.stage in ("train", "all"):
        train_grid(arms, SEEDS)
        failed, info = rho_failed(SEEDS)
        (ART / "rho_optimization.json").write_text(json.dumps({"failed": failed, **info}, indent=2) + "\n")
        if failed:
            train_grid(["symmetric_q"], SEEDS)
            arms = arms + ["symmetric_q"]
    if a.stage in ("select", "all"):
        have = ["raw", "raw_matched", "additive", "repaired_rho", "sixdist", "rich8_unsorted", "rich8_sorted"]
        if (ART / "rho_optimization.json").exists() and json.loads((ART / "rho_optimization.json").read_text())["failed"]:
            have.append("symmetric_q")
        select(have, SEEDS)
    if a.stage in ("eval", "all"):
        have = json.loads((ART / "selection.json").read_text())
        arms = sorted({r["arm"] for r in have})
        payload = evaluate(list(SEEDS), arms, "3seed")
        close = close_contrasts(payload)
        (ART / "continuation_decision.json").write_text(json.dumps({
            "rule": "add seeds 71,83 only if a pre-stated contrast has seed-wise parent intervals that are not all strictly on the same side of 0",
            "close": close,
        }, indent=2) + "\n")
        if close:
            extra_arms = set()
            for item in close:
                if "additive" in item["contrast"]:
                    extra_arms.update(["additive", "repaired_rho"])
                    if item["contrast"].startswith("symmetric_q"):
                        extra_arms.add("symmetric_q")
                if item["contrast"] == "sorted_minus_unsorted":
                    extra_arms.update(["rich8_unsorted", "rich8_sorted"])
            train_grid(sorted(extra_arms), EXTRA_SEEDS)
            select(sorted(extra_arms), EXTRA_SEEDS, merge=True)
            merged = json.loads((ART / "selection.json").read_text())
            evaluate(sorted({r["seed"] for r in merged}), sorted({r["arm"] for r in merged}), "continued")
        else:
            (ART / "metrics_continued.json").write_text(json.dumps({"continued": False, "source": "metrics_3seed.json"}) + "\n")


if __name__ == "__main__":
    main()
