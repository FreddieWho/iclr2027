#!/usr/bin/env python3
"""P1-A coordinate: recomputable confidence-vs-update analysis on frozen bank.

Per bank path (oracle single-cross enforced here): start_correct, start_conf
(|logit|), endpoint_correct, arrival_at_new_class, third_class/early/late/
jumpback flags, margins, crossing distance, displacement, family, classes.
Per-sample JSON out. Dual test: (1) conf->current correctness on static
reference (bank singles endpoints); (2) conf->update failure on
start-correct+flipped subset. Models: r04b s11/s23/s47 x clean/flipmine.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import (  # noqa: E402
    oracle_at, scan_linear, locate_oracle_turns, locate_model_turns,
    match_turns)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "p123_upgrade" / "p1"
MODELS = [(s, m, ART / f"r04b_s{s}" / m) for s in (11, 23, 47)
          for m in ("clean", "flipmine")]


def make_predict(model, stats):
    def predict_fn(xs):
        xs = np.asarray(xs, dtype=np.float32)
        with torch.no_grad():
            out = []
            for s in range(0, len(xs), 256):
                out.append(model(preprocess(xs[s:s + 256], stats)).numpy())
        lg = np.concatenate(out).reshape(-1)
        return lg, (lg > 0).astype(int)
    return predict_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    Sx, Se = b["Sx"], b["Se"]
    Smeta = json.loads(str(b["Smeta"]))
    assert len(Qmeta) == len(Qx) and len(Smeta) == len(Sx)
    a.out.mkdir(parents=True, exist_ok=True)
    for seed, mname, mp in MODELS:
        model, stats = load_model(mp)
        model.eval()
        predict_fn = make_predict(model, stats)
        rows = []
        for qi, (x, e) in enumerate(zip(Qx, Qe)):
            x = np.asarray(x, dtype=float)
            e = np.asarray(e, dtype=float)
            try:
                scan = scan_linear(x, e, predict_fn, n_scan=51)
            except Exception:
                continue
            ot = locate_oracle_turns(x, e, scan)
            if len(ot) != 1:
                continue
            mt = locate_model_turns(x, e, scan, predict_fn)
            matched = match_turns(ot, mt)
            m = matched[0]
            sl = int(scan["oracle_label"][0])
            el = int(scan["oracle_label"][-1])
            pl = scan["pred_label"]
            lg = scan["logit"]
            start_ok = bool((lg[0] > 0) == sl)
            end_ok = bool((lg[-1] > 0) == el)
            # arrival at NEW oracle class (not any change)
            new_cls = el
            arrived = bool(any(int(pl[j]) == new_cls for j in range(len(pl))))
            third = bool(any(int(pl[j]) not in (sl, el) for j in range(len(pl)))) if sl != el else False
            status = m["status"]
            lag = m.get("error")
            nm = m["n_model_turns"]
            if status == "missing":
                outcome = "no_cross" if nm == 0 else "wrong_target_multi"
            elif status == "ok":
                outcome = "hit"
            else:
                outcome = "abnormal_" + status
            rows.append({
                "qid": Qmeta[qi]["qid"], "parent": Qmeta[qi]["parent"],
                "ptype": Qmeta[qi]["ptype"], "fam": Qmeta[qi]["fam_A"] + "+" + Qmeta[qi]["fam_B"],
                "start_correct": start_ok, "start_conf": float(abs(lg[0])),
                "endpoint_correct": end_ok, "arrived_new_class": arrived,
                "third_class": third, "outcome": outcome, "lag": lag,
                "t_star": ot[0]["t_star"],
                "start_margin": float(scan["oracle_margin"][0]),
                "end_margin": float(scan["oracle_margin"][-1]),
                "displacement": float(np.linalg.norm(e)),
                "start_lab": sl, "end_lab": el})
        # static reference: bank singles endpoints (unscreened, natural-ish)
        ref_conf, ref_ok = [], []
        for x, e, sm in zip(Sx[::25], Se[::25], Smeta[::25]):
            try:
                y1, _, _ = oracle_at(np.asarray(x, float) + np.asarray(e, float))
            except ValueError:
                continue
            lg1, _ = predict_fn((np.asarray(x, float) + np.asarray(e, float))[None])
            ref_conf.append(float(abs(lg1[0])))
            ref_ok.append(int((lg1[0] > 0) == y1))
        json.dump({"rows": rows,
                   "ref": {"conf": ref_conf, "ok": ref_ok}},
                  open(a.out / ("p1coord_s%d_%s.json" % (seed, mname)), "w"))
        n1 = sum(1 for r in rows if r["start_correct"])
        print("SAW s%d %s: paths=%d start_correct=%d" % (seed, mname, len(rows), n1), flush=True)
    print("DONE p1_coordinate")


if __name__ == "__main__":
    main()
