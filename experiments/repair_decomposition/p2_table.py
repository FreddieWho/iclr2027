#!/usr/bin/env python3
"""P2-B immutable scene x action table (reuses U02 cache frames/feasibility).

49 lib actions per scene in lib order (12544 = 256 x 49 verified). Logits for
s11/s23/s47 x clean/flipmine. Saved once with sha; all P2 analyses read it.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import action_library  # noqa: E402
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
OUT = ROOT / "artifacts" / "p123_upgrade" / "p2"


def main():
    c = np.load(ROOT / "artifacts" / "next6_ef0f7a3" / "u02" / "cache.npz")
    frames, feas, scene_of, cross = c["frames"], c["feas"], c["scene_of"], c["cross"]
    assert len(frames) == 256 * 49 and len(frames) % 256 == 0
    lib = action_library()
    assert len(lib) == 49
    costs = np.array([a["cost"] for a in lib])
    act_idx = np.tile(np.arange(49), 256)
    logits = {}
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            m, s = load_model(ART / ("r04b_s%d" % seed) / mname)
            m.eval()
            with torch.no_grad():
                out = []
                for i in range(0, len(frames), 512):
                    out.append(m(preprocess(frames[i:i + 512].astype(np.float32), s)).numpy())
            logits["s%d_%s" % (seed, mname)] = np.concatenate(out).reshape(-1).astype(np.float64)
            print("SAW s%d %s" % (seed, mname), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "p2_table.npz", frames=frames, feas=feas, scene_of=scene_of,
             cross=cross, act_idx=act_idx, costs=costs,
             **{"lg_" + k: v for k, v in logits.items()})
    h = hashlib.sha256(open(OUT / "p2_table.npz", "rb").read()).hexdigest()
    json.dump({"n_scenes": 256, "n_actions": 49, "models": sorted(logits),
               "source": "u02 cache frames/feas (2026-09-18)", "sha256": h},
              open(OUT / "p2_table.manifest.json", "w"), indent=1)
    print("SAW table sha=%s" % h[:12])


if __name__ == "__main__":
    main()
