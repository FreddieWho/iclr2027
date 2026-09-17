#!/usr/bin/env python3
"""T5R3 adapter probe: slot stability, centroid/zone oracle semantics, inventory."""
import numpy as np
import pandas as pd
from pathlib import Path

VIEW = Path("/home/huyudi/012_conference/iclr2027/artifacts/phase3/task_semantic_repair_v1/data_views")
T5R3 = Path("/home/huyudi/012_conference/iclr2027/artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3")
name = "valid"
raw = np.load(VIEW / f"positions_raw_{name}.npy").astype(float)
team = np.load(VIEW / f"team_slots_{name}.npy").astype(int)
index = pd.read_parquet(VIEW / f"snapshot_index_{name}.parquet")
print("shapes:", raw.shape, team.shape, len(index))
print("raw range: x", raw[..., 0].min(), raw[..., 0].max(), " y", raw[..., 1].min(), raw[..., 1].max())
print("slot values:", np.unique(team), "team sizes per snapshot (unique):",
      np.unique(team.sum(axis=1)))
print("slot index-stable:", bool((team == team[0]).all()),
      "| n distinct slot patterns:", len(np.unique(team, axis=0)))
print("matches:", sorted(index["source_match_id"].astype(str).unique()))
print("J03WQQ present:", bool((index["source_match_id"].astype(str) == "J03WQQ").any()))
# centroid mapping: mean of slot0/slot1 vs stored home/away
m0 = (raw * (team == 0)[..., None]).sum(axis=1) / (team == 0).sum(axis=1, keepdims=True)
m1 = (raw * (team == 1)[..., None]).sum(axis=1) / (team == 1).sum(axis=1, keepdims=True)
stored = index[["home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y"]].to_numpy(float)
e00 = np.abs(m0 - stored[:, :2]).max()
e01 = np.abs(m0 - stored[:, 2:]).max()
print(f"slot0==home maxerr {e00:.6f} | slot0==away maxerr {e01:.6f}")
home_first = e00 < e01
H = stored[:, 0]
z = index["home_field_zone"].astype(str).to_numpy()
print("zone classes:", sorted(set(z)))
for cls in sorted(set(z)):
    print(f"  {cls}: home_cx min/med/max = {H[z==cls].min():.3f}/{np.median(H[z==cls]):.3f}/{H[z==cls].max():.3f} n={(z==cls).sum()}")
print("checkpoints:", sorted(p.name for p in (T5R3 / "models").glob("raw_single_channel*")))
import json
man = json.load(open(T5R3 / "manifest.json"))
print("manifest keys:", list(man.keys())[:10])
