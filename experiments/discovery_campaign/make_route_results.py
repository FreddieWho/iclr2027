#!/usr/bin/env python3
"""Emit schema-conformant route_result.json for R01..R06 (P1-c) + parquet copies (P1-d).

Primary numbers are COMPUTED from data files, never hand-copied.
Conformance self-check: required top-level + primary_result fields.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts" / "discovery_campaign"
R = ROOT / "reports" / "discovery_campaign"
PACK = ROOT / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
COMMIT = "f86dbdf6107bad3d156b250a3c23bd5dcd8bb9d2"
OUT = R / "route_results"
OUT.mkdir(exist_ok=True)


def J(p):
    return json.loads((A / p).read_text())


def check(rec: dict):
    for k in ("route_id", "status", "evidence_level", "primary_result", "artifacts"):
        assert k in rec, k
    for k in ("name", "value", "baseline_value", "denominator", "unit"):
        assert k in rec["primary_result"], k
    assert isinstance(rec["artifacts"], list)


def save(rec: dict):
    check(rec)
    (OUT / f"{rec['route_id']}_result.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(rec["route_id"], "->", rec["primary_result"]["name"],
          rec["primary_result"]["value"], "| status:", rec["status"])


BASE_SPLITS = {"train": "scenes/train_101 (512, seed 101)",
               "eval": "scenes/eval_202 (512, seed 202)",
               "holdout": "scenes/holdout_303 (256, seed 303, R05/pristine)",
               "recheck": "scenes/recheck_{404,505,606} (512 each)"}

# ---------- R01 ----------
rows = list(csv.DictReader(open(A / "r01_round2_bank8" / "risk_by_scene.csv")))
from collections import defaultdict
piv = defaultdict(dict)
for r in rows:
    if r["table"] == "preserving":
        piv[(r["model"], r["epsilon"])][r["law"]] = float(r["risk"])
gap = max(max(d.values()) - min(d.values()) for d in piv.values())
save({"route_id": "R01", "status": "true_negative_bank4_and_bank8",
      "evidence_level": "controlled_bank_two_sizes",
      "new_question": "Does joint dependence move task risk at fixed marginals/energy?",
      "what_changed": "bank 3 modes -> 35 modes; laws still flat",
      "source_commit": COMMIT, "data_splits": BASE_SPLITS,
      "perturbation_contract": "antithetic +/- pairs; plus_prob deviation max 1.1e-16",
      "task_oracle": "core.relations.segment_relation, margin>=0.02",
      "source_selection_or_target_whitebox": "Q_source_* from source scenes only; Q_whitebox reference",
      "n_parent_scenes": 256, "n_legal_pairs": 35,
      "primary_result": {"name": "max_cross_law_preserving_risk_gap",
                         "value": round(float(gap), 4), "baseline_value": 0.0,
                         "denominator": 256, "unit": "rate"},
      "compute": {"forward_calls": 256 * 2 * 35 * 2 * 3 * 3 + 256 * 2 * 3 * 2 * 3 * 3,
                  "note": "exact arithmetic: scenes×modes×arms×dirs×eps×models (bank8+bank4)"},
      "key_counterevidence": "whitebox curve grows with eps but is label-flip contaminated",
      "nearest_baseline": "Q_ref uniform over same bank",
      "next_conceptual_trial": "T5R3 message-passing checkpoints (probe passed)",
      "paper_section_replaced": "none (negative boundary)",
      "artifacts": ["artifacts/discovery_campaign/r01_round1",
                    "artifacts/discovery_campaign/r01_round2_bank8"]})

# ---------- R02 ----------
pools = []
for d in ("r02_mlpA", "r02_mlpB", "r02_mlpC",
          "r02_recheck_404_mlpA", "r02_recheck_404_mlpB", "r02_recheck_404_mlpC",
          "r02_recheck_505_mlpA", "r02_recheck_505_mlpB", "r02_recheck_505_mlpC",
          "r02_recheck_606_mlpA", "r02_recheck_606_mlpB", "r02_recheck_606_mlpC"):
    r = J(f"{d}/R02_result.json")
    pools.append(r)
n_scenes = sum(r["n_scenes"] for r in pools)
n_miss = sum(r["n_model_miss_given_flip"] for r in pools)
from r02_search import candidates_for_scene
ncand = len(candidates_for_scene(np.zeros((4, 2)), np.random.default_rng(0)))
save({"route_id": "R02", "status": "confirmed_four_seeds_three_archs",
      "evidence_level": "oracle_margin_floor_0.03_dual_rates",
      "new_question": "Do real relation flips go unfollowed by model decisions?",
      "what_changed": "recheck 3x512x3 reproduced round 1",
      "source_commit": COMMIT, "data_splits": BASE_SPLITS,
      "task_oracle": "core.relations.segment_relation, margin>=0.03",
      "representation_layer": "penultimate + hidden (kNN worse)",
      "n_parent_scenes": n_scenes,
      "n_attempted": n_scenes,
      "primary_result": {"name": "unconditional_miss_rate_pooled_12_searches",
                         "value": round(n_miss / n_scenes, 4), "baseline_value": 0.0,
                         "denominator": n_scenes, "unit": "rate"},
      "compute": {"oracle_evals": n_scenes * ncand,
                  "note": f"exact: scenes x {ncand} structured candidates"},
      "key_counterevidence": "stealth feat dist >> null: encoder moves, boundary not crossed (R05: same-cluster)",
      "nearest_baseline": "label-preserving null edits at matched input norm",
      "next_conceptual_trial": "none needed; phenomenon confirmed",
      "paper_section_replaced": "abstract sentence 1 candidate",
      "artifacts": ["artifacts/discovery_campaign/r02_mlpA",
                    "artifacts/discovery_campaign/r02_recheck_404_mlpA"]})

# ---------- R03 ----------
trows = list(csv.DictReader(open(A / "r03_transfer" / "transfer_matrix.csv")))
diffs = [float(r["union_coverage"]) for r in trows
         if r["k"] == "8" and r["selection"] == "greedy"]
rnds = [float(r["union_coverage"]) for r in trows
        if r["k"] == "8" and r["selection"] == "random"]
gain = float(np.mean(diffs) - np.mean(rnds))
save({"route_id": "R03", "status": "bounded_secondary",
      "evidence_level": "frozen_source_selection_three_archs",
      "new_question": "Do few source-selected modes dominate target failures?",
      "what_changed": "dedicated mini-run (was R01 byproduct)",
      "source_commit": COMMIT, "data_splits": BASE_SPLITS,
      "perturbation_contract": "bank-8 frozen templates, both-arm oracle validity",
      "task_oracle": "core.relations.segment_relation, margin>=0.02",
      "source_selection_or_target_whitebox": "greedy cover k=1,2,4,8 on source; random control same k",
      "n_parent_scenes": 256 + 256,
      "primary_result": {"name": "union_coverage_gain_greedy_minus_random_k8",
                         "value": round(gain, 4), "baseline_value": 0.0,
                         "denominator": 256, "unit": "rate"},
      "compute": {"oracle_evals": int(J("r03_transfer/R03_result.json")["oracle_evals"])},
      "key_counterevidence": "fixed-weight risk transfer is zero; coverage saturates at k=4",
      "nearest_baseline": "seed-fixed random template sets, same k",
      "next_conceptual_trial": "none (bounded)",
      "paper_section_replaced": "none (supports A-story weakly)",
      "artifacts": ["artifacts/discovery_campaign/r03_transfer"]})

# ---------- R04 ----------
miss_eq, miss_cl, nden = [], [], 0
for sd in (11, 23, 47):
    for m, acc in (("equal", miss_eq), ("clean", miss_cl)):
        r = J(f"r04_eval_s{sd}_{m}/R02_result.json")
        acc.append(r["miss_rate_given_flip"])
    nden += J(f"r04_eval_s11_equal/R02_result.json")["n_scenes"]
save({"route_id": "R04", "status": "confirmed_three_seeds_equal_weight_final",
      "evidence_level": "same_bank_same_budget_minimax_equal_adv_clean",
      "new_question": "Does dependence-coverage training fix follow-rate?",
      "what_changed": "stop-rule fired: minimax==equal==adv, simplified to equal",
      "source_commit": COMMIT, "data_splits": BASE_SPLITS,
      "perturbation_contract": "bank-8 global weights conditioned on frozen legal masks",
      "task_oracle": "oracle arm labels for ALL methods (no privileged labels)",
      "n_parent_scenes": 512,
      "primary_result": {"name": "miss_rate_equal_vs_clean_seed11_23_47",
                         "value": round(float(np.mean(miss_eq)), 4),
                         "baseline_value": round(float(np.mean(miss_cl)), 4),
                         "denominator": nden, "unit": "rate"},
      "compute": {"note": "300 epochs full-batch; per-epoch forwards/seed: clean 512, others 512*(1+70)"},
      "key_counterevidence": "gain is baseline-dependent (seed23: -4pp); preserving risk flat",
      "nearest_baseline": "clean-only + per-scene worst-arm adv + uniform-sign (P0-b pending)",
      "next_conceptual_trial": "cross-arch confirmation (running)",
      "paper_section_replaced": "methods sentence candidate",
      "artifacts": ["artifacts/discovery_campaign/r04_round1",
                    "artifacts/discovery_campaign/r04_seed23",
                    "artifacts/discovery_campaign/r04_seed47"]})

# ---------- R05 ----------
m05 = J("r05_mlpA/R05_result.json")["metrics"]
save({"route_id": "R05", "status": "true_negative_frozen_mlp",
      "evidence_level": "metric_knn_hiddenlayer_three_iterations",
      "new_question": "Is flip info recoverable from frozen features?",
      "what_changed": "metric->kNN->hidden layer, all zero",
      "source_commit": COMMIT, "data_splits": BASE_SPLITS,
      "task_oracle": "R02 flip pairs + oracle-verified nuisance edits",
      "representation_layer": "penultimate (f=32) + hidden (h=64)",
      "distance_metric": "low-rank PSD edit (rank 4) vs euclidean vs random-pair control",
      "n_parent_scenes": 256,
      "primary_result": {"name": "holdout_pairwise_AUC_diagnostic_minus_euclidean",
                         "value": round(m05["diagnostic"]["pairwise_AUC"]
                                        - m05["euclidean"]["pairwise_AUC"], 4),
                         "baseline_value": 0.0, "denominator": 256 * 255 // 2,
                         "unit": "auc_gap"},
      "key_counterevidence": "diagnostic eigenvalues 119/95/62/35 (directions exist, do not transfer)",
      "nearest_baseline": "same-budget random-pair metric (identical AUC)",
      "next_conceptual_trial": "none on small MLPs; big-encoder branch untested (no GPU)",
      "paper_section_replaced": "none (boundary finding)",
      "artifacts": ["artifacts/discovery_campaign/r05_mlpA",
                    "artifacts/discovery_campaign/r05_knn_transfer"]})

# ---------- R06 ----------
s06 = J("r06_scatter_full/R06_scatter_result.json")["results"]
spreads = []
for m in ("mpnn", "mlp"):
    curves = [np.array(s06[f"{m}/{l}"]["curve"]) for l in
              ("clean", "Q_ref", "Q_source_dro", "Q_source_greedy")]
    spreads.append(float(np.abs(np.array(curves) - np.mean(curves, axis=0)).max(axis=0)[10]))
save({"route_id": "R06", "status": "parked_model_quality_bound",
      "evidence_level": "true_chaos_oracle_nonchaotic_learned_models",
      "new_question": "Does dependence shift long-horizon rollout error?",
      "what_changed": "spring(contractive)->scatter(true chaos, x24 oracle divergence)",
      "source_commit": COMMIT,
      "perturbation_contract": "bank-4 on 4/5 particles, truly re-integrated",
      "n_parent_scenes": 50,
      "primary_result": {"name": "max_cross_law_spread_at_t10",
                         "value": round(max(spreads), 4), "baseline_value": 0.0,
                         "denominator": 50, "unit": "distance"},
      "compute": {"true_sims": (200 + 50 + 50) * 300,
                  "note": "model rollouts: scenes x modes x arms x 300 steps x 2 models"},
      "key_counterevidence": "learned dynamics non-chaotic; whitebox==clean from t10",
      "nearest_baseline": "clean-initial rollout",
      "next_conceptual_trial": "chaos-capable one-step models (needs GPU-scale training)",
      "paper_section_replaced": "none (parked)",
      "artifacts": ["artifacts/discovery_campaign/r06_trial1",
                    "artifacts/discovery_campaign/r06_scatter_full"]})

print("schema records written to", OUT)
