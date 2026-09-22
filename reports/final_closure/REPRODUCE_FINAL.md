# REPRODUCE_FINAL: one-key recompute for every manuscript number

Conventions: `export LD_LIBRARY_PATH=/opt/anaconda3/lib:$LD_LIBRARY_PATH`
first on this machine; model forward batch 256/512 as coded; torch threads
fixed per script. Frozen banks are read-only inputs (verify sha against
manifest before running).

## P1 dev (bank_dev512, sha 21786139…)
python3 experiments/next_novelty/p1_unified.py --bank dev512
python3 experiments/next_novelty/p1b_analysis.py \
  --indir artifacts/next_novelty/p1_unified --out artifacts/next_novelty/p1b_res
python3 experiments/next_novelty/p1g_repair.py \
  --indir artifacts/next_novelty/p1_unified --out artifacts/next_novelty/p1g_res

## P1 confirm (bank_confirm1007, sha 28365ae3; frozen thresholds/settings)
python3 experiments/next_novelty/p1_unified.py --bank confirm1007 \
  --p1old artifacts/p123_upgrade/p1_confirm --out artifacts/next_novelty/p1_unified_confirm
python3 experiments/next_novelty/p1b_analysis.py \
  --indir artifacts/next_novelty/p1_unified_confirm \
  --out artifacts/next_novelty/p1b_confirm \
  --freeze artifacts/next_novelty/p1_freeze.json
python3 experiments/next_novelty/p1g_repair.py \
  --indir artifacts/next_novelty/p1_unified_confirm \
  --out artifacts/next_novelty/p1g_confirm --freeze artifacts/next_novelty/p1_freeze.json

## P3 migration (dev bank; parent-cluster CIs)
python3 experiments/ccm_audit/p3_final.py --bank dev512
# keepbal fair control (fresh training, 3 seeds):
python3 experiments/ccm_audit/p3d_fair.py --seed {11,23,47} --epochs 300
python3 experiments/ccm_audit/p3d_eval.py \
  --ckptdir artifacts/next_novelty/p3d_fair --out artifacts/next_novelty/p3d_fair_eval
# relfeat closure + conditional extension:
python3 experiments/discovery_campaign/r04b_methods.py \
  --train artifacts/discovery_campaign/scenes/train_101 \
  --output artifacts/discovery_campaign/r04b_s{23,47}_relfeat \
  --methods relfeat --seed {23,47}
python3 experiments/ccm_audit/relfeat_eval.py
python3 experiments/ccm_audit/relflip_train.py --seed {11,23,47}
python3 experiments/ccm_audit/relfeat_eval.py --extra relflip \
  --out artifacts/next_novelty/relflip_eval

## P2 operating point (frozen wide grid in code; do NOT widen post-hoc)
python3 experiments/repair_decomposition/p2_operating.py
python3 experiments/repair_decomposition/p2_affine_atoms.py

## Unit tests (all must pass; no vacuous assertions)
python3 -m pytest tests/test_next_novelty.py tests/test_p123_upgrade.py -q

## Figures (data only from frozen JSONs)
python3 scripts/figures/fig_p1_p3.py  # -> paper/figures/fig2_p1_dual.pdf, fig3_p3_migration.pdf

## Manuscript numbers
Every number in paper/ traces to reports/final_closure/MASTER_CLAIM_LEDGER.md;
run `grep -n "[0-9]" paper/sections/*.tex` against the ledger for the audit.
