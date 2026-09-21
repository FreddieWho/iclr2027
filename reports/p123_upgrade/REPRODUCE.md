# REPRODUCE (P123 upgrade)

## Inputs (hashes)
- bank_dev512.npz sha 2178613957f4 (474 paths + 126151 singles, x/e stored)
- confirm_1007 scenes sha 5c172db3 (SEALED, unused)
- p2_table.npz sha e7ee314b (U02 cache + 6 logit vectors)
- Checkpoints sha in EVIDENCE_CORRECTIONS.md (r04b s11/23/47, T5R3 base/cover s11/23/47)

## One-key recompute (CPU)
python3 experiments/p123_upgrade/build_bank.py --pool eval_202 --tag dev512 --n_parents 512 --seed 777
python3 experiments/confident_blindspots_v2/p1_coordinate.py
python3 experiments/confident_blindspots_v2/p1_analyze.py
python3 experiments/confident_blindspots_v2/p1_football.py --seed 11  # +23/47
python3 experiments/repair_decomposition/p2_table.py   # needs 6 ckpts above
python3 experiments/repair_decomposition/p2_decompose.py
python3 experiments/ccm_audit/paired_compare.py
python3 tests/test_p123_upgrade.py

## Seeds
Sampling/training/model seeds recorded per script (--seed) and in manifests.
Bootstrap: dev 2000 (code default); final comparisons 10000 where stated.
Parent-clustered for coordinate paths; football per-event tables archived
(event-level clustering for any future inference).
