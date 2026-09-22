# REPRODUCE (one-key recompute)

## Sample banks (frozen .npz + manifest sha)
- dev: artifacts/p123_upgrade/bank/bank_dev512.npz (sha 2178613957f4)
- confirm: artifacts/p123_upgrade/bank/bank_confirm1007.npz (sha 28365ae3)
- build: python3 experiments/p123_upgrade/build_bank.py --pool <name> --tag <t> --n_parents N --seed S

## P1 coordinate
python3 experiments/confident_blindspots_v2/p1_coordinate.py --bank <tag> --out <dir>
python3 -c "import sys; sys.path.insert(0,'experiments/confident_blindspots_v2'); from pathlib import Path; import p1_analyze as A; A.IN=Path('<dir>'); A.OUT=Path('<dir>'); sys.argv=['x','--out','<dir>']; A.main()"

## P1 football
python3 experiments/confident_blindspots_v2/p1_football.py --seed <11|23|47>

## P2 action table + decomposition
python3 experiments/repair_decomposition/p2_table.py   # reuses U02 cache frames
python3 experiments/repair_decomposition/p2_decompose.py

## P3 paired comparison
python3 experiments/ccm_audit/paired_compare.py --bank <tag>

## Unit tests
python3 -m pytest tests/test_p123_upgrade.py -v

## Checkpoints (sha256 in EVIDENCE_CORRECTIONS.md)
r04b_s{11,23,47}/{clean,flipmine}; T5R3 base + cover s{11,23,47}.
CPU-only; torch threads fixed per script.
