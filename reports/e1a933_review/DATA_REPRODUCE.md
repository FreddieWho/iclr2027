# Worker A reproduction

Base HEAD is e1a933e6b32dd07c6895c7925cbccd9004da7f54; repaired working-tree source hashes are in artifacts/e1a933_review/data_execution_receipt.json. Historical f095 checkpoints/banks are inputs only. SciPy requires the existing conda libstdc++ on this host.

The commands below reproduce into NEW directories. Choose an unused run tag; completed-stage outputs refuse overwrite. O01 intermediate training can resume existing per-candidate checkpoints after interruption; U10 stage provenance guards require a fresh output directory; do not mix parameters in one directory. Archived actual runs used the same stage commands without --out-dir, producing data_u10/data_o01/data_lineage/data_d02/data_d03/data_d09/data_source_eval under artifacts/e1a933_review. A syntax indentation fix and missing conda-library retry occurred before the successful D03/lineage/O01 runs; no old artifacts were written.

```bash
cd /home/huyudi/012_conference/iclr2027
export LD_LIBRARY_PATH=/opt/anaconda3/lib:${LD_LIBRARY_PATH:-}
run_tag=reproduce_01
run_root=artifacts/e1a933_review/${run_tag}
mkdir "$run_root"
python experiments/e1a933_review/data_u10.py --stage train --out-dir "$run_root/u10"
python experiments/e1a933_review/data_u10.py --stage eval --out-dir "$run_root/u10"
python experiments/f095_campaign/d02_orbits.py --out-dir "$run_root/d02"
python experiments/f095_campaign/d03_eval.py --out-dir "$run_root/d03"
python experiments/f095_campaign/d09_audit.py --out-dir "$run_root/d09"
python experiments/e1a933_review/data_lineage.py --out-dir "$run_root/lineage"
for stage in gen train eval; do
  python experiments/e1a933_review/data_o01.py --stage "$stage" --out-dir "$run_root/o01"
done
python experiments/e1a933_review/data_source_eval.py --o01-dir "$run_root/o01" --out-dir "$run_root/source_eval"
python -m pytest -q experiments/e1a933_review/test_data_repairs.py
```

All seed loops are explicit inside the audited Python scripts: 11, 23, 47. U10 trains 18 models; O01 trains 108 candidates and preserves all 300-epoch curves. Full-batch training uses separate fixed initialization seed and subset RNG, no stochastic batch ordering. No GPUs, downloads, external data or paid resources used.

Files added after inference for human review (R04_BEFORE_AFTER.csv, CANDIDATE_SET_OVERLAP.json and Markdown reports) summarize saved model scores and metadata; they are not additional fitting or held-out selection. O01_FACTORIAL.csv includes CCM denominators, R_full/M, original-111 count and degradation. Parent bootstrap shares every arm/state within parent.

Scope now closed (was NOT_RUN at the time of writing): exhaustive historical augmentation-state duplicate matching (R03, 392412 states, 0 exact/0 near for the quartet banks); equal-parameter U10 typed vs distance optimization search (O02, typed deficit survives matched capacity + budget); O01 extra optimizer families and warm-start/plain/PCT/replay factorial (O01, 648 arms). Still NOT_RUN: new U10 difficulty/chirality tasks (optional per the O02 card); source training-order randomization; O04 typed-encoder variant. None of these is represented as a completed scientific test.
