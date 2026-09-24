"""Make O04 tables from completed frozen-head reanalysis; no fitting."""
import csv,hashlib,json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args()
d=json.loads((a.run/"summary.json").read_text());results=[];parent_rows=[];convex=[]
for tag,row in d["rows"].items():
 for name,r in row["results"].items():
  results.append(dict(encoder_seed=tag,head=name,**{k:r[k] for k in ("S","J_star","J","J_head_dev_threshold","head_dev_threshold","ordering_failure","global_incompatibility","operating_point_gap")}))
  f=np.load(a.run/tag/f"{name}_scores.npz")
  for parent in np.unique(f["parents"]):
   mask=f["parents"]==parent
   parent_rows.append(dict(encoder_seed=tag,head=name,parent=int(parent),n_quartets=int(mask.sum()),**{k:float(f[k][mask].mean()) for k in ("S","J_star","J","J_head_dev")}))
 for name,ls in json.loads((a.run/tag/"solver_logs.json").read_text()).items():
  if not name.startswith("mlp"):convex.extend(ls)
for name,rs in (("results.csv",results),("parent_metrics.csv",parent_rows)):
 with (a.run/name).open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
rows=[]
for arm in ("raw_clean","raw_flipmine","relfeat","relflip"):
 for name in ("logistic_static","logistic_flip","ranking_static","ranking_flip","mlp_static","mlp_flip"):
  vs=[r for r in results if r["encoder_seed"].startswith(arm+"_s") and r["head"]==name]
  rows.append("| "+arm+" | "+name+" | "+" | ".join(f"{np.mean([r[k] for r in vs]):.4f}" for k in ("S","J_star","J","J_head_dev_threshold"))+" |")
text="""# O04 frozen readout reanalysis

Status: EXECUTED_BOUNDED_REANALYSIS; full O04 remains PARTIAL.

12 frozen encoders (4 arms × seeds 11/23/47), 6 selected heads each, all 237 legacy dev512 A/B/AB joint events from 106 parents. This is the original U04 three-state criterion, not four-state accuracy. No new external data. Old outputs remained read-only.

The clean train_101 parents were deterministically split into 409 head-fit and 103 head-dev parents. All edits followed their parent. Frozen encoders had already seen the head-dev parents: this split protects head fitting and selection, not independence of the representation. Clean head-fit moments alone standardize the frozen feature cache; the same moments apply to static, flip, and evaluation. Model/hash lineage cannot be inferred from integer parent IDs: see the separate R03 lineage audit before asserting train/evaluation disjointness.

Static/flip logistic heads start from the identical zero vector; each receives the same three L2 candidates (0.0001, 0.001, 0.01) and L-BFGS-B convergence tolerances. MLP heads start identically for each width/seed, use widths 16 and 64, and receive 300 Adam epochs, with head-dev BCE selection every 10 epochs. Static selection uses static head-dev states; flip selection uses a 50:50 clean/flip head-dev mixture, matching each training objective. Thus supervision regimes differ intentionally; selection is not based on evaluation J. MLP results concern these two budgets only.

The ranking intervention adds logistic positive/negative comparisons across different head-fit parents (4096 candidate pairs, same-parent pairs removed), weight 0.2, to the same state-labelled BCE. Both BCE and ranking use the same convex solver/iteration cap and L2 search; actual iterations are logged, not forced equal. This extends supervision using existing train state labels and is not the original unseen-combination training claim. Every prediction remains a single-state function: no qid, quartet threshold, or evaluation label enters inference.

## Results

Means below average three model seeds on the SAME exposed bank; seeds are not independent tasks. J uses fixed zero logit threshold. J_dev uses a threshold selected for weighted single-state accuracy on head-dev states. J* is the evaluation-truth oracle threshold bound, never deployable performance.

| Frozen encoder | Head/regime | S | J* | J | J_dev |
|---|---|---:|---:|---:|---:|
"""+"\n".join(rows)+"\n\n"
text+=f"All {len(convex)} convex candidate fits report solver success; maximum final gradient infinity norm {max(c['grad_inf'] for c in convex):.3g}. The successful termination reason and full objective traces are saved; success is numerical optimization evidence, not scientific validation.\n\n"
text+="""## Interpretation and limits

Frozen relational encoders retain the highest S/J under these refits. However S changes when the head changes: e.g. raw_clean seed 11 logistic-flip S=0.5738 versus MLP-flip S=0.4810. Therefore S is a property of the final encoder-plus-head scalar score. Relative stability for some relational encoders supports only a bounded empirical observation, not encoder-only or all-head invariance.

The BCE-versus-ranking flip comparison gives limited, seed/encoder-dependent changes; it does not close the large S−J* gap. No universal readout or probability failure is established. Head-dev threshold changes do not equal evaluation-oracle gains.

`summary.json` contains same-parent paired deltas for S, J*, J, both within encoder across heads and across encoder arms at fixed head/seed. It reports quartet-weighted means and parent-equal means with parent bootstrap intervals. Each row uses the same parents; no seed/parent/task denominator inflation. J* paired intervals hold the sample-chosen oracle threshold fixed and are descriptive, not uncertainty for a deployable decision rule. `parent_metrics.csv` and each `*_scores.npz` retain parent IDs and per-event flags/scores. Paired diagnostics are exploratory and not multiplicity-adjusted.

NOT_RUN: new-task or new-model prospectively specified intervention prediction; D10 downstream echo following R09; broader nonlinear architecture search; context-conditioned correction; fresh-bank independent confirmation. This reanalysis cannot make the decomposition a validated prospective intervention diagnostic. Training-bank exposure and mining selection risks remain bounded by the separate lineage review; no split here retroactively repairs historical encoder leakage.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_objective_check.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_refit.py --output artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY
python experiments/e1a933_review/readout_report.py --run artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY --report reports/e1a933_review/O04_readout_FRESH.md
```

The actual executed run is `artifacts/e1a933_review/readout_refit_20260924/`; its stdout is the sibling `.log` file. The fitter refuses an existing directory. SciPy is imported before torch because this environment otherwise selects a system libstdc++ lacking GLIBCXX_3.4.29. The failed import occurred before artifact creation; no computation was silently dropped. Checks cover finite-difference BCE/ranking gradients, convex convergence, common AB reconstruction, and unchanged frozen backbone hashes.
"""
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text)
manifest={str(p.relative_to(a.run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.run.rglob("*")) if p.is_file() and p.name!="SHA256.json"}
(a.run/"SHA256.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(a.report)
