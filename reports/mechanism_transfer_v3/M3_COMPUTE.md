# M3 compute — no-run record (mechanism_transfer_v3)

Status: **not executed**. This file records why M3 did not run, what already
covers the compute question, and the exact trigger conditions that would reopen it.
No M3 training, measurement, or artifact exists in this round.

## 1. What M3 would have asked

Per `docs/2c3e2d9_mechanism_transfer_pack/05_M3_COMPUTE_OPTIONAL.md`:
64→224 upsampling adds no new information yet shows a large J gain, while changing
~12.25× convolution MACs. M3 would separate training-compute, inference spatial
grid, early-downsampling path, and effective object resolution with the minimal
cells A (64 standard), B (64-upsampled-224 standard, old anchor), C (64 standard +
matched training MAC), D (64 with preserved early grid, matched inference MAC via
width/late-block budget), E (conditional: RGB visible-object local sampling/ROI +
global position code, matched inference MAC, no oracle localization).
None of these cells was run: there is no `m3/` directory under
`artifacts/mechanism_transfer_v3/` (only `m1/`, `m2/`) and no `m3/` module under
`experiments/mechanism_transfer_v3/` (only `common/`, `m1/`, `m2/`, `tests/`).

## 2. Why it did not run

1. **N02 already partly answered the compute question.**
   `reports/n02_upgrade/REPORT.md`: on the new backend (2080 Ti, torch 2.5.1+cu121;
   all comparisons internal to the new backend), P0 reproduction passed
   (B−A standard flip 6/6 CIs exclude 0); P1 judged `COMPUTE_OR_CAPACITY_MAJORITY_BY_MEAN`
   — random-init means A ≈ 0.500/0.522/0.478, W ≈ 0.702/0.708/0.629,
   B ≈ 0.809/0.837/0.815, f = (W−A)/(B−A) = +0.619/+0.552/+0.434, mean +0.535
   (wide net 178M params, 2.25G MACs ≥ B's 1.81G; compute and capacity not separable);
   P2 (small-edit mechanism) stays `UNRESOLVED` with two counter-evidence lines
   (lowstride-interaction signs mixed under pretrained, reversed under random;
   the old-bank pretrained hit did not reproduce on the new backend).
   Paper impact was already applied: one appendix sentence, no main-claim upgrade.
   A fresh M3 grid would largely re-spend GPU to re-quantify a confound already
   moved from "completely unaddressed" to "about half quantified."
2. **No spare GPU exists.**
   Resource authority is the single current 2080 Ti instance only — no new rental,
   no farm expansion
   (per `docs/2c3e2d9_mechanism_transfer_pack/02_MASTER_PROMPT.md`: "用户已有2080 Ti实例，
   允许使用；不自行新增租机" and `06_PAPER_GPU_AND_EVIDENCE.md` §4: single-GPU default
   one training queue).
3. **M3 must not preempt M1/M2.**
   Priority order is fixed: M2 strong visual baselines + mask repair → M2 frozen-module
   bridge; M1 small coordinate models on CPU or GPU-idle; M3 only "independently small
   cells" that do not take the main GPU queue
   (per `02_MASTER_PROMPT.md` §执行-4 and `06_PAPER_GPU_AND_EVIDENCE.md` §4).
   The queue was consumed by the M2 16-cell GPU matrix
   (`artifacts/mechanism_transfer_v3/m2/gpu_run_remote/out/`) and the route2 v2/v3
   reruns (12 arms × 3 seeds + clean-only baselines). Closeout commit `be5ce3e`
   records the outcome: "M3默认不跑".

## 3. Exact trigger conditions to reopen M3

M3 reopens only if ALL of the following hold (all from `05_M3_COMPUTE_OPTIONAL.md`
"读法" and "升级要求" plus the master-prompt gating):

1. **Explanatory need**: M1 or M2 needs the N02/compute reading to interpret its own
   result (per `02_MASTER_PROMPT.md` §执行-4: "M3在前两线需要解释N02…时并行小规模运行"),
   OR spare capacity exists that does not preempt the M1/M2 queue.
2. **Matched-cost design**: cells report the training-cost frontier and the
   inference-cost frontier separately (train MAC/updates/wall time/VRAM vs inference
   MAC/grid); no claim of "fully fair" matching if all resources cannot be matched
   at once. Mask sampling must be area-faithful; small-edit slices need sufficient
   independent parents before results are viewed (no repeat of the 5-quartet/3-parent slice).
3. **Pre-stated prediction**: one model-size or observation-perturbation prediction is
   registered before running (not a post-hoc best-resolution pick).
4. **Reading rule committed in advance**:
   - C ties B → write the compute-budget/optimization explanation;
   - D/E beat B at matched inference MAC **and** replicate in an independent
     observation style → a "spatial compute allocation affects change-reliability"
     result;
   - static and flip improve proportionally → NOT a composition-specific mechanism;
   - A/B advantage depends on pretraining → report stratified, not averaged into a
     general law.
5. **Entry bar**: main-text entry only if the major resource confound is excluded;
   otherwise appendix-level compute-sensitivity at most.

If any condition fails, M3 stays closed and the compute reading remains the N02
appendix sentence (about half quantified, mechanism UNRESOLVED).
