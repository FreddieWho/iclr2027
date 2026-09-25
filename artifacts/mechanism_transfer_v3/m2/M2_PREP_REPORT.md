# M2 prep report — expanded bank, mask audit, analytic ceiling

Scope: M2 CPU preparation steps 0–3, which need no frozen relation head and no
GPU. Nothing in this report is a transfer result, a learned-front result, or a
method claim.

Artifacts: `artifacts/mechanism_transfer_v3/m2/` (`bank/`, `mask_audit.json`,
`analytic_baseline.json`, `PREP_RECEIPT.json`).
Code: `experiments/mechanism_transfer_v3/m2/` (+ `DECISION_NOTES.md`).

## 1. Expanded blinded bank

`artifacts/mechanism_transfer_v3/m2/bank/data.npz`, archive SHA256
`69c4bb5c0986701ee1ab7a20e6b50a3a8ae22e2aa959fb57906947fbe6184342`,
1.73 MB, uint8 native-64 RGB.

| split | parents | singleton images | clean | edited | zero-edit parents | at 2-edit cap |
|---|---:|---:|---:|---:|---:|---:|
| train | 1024 | 2532 | 1024 | 1508 | 250 (24.4%) | 734 (71.7%) |
| dev | 128 | 333 | 128 | 205 | 24 (18.8%) | 101 (78.9%) |
| test | 384 | 946 | 384 | 562 | 100 (26.0%) | 278 (72.4%) |

Quartets (test parents only, P, A, B, AB order): **692 quartets over 351
parents**, 384 eligible parents, 2768 images, 0 AB training images.

Frozen contract (`bank_contract.FROZEN`), fixed before generation: independent
per-split parent and observation seeds in the 8328xx block (deliberately not the
pilot's 8327xx block); one oracle margin floor 0.02; one radius schedule
0.10/0.20/0.35/0.50; 16 draws for both miners; at most 2 single edits and 2
quartets per parent; `>= 20` changed pixels at channel threshold 0.2 for every
edited state; L-inf dedup 1e-4.

Realized yield, reported as rejected candidates rather than requested draws:

| stage | drawn | rejected | accepted |
|---|---:|---:|---:|
| singleton single edits (test) | 22 328 | 21 766 counted by reason | 562 |
| quartet single edits (test) | 49 152 | 9 728 (6 119 label not preserved, 3 609 below margin) | 39 424 kept candidates |
| quartet pairs (test) | 541 443 | 539 778 combination preserved, 968 below margin, 5 below pixel threshold | 692 |

Two yield facts the legacy miner did not record. First, the frozen margin floor
rejects 3 609 of 49 152 single-edit candidates and 968 pair candidates that the
legacy 0.005 floor would have admitted. Second, **the changed-pixel filter
rejects candidate quartet states that are visually copies of the parent**: five
candidate quartets were dropped for that reason at 0.02 margin, and on the
archived pilot bank, which had no such filter, 3 of 28 quartets contain a state
that changes fewer than 20 pixels from P (two at 18 pixels, one at 2 pixels).
The filter binds tightly on the expanded bank: the per-state minima over its 692
quartets are 21, 20 and 36 changed pixels for A, B and AB.

Geometry separation, realized not requested: minimum pairwise parent L-inf is
0.144 (train), 0.165 (test); minimum test-to-train parent L-inf is 0.183;
minimum quartet-state-to-train-parent L-inf is 0.151. Parent keys and singleton
image hashes are pairwise disjoint across splits.

Realized coverage (test split, recorded from accepted states rather than draw
counts): all eight edit families appear (shift_AB 132, shift_CD 119,
move_node0/1/2/3 80/72/82/67, rot_AB 6, rot_CD 4); realized radii span
0.10-0.50 with mean 0.203; realized oracle margins span 0.0201-1.030 with mean
0.119, i.e. the frozen floor binds and is not decorative. The 692 quartet edits
cover the same eight families (shift_AB 260, shift_CD 271,
move_node0/1/2/3 221/205/171/192, rot_AB 22, rot_CD 42).

Delete-and-regenerate determinism: passed. Two independent full generations
produced equal SHA256 for all 48 arrays and for the archive.

Every parent coordinate (role-ordered A/B/C/D, colour-preserving endpoint sort
applied by the renderer), every state's edit vector, edit family, radius, oracle
margin and changed-pixel count is persisted, so privileged-geometry supervision
and realized coverage are auditable.

## 2. Mask / pooling audit

`artifacts/mechanism_transfer_v3/m2/mask_audit.json`. The two failure modes are
counted separately and never merged.

At the 7x7 grid the ResNet-18 layer4 actually produces:

| split | mode | red erased of native-visible | blue erased of native-visible | mean valid pooled cells (of 49) |
|---|---|---:|---:|---:|
| train | nearest (legacy) | 218/2532 = 8.61% | 134/2532 = 5.29% | 2.25 / 2.72 |
| train | area (fixed) | 0 | 0 | 8.72 / 9.21 |
| dev | nearest | 30/333 = 9.01% | 15/333 = 4.50% | 2.25 / 2.68 |
| dev | area | 0 | 0 | 8.75 / 9.09 |
| test | nearest | 77/946 = 8.14% | 71/946 = 7.51% | 2.25 / 2.58 |
| test | area | 0 | 0 | 8.67 / 8.95 |

At 14x14 the nearest rule already loses only 16 of 3811 train/dev/test images
(0.4%), and area loses none. **Missing colour is zero everywhere**: every
erasure measured here is downsampling loss, not an absent object. On quartets,
191 of 692 (27.6%) contain at least one state with a natively visible, erased
colour channel under the legacy rule; under the area rule, none.

Palette ambiguity check: over 3 874 816 test pixels, zero pixels fall in the
green threshold gray zone, zero pixels match both colour masks, and zero pixels
are neither colour nor near-background. The frozen thresholds 0.15/0.35 are
unambiguous on this renderer.

Logged audit only, pilot bank (28 quartets, 19 parents, archived corrected v2
predictions): 7 quartets carry an erased channel. Per-seed J3 on the erased
subset is 0.571/0.714/0.571 (direct), 0.571/0.286/0.286 (additive),
0.286/0.286/0.143 (representation), 0.857/0.429/0.857 (interaction) against
0.571/0.571/0.524, 0.333/0.333/0.619, 0.190/0.333/0.333, 0.714/0.476/0.429 on
the non-erased subset. The per-seed directions disagree in both arms, n=7. No
effect is estimable and none is claimed.

## 3. Analytic image ceiling

`artifacts/mechanism_transfer_v3/m2/analytic_baseline.json`. Method: native RGB
to the frozen colour masks, per-colour principal-axis line fit with the known
stamp offset removed, oracle crossing rule on the four recovered endpoints.
It reads no coordinates, no oracle masks, no parent ids, no edit answers.

Pilot bank (`artifacts/e832_focus/route2/data`, 28 quartets / 19 parents,
development evidence only):

| quantity | value |
|---|---|
| test singleton accuracy (319 images) | 0.978 |
| quartet P / A / B / AB accuracy | 1.000 / 0.929 / 0.857 / 0.929 |
| **quartet J3 = J4** | **0.750**, parent-cluster 95% CI [0.500, 0.895] over 19 parents |
| baseline-110 (A,B correct, AB wrong) | 2 of 28 |
| fully fitted quartets | 28 of 28 |

For orientation, the archived contract-corrected v2 learned arms on the *same*
28-quartet bank have mean J3 direct 0.571, additive 0.417, representation 0.274,
interaction 0.583, and the clean-only direct baseline 0.155. This is a
point-estimate comparison on one bank, not a paired test; it says the analytic
reference is above every pilot learned arm, which is what a ceiling statement
should say.

Expanded bank (692 quartets, 351 parents; 685 quartets with all four states
fitted):

| quantity | value |
|---|---|
| test singleton accuracy (946 images) | 0.980 |
| quartet P / A / B / AB accuracy | 0.997 / 0.987 / 0.946 / 0.945 |
| **quartet J3 = J4** | **0.886**, parent-cluster 95% CI [0.857, 0.917] over 349 parents |
| baseline-110 | 38 of 685 |
| endpoint recovery, mean / median / p90 | 0.0286 / 0.0264 / 0.0416 scene units |
| failure decomposition | 943/946 fitted; 3 red "too few pixels"; 63 images touch the image border |

The endpoint error sits at the ~1-pixel quantisation floor of a 64-pixel
renderer, so the remaining analytic errors are not a fitting weakness but a
resolution and occlusion limit.

## 4. What this establishes, and what it does not

Established for step 4:

* a hash-locked, deterministic, split-disjoint bank with 692 test quartets over
  351 parents, ~25x the pilot's 28 quartets, 384 eligible parents inside the
  pre-registered 200-500 band, and 38 analytic baseline-110 quartets against the
  pilot's 2;
* the input-contract defect from R4 is quantified on the real bank (8.1% red and
  7.5% blue test-image erasure under the legacy rule, 0% under the fixed rule,
  all of it downsampling loss) so step 1 cannot silently use the legacy pooling;
* an oracle-free analytic ceiling of J3 0.886 (CI [0.857, 0.917]) on the
  expanded bank and 0.750 (CI [0.500, 0.895]) on the pilot bank. Any learned
  front at or below the pilot's 0.583 is now interpretable against a strong
  non-learned reference rather than against chance.

Not established, and not claimable from this work:

* no learned front, no frozen-head transfer, no adaptation result;
* no claim that the analytic ceiling is reachable by a learned perceptual front
  — step 4 must measure it;
* no claim about the mechanism contract (layered/transparent isolation), the
  interventions of M2.3, or the external renderer of M2.4 — none were run;
* no cross-bank pooling: the pilot's 28 quartets and the expanded 692 are
  separate denominators and must stay separate;
* the pilot erasure/error correlation is descriptive with n=7 erased quartets;
* no sealed pool was read.

## 5. Blockers for the next steps

* **No CUDA on this host** (`torch.cuda.is_available() == False`). M2 steps 1–3
  in the scout's build order (direct ResNet-18 baseline, true-geometry plus
  frozen head, geometry front) cannot run here.
* **AI Galaxy ownership/release is unresolved** in `STATUS.md`, so no remote
  spend was started; the bank is small enough (1.73 MB) to ship as uint8.
* **Step 4 needs a frozen head hash.** M1.2 has since produced trained
  checkpoints under `artifacts/mechanism_transfer_v3/m1/training/runs/`, but
  none was available to this worker. The M1 writer must register the chosen
  head, its class and its weight SHA256 before step 4 starts.

## 6. Reproduce

```bash
cd /home/huyudi/012_conference/iclr2027
python3 experiments/mechanism_transfer_v3/m2/generate_bank.py --out artifacts/mechanism_transfer_v3/m2/bank
python3 experiments/mechanism_transfer_v3/m2/generate_bank.py --validate-only
python3 experiments/mechanism_transfer_v3/m2/mask_audit.py
python3 experiments/mechanism_transfer_v3/m2/analytic_image_baseline.py
python3 -m unittest discover -s experiments/mechanism_transfer_v3/m2 -p 'test_*.py' -v
```

Full generation with the delete-and-regenerate determinism check takes ~54 s on
CPU; all four M2 prep scripts together are under two minutes.
