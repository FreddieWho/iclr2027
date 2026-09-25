# M2 prep decision notes

Short records of onsite implementation choices in the M2 CPU prep step, with the
contract or measurement that justifies each one. Nothing here changes the task,
the frozen seeds, the supervision, or the metric definitions.

## D1 — Singleton and quartet label semantics follow the archived constructions

**Chosen.** Singleton supervision = a single atomic edit that *flips* the label.
Quartet supervision = an *emergent* quartet where each single edit *preserves*
the label and the combination flips it.

**Why.** These are the established project constructions:
`experiments/e832_focus/route2_visual/data_generator._single_states` rejects
`y1 == y0`; `experiments/next6_ef0f7a3/u01_quartet.mine_quartets` keeps only
pairs with `ya == yb == y0` and `yc != y0`. An earlier draft of this module had
the quartet rule inverted (singles flip, combination preserves); that produced
labels `[y0, 1-y0, 1-y0, y0]` and a yield profile (278 eligible parents, 1.38
quartets/parent) that did not match the archived pilot or the pre-run yield
estimate. It was corrected before any artefact was frozen.

**Evidence.** `test_quartets_follow_the_emergent_contract` asserts
`labels[:,0] == labels[:,1] == labels[:,2]` and `labels[:,3] != labels[:,0]`;
`test_singleton_edits_flip_the_label` asserts the opposite for singletons. After
the correction the realized quartet yield is 1.80 per eligible parent, matching
the model-blind pilot estimate of 1.84.

## D2 — One render seed per parent, shared by all of its states

**Chosen.** Every render of one parent (its clean image, all of its single-edit
states, and all four states of any of its quartets) uses one deterministic seed.

**Why.** The renderer draws one uniform background scalar per image. If each
state used a different seed, a render difference would be dominated by the
background change, and the frozen ``>= 20 changed pixels`` filter would be
measuring nothing. Holding the background fixed makes the filter and the
AB-vs-P comparison measure geometry.

**Disclosed difference.** The archived pilot rendered quartet states with
`state_index`-dependent seeds, so its four states have different backgrounds.
The expanded bank deliberately does not reproduce that. This is a nuisance
change, not a task change; the labels, roles and oracle are untouched.

## D3 — uint8 image storage with a checked round-trip

**Chosen.** Images are stored as uint8; consumers obtain float32 in [0,1] as
`images.astype(float32)/255`.

**Why.** `06_PAPER_GPU_AND_EVIDENCE.md` §4 asks for uint8 caching so the bank can
be shipped to the GPU host without moving 224-px float tensors. The renderer
palette is 28 distinct red values, a uniform gray background and 28 blue values,
so uint8 is lossless for the mask contract.

**Evidence.** `test_uint8_quantisation_preserves_the_frozen_masks` asserts the
red and blue masks are bit-identical before and after quantisation;
`to_uint8_checked` fails if any channel moves by more than half a step.

## D4 — Split sizes 1024 / 128 / 384 test parents

**Chosen.** train 1024, dev 128, drawn test 384.

**Why.** The model-blind yield pilot measured the fixed-margin, mixed-radius
schedule at ~93% eligible parents and 1.84 quartets per eligible parent. 384
drawn test parents land inside the M2.0 target band of 200-500 eligible
parents, and 1024 train parents are the low end of the suggested 1024-4096
range. Realized: 384 eligible, 351 with retained quartets, 692 quartets.

## D5 — The total quartet cap is a resource bound, not a statistical choice

**Chosen.** `max_quartets_total = 1024` (non-binding). The substantive frozen
constraint is at most 2 quartets per eligible test parent.

**Why.** The first draft fixed a total cap of 384 before the yield was known.
The realized yield is 692 quartets at 1.73 MB, and the pair stage costs ~5 s, so
a binding total cap would discard half the available power for no resource
reason. `06` §5 allows bank size to be adjusted by yield; it forbids adjusting
it by result direction, and no model result exists at this point. The realized
count is recorded (`yields.quartet.accepted`) rather than imposed.

## D6 — Quartets are mined on test parents only

**Chosen.** Quartet mining uses the test parents; train and dev keep singletons
only.

**Why.** M2.0 requires the target AB test not to enter training or selection.
The realized geometry separation is large: minimum quartet-state to train-parent
L-inf is 0.151 and to dev-parent is 0.178, versus an L-inf dedup threshold of
1e-4.

## D7 — The analytic endpoint fit removes the known stamp offset

**Chosen.** The per-color principal-axis fit subtracts
`stamp_half_width * (|v_x| + |v_y|)` from the extreme projection before placing
the endpoints.

**Why.** `n08_visual.render` stamps an axis-aligned square of half-width 2 at
each sampled pixel, so the raw projection extremes overshoot the true endpoint
centres by exactly that amount. The offset is renderer geometry published in the
repository, not oracle information, and it is recorded in the manifest.

**Evidence.** `test_line_fit_recovers_known_endpoints_and_label`; the measured
mean endpoint error on 943 fitted test images is 0.0286 scene units, close to
the ~1 pixel quantisation floor, so no second estimator was added.

## D8 — Mask audit grids 7x7 and 14x14

**Chosen.** Report the legacy nearest rule and the fixed area rule at the two
grids the ResNet-18 front actually produces (layer4 at 7x7 and layer3 at 14x14),
in addition to the per-quartet state count at 7x7.

**Why.** These are the grids the M2 front will use. The yield scout's 2x2 stress
grid describes a configuration that will not be run and would overstate the
erasure rate; it is omitted rather than reported as if it were operative.

## D9 — The pilot error correlation is descriptive only

**Chosen.** The archived 28-quartet pilot bank is correlated against the
archived corrected v2 predictions and reported with its n, never as evidence.

**Why.** Seven quartets carry an erased channel. At that n no effect is
estimable, and the per-seed directions disagree. Repair R6 forbids turning a
wide-interval exploratory slice into a gate in either direction.

## D10 — Step 4 blockers recorded, not worked around

No frozen relation head was available to this worker, and no CUDA device exists
on this host. Rather than train a substitute head or fabricate a transfer
result, the missing items are listed explicitly in `PREP_RECEIPT.json`
(`not_run`, `blockers`).
