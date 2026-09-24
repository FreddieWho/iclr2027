# R04 start-time confidence and D02 orbit metric repair

R04 completed: production d03_eval.start_confidence_mask indexes original parent logits, and every candidate on the same parent shares retention. Threshold remains declared train_101 2/3 |logit| quantile, frozen per model; actual coverage is reported, no claim that evaluation coverage equals one third.

Both unconditional endpoint risk and start-correct-conditioned update risk now have explicit denominators. Saved s*_singles.npz includes parent, start/endpoint scores, labels, old/new retained masks and all candidate-order selectors. R04_BEFORE_AFTER.csv provides 72 arm×seed×order×preserve/flip rows.

Production regression start=[10,1], threshold5 retains first parent; reordered candidate sequence preserves parent decisions. Old endpoint-based selection cannot satisfy this production contract.

Candidate set overlap (Jaccard):
- flip_first_random: 0.656421 (1559/2375).
- flip_first_stratified: 0.767296 (1708/2226).
- flip_random_stratified: 0.653636 (1555/2379).
- keep_first_random: 0.028057 (393/14007).
- keep_first_stratified: 0.097394 (1278/13122).
- keep_random_stratified: 0.027397 (384/14016).

Candidate order results are conditional on the recorded candidate pool and evaluation filtering. These overlaps do not identify training-order effects. NOT_RUN: an additional train-order randomization trial; the data/recipe factorial O01 is not that trial. Main P1B original confidence result is outside this local field correction.

D02 completed: all8_raw and all8_groupavg are separate. Group closure is evaluated by composing each of eight relabelings with all eight saved scores; average-model orbit logits are asserted equal at 1e-12 float64 tolerance and all8_groupavg exactly equals its identity J. Saved forward multiplier=8; this does not turn raw-model all8 into a group-averaged-model score. Data: artifacts/e1a933_review/data_d02/D02_SUMMARY.json.

Production tests: 3 passed (test_data_repairs.py). No repository-wide suite run; tests directly cover normalization through saved/restored checkpoint, start-time filtering, and raw versus group-average orbit metrics.
