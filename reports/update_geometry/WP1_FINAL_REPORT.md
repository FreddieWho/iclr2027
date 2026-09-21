# WP1 Final Report (2026-09-21): composition is a missed update; incidence is not its law

## What survived
Compositional blindness = missed semantic update along composed paths:
- 229 emergent quartets → 458 paths, 100% oracle single-crossing.
- 196/237 primary paths pure zero-transition; 38 single late/early; 3 ok; 0 recross.
- Attribution: **111/111 AB-miss quartets contain a MISS_NO_CROSS path**; 0 abnormal.
- This part is clean, strong, and mechanism-grade.

## What died
Incidence (I = crossing angle) as the law of update failure:
1. Emergent range: miss 0.79–0.88 flat across I quintiles (n=237, logistic ≈ null).
2. Aimed pools (5066 paths, 3 mining seeds): directional raw gap (M_tan 0.75 vs
   M_nor 0.62, Δ=+0.14, 73 pairs) that **fails the pre-registered SMD gate**
   (0.94–1.27 ≫ 0.10) — confounded, not a claim.
3. Structural cause: high-I crossings intrinsically start/end deeper
   (start-margin SMD 1.27 even after dropping the post-treatment endpoint
   margin; common-support window still SMD 0.7–1.1). Grazing-vs-decisive paths
   with matched margins barely exist — the estimand has no empirical support
   in this task geometry.
4. Consequence: the 84%-vs-44% gap is NOT one continuous incidence law (at
   least not measurably); it stays what E1 called it — two true numbers from
   two denominators, with the random screen over-sampling boundary-hugging
   grazes the model misses most.

## Mechanism note (non-claim)
Miss paths start MORE confident (|logit| 13.4 vs 2.6) — recorded as a
curiosity consistent with "confident far-region starts fail to update,"
confounded, not interpreted.
