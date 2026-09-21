# WP1 Discovery Report: composition paths (2026-09-21)

Model `r04b_s11/clean` (frozen), pool `eval_202` (512 parents).
229 valid emergent quartets → 458 composed paths, all oracle single-crossing,
407 with incidence (0 gradient-audit failures). Primary (start-correct): n=237.

## Q1: composition = missed update? YES, 100%
- 196/237 pure zero-transition miss; 38 single late/early turns; 3 ok.
  Zero multi-turn/recrossing paths (WP1.9 answered: no cross-back).
- Attribution: 111 AB-miss vs 17 AB-hit quartets (both starts correct);
  **111/111 miss cases contain a MISS_NO_CROSS path**; 0 abnormal.
- Start confidence curiosity (non-claim): miss paths start MORE confident
  (|logit| 13.4 vs 2.6) — recorded, likely margin-confounded, not interpreted.

## Q2: does miss depend on incidence? NO (observable range)
- Miss 0.79–0.88 flat across I quintiles; logistic coef −0.39 (≈null).
- Distribution restriction: I max 0.81, I≥0.75 only 4/407. Emergent
  quartets do not populate the decisive-crossing regime.

## Q3: one law for 84% vs 44%? NOT SUPPORTED here
- Emergent-path miss ≈ 0.83 at all observable I. The targeted-44% regime
  (high-I decisive crossings) is absent from this distribution, so neither
  a continuous law nor its absence can be closed from emergent paths alone.

## Matched tangent/normal from emergent paths: INFEASIBLE
- 57 tangent vs 4 normal; SMD fails (1.4–1.9). Normal crossings must be
  deliberately constructed (WP1.14 aimed pools), not harvested from emergent.
