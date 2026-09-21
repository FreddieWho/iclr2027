# Update-Geometry Master Protocol (frozen 2026-09-21, before any new number)

Strategic goal: test whether transition blindness / composition blindness /
consequence failure / flipmine repair unify into one mechanism chain
(semantic boundary → crossing → missed update → composition wrong →
consequence wrong → boundary-targeted coverage repairs). If data refuses,
keep the current paper structure.

## Frozen facts (from Phase-0 audit, never redefined post-hoc)
- 151/176 = 85.8% fresh compositional conditional miss (holdout_909, sealed/used).
- Targeted fresh path transition miss 28/64 = 43.8%.
- Random-edit single-turn paths 108/128 = 84.4% (different denominator; never mix/average).
- Fresh action invalid 75.0% (λ=2).
- Flipmine fresh C1 52.4%→34.7%, no static cost (18.0%→10.7%), no C0 inflation.
- Football 3-seed + pixel 3-training-seed evidence stands.
- Foundation branch closed: no DINO, no mid-layer, no Task 3.

## Models (frozen checkpoints, zero training unless WP5 authorizes)
- Primary: `artifacts/discovery_campaign/r04b_s11/clean` (coordinate MLP, E1 headline model).
- Secondary: `artifacts/discovery_campaign/r04b_s11/flipmine`.
- WP4 may add existing failed-alternative checkpoints as natural variation (no retraining).

## Data discipline
- Discovery/engineering: `train_101`, `eval_202` (+recheck pools for n only, never for selection).
- Fresh confirm: `geometry_confirm` minted once with generator seed **26092201**,
  parent-disjoint from train/dev/E1 pools; models frozen; one-shot.
- `paper/` untouched throughout. No post-hoc metric/cutoff changes.

## Pre-registered constants (all WPs)
- Coarse scan t = 0,.02,…,.1 (51 pts); bisection 12 iters (|Δt| < 1e-4).
- Incidence I = |n_o^T v|/(||n_o||·||v||), n_o = gradient of the flipping
  orientation value at x* (sign toward class 1); gradient epsilon audit (2 epsilons).
- Tangent I ≤ 0.25 / Normal I ≥ 0.75 (frozen before confirm).
- Matching covariates: displacement norm, start/end oracle margin, base label,
  edit family, endpoint label (+ model start confidence ideally); |SMD| < 0.10.
- Bootstrap 10,000; resampling unit = parent scene (never path sample points).
- Effect sizes (rate diff / OR / median lag / alignment diff + CI), never p-only.

## WP1 (highest priority)
Quartet path decomposition (γ_A, γ_B) on valid emergent quartets
(yA=yB=y0, yAB≠y0); record full trajectory fields; primary = oracle exactly
one crossing + model start correct. Outcomes: P(miss|I) curve, lag|I,
matched tangent-vs-normal ΔM (paired bootstrap), composition→transition
attribution fraction. Success §17-WP1 (needs discovery + fresh confirm same
direction + not a displacement/margin confound).

## WP2 (mechanism: why)
Reuse WP1 oracle crossings + independent boundary sampling (label-balanced,
parent-disjoint confirm). Metric A local alignment (signed cosine, keep sign),
Metric B offset D within frozen radius r = 0.25 × median atomic-edit norm,
Metric C true normal alignment where a model root exists; NO_LOCAL_MODEL_BOUNDARY
is a first-class outcome (coverage R_boundary). Primary: clean vs flipmine on
identical points, paired bootstrap.
**Audit caveat (N02): model-gradient↔outcome association was already null on
clean mined flips (cos 0.087 miss vs 0.067 correct). WP2 does NOT re-test
prediction; the live question is repair shift (coverage↑/offset↓/alignment↑).
WP2-success-#2 (mismatch↔miss link) is therefore pre-weakened; if absent, the
ceiling is BOUNDARY_ASSOCIATION_ONLY via the clean-vs-flipmine shift.**

## WP3 (cross-family)
Circle-overlap + flat-boundary families, same capacity/optimizer philosophy,
equal train budgets, 3 seeds, static competence gate first. Claims capped at
§8 (generalizes / attenuated, never a complexity theorem). Kill: inconsistent
direction → SEGMENT_SPECIFIC, no 4th family.

## WP4 (propagation)
Reuse frozen checkpoints; unified eval scenes; model-level Spearman + scatter
only if n<15 (no SEM/mediation); scene-level P(action invalid|update miss) vs
hit + risk diff/OR; logistic control static+transition with cluster bootstrap.
Carry N06 mechanism line: λc=0 shows no judgment gap (flipmine gain is
confidence-under-cost). Never write "causal mediation" beyond data.

## WP5 (conditional on G1 AND G2)
Incidence-balanced boundary coverage; 3 arms (Random / Uniform Flipmine /
Geometry-balanced) equal budget, same init distribution/optimizer/steps/seeds≥3;
no composition training. **Noise-floor discipline (N04): single-seed retrain
comparisons are non-claims (±15pp floor); superiority needs multi-seed
consistency with effect >> floor.** Negative → UNIFORM_FLIPMINE_SUFFICIENT, stop.

## Execution order
Phase0 audit → WP1 discovery → WP1 fresh confirm → (NO: minimal WP2 diagnostic,
WP5 parked) / (YES: WP2 → WP3 + WP4 → WP5 gate → FINAL CLAIM FREEZE).
WP3/WP4 may parallelize after WP2; confirm data viewed once, never re-tuned.

## Verdict vocabularies (§15) and claim ladder (§8, A–E) apply unchanged.
Stop rule §17: no sixth direction after this round.
