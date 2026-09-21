# Novelty Positioning: Semantic Update Geometry (2026-09-21)

Candidate core proposition: **static correctness does not guarantee correct
semantic updating as the world changes**; update failures are geometrically
structured by how a change crosses the true semantic boundary and by the
local misalignment between the semantic boundary and the model decision
boundary. Compositional failure is demoted to a special construction of
semantic-update failure. All of this is conditional on WP1–WP5 evidence.

## 1. Generic compositionality (atomic → composition, held-out combos)
The neighborhood is crowded: benchmarks test unseen combinations at the data
level and ask whether the model gets the held-out output right. Our bar is
strictly harder and different in kind: correct atomic judgments are
*conditioned on* (151/176 headline), and the object is the *decision update*
along a continuous controlled change, not the unseen output. The N04
displacement-matched control (+29pp composition-specific over matched singles)
is what separates "composition" from "farther walk" — most benchmarks lack
that control.

## 2. Geometric / spatial reasoning failure ("does the model get it right")
Static-image geometric benchmarks ask whether a judgment is correct. We ask
whether a *correct* judgment *updates correctly* when the world crosses a
known semantic transition. The 84%-vs-44% denominator discipline is the
sharpest statement of the difference: the same phenomenon sampled in two
crossing geometries gives two numbers, and the geometry (tangential/grazing
vs decisive crossing), not the model, explains the gap. No static benchmark
can express that.

## 3. Decision-boundary geometry / adversarial robustness
That literature studies the *model's own* boundary (its shape, curvature,
distance to it) — usually without any ground-truth semantic boundary to
compare against. We invert the setup: the semantic boundary is externally
defined by the oracle (exact crossing localizable by bisection to |Δt|<1e-4),
and the model boundary is measured *against* it (offset D, coverage
R_boundary, alignment A). Closest neighbor: Concept Boundary Vectors
(arXiv 2412.15698) — representation-space directions across concept
boundaries. Difference: CBV characterizes where a concept lives in
representation space; we characterize whether a decision *follows* an
input-space semantic transition, crossing by crossing, with the oracle
crossing as ground truth. Adversarial work perturbs to break the model; our
paths are oracle-labeled semantic transitions, not attacks.

## 4. Counterfactual / hard-positive training ("targeted examples help")
That literature proves targeted examples improve compositionality/robustness,
usually framed as more/better hard data. Our candidate upgrade (conditional
on WP2/WP5): the improvement specifically tracks *semantic-boundary
coverage/alignment and update geometry* — flipmine works because it covers
missing boundary-crossing transitions (U03's parked pixquartet is the
contrast case: near-boundary equal-weight supervision actively hurts, so it
is not "more hard examples" but *which* crossings). If WP5 shows
incidence-balanced coverage beats uniform flipmine on low-I blind spots with
no static/C0 cost, the repair claim becomes geometric rather than generic.

## What we do NOT claim
- No universal complexity theorem (3 families max, §8 caps).
- "Associated with" precedes "caused by" unless evidence forces it (WP2/WP4).
- Failed alternatives (slope/SDF/dense/relational/quartet/X03) are boundary
  evidence, not strawmen: they delimit "the defect is in updating, not seeing."
- Single-frame, one coordinate encoder family + one small CNN, football
  without pristine holdout — all carried as explicit limits (Sec. 07).
