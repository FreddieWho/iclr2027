# WP4 Report: propagation without update-primacy (2026-09-21)

Unified eval_202 scenes, 6 frozen r04b_s11 arms (eval-only, deterministic;
single-seed arms → model ranking stability unproven, L-006 caveat).

## Model-level (n=6, Spearman + scatter, no SEM)

| arm | static_err | trans_miss | comp_cond | action_invalid |
|---|---|---|---|---|
| clean | 0.170 | 0.558 | 0.828 | 0.770 |
| flipmine | 0.123 | 0.290 | 0.830 | 0.684 |
| fliprand | 0.125 | 0.424 | 0.825 | 0.705 |
| supmine | 0.164 | 0.442 | 0.709 | 0.698 |
| auxmargin | 0.180 | 0.516 | 0.831 | 0.813 |
| relfeat | 0.059 | 0.447 | 0.778 | 0.518 |

Spearman vs action_invalid: **static 0.94 / trans 0.54 / comp 0.54**.
Static error — not update metrics — best tracks consequence across arms.

## Repair propagation (clean→flipmine)
Δstatic −0.047, Δtrans −0.268, Δcomp ≈ 0 (0.828→0.830, n≈53–81, flat),
Δaction −0.086. Chain moves together except emergent-composition, which is
flat here (different construction from the E1 C1 metric; small-n).

## Scene-level (798 pooled scene-records, parent-cluster bootstrap)
- Raw: P(invalid|trans miss) = 0.633 < P(invalid|trans hit) = 0.743;
  RD = −0.110, CI [−0.189, −0.032]. REVERSED vs naive expectation.
- Logistic (invalid ~ static_correct + trans_miss): static −2.71 (dominant
  protective), trans_miss +0.46 (adds risk once static is held fixed).
- Reading: trans-miss scenes tend to be static-correct scenes (confounding);
  within equal static correctness, missed updates still carry extra downstream
  risk. No causal-mediation language beyond this.

## Consequence for Level D
"Update-sensitive measures better track downstream reliability than static
accuracy" is REJECTED at model level (0.54 vs 0.94). The paper keeps the
consequence module as-is (flipmine improves action outcomes) without the
metric-upgrade claim.
