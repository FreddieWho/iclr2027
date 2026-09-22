# FINAL_EVIDENCE_TABLE (rebuilt 2026-09-22; supersedes reports/FINAL_EVIDENCE_TABLE.md for new claims; old file kept as history)

## A. Headline phenomenon (confirm: fresh holdout_909, sealed)
| claim | number | artifact |
|---|---|---|
| conditional composition miss (clean) | 151/176 = 85.8% | E1 result.json |
| emergent conditional (flipmine, NON-paired denom) | 109/141 = 77.3% | E1 result.json |
| targeted transition miss | 28/64 = 43.8% -> 20/64 = 31.3% | E1 result.json |
| action invalid lam2 | 0.75 -> 0.625 (n=144) | E1 result.json |
| static error | 0.180 -> 0.107 (n=512) | E1 result.json |

## B. P1 confidence (CONFIRM: confirm1007 bank + dev bank; coordinate only)
| claim | number | artifact |
|---|---|---|
| update err by confidence | Q0 0.36-0.61 -> Q3 0.98-1.00 (12 splits; eleven 1.00, one 0.983) | p1b_res + p1b_confirm |
| static err by confidence | Q0 0.27-0.51 -> Q3 0.02-0.06 (classic direction) | P1B.json |
| retention top25 | flip 1.00 / preserve-update 0.00 | P1B.json |
| grouped-CV conf AUC | conf pooled 0.78-0.95 (clean 0.93-0.95); difficulty 0.42-0.61 | P1B.json grouped_cv |
| repair residual paired CI | all exclude 0 (dev marginal s11/s23 touch 0) | P1G.json |
| football P1 | floor/null (old numbers VOID, mapping bug) | p1football_*.json |

## C. Repair component metrics (confirm where noted, else dev/fresh as labeled)
| claim | number | artifact |
|---|---|---|
| single-flip C1 (PAIRED same samples) | 130/248 -> 86/248 | E1 result.json |
| C0 false alarm | 0.262 -> 0.238 | E1 result.json |
| U02 feasible-set coverage | refusal 0.43 -> 0.12 (n=93 test scenes) | u02 result.json |

## D. P3 migration / joint consistency (dev bank + confirm bank)
| claim | number | artifact |
|---|---|---|
| R_endpoint (baseline-110) | dev 0.43-0.48 | P3_FINAL.json metrics.R_endpoint |
| R_full (baseline-110) | dev 0.03-0.10; relflip confirm 0.45-0.50 | P3_FINAL.json; RELFLIP_CONFIRM.json |
| M migration | dev 0.36-0.40; relflip confirm 0.25-0.30 | same |
| J joint consistency | clean 0.05; flipmine 0.06-0.08; relfeat 0.13-0.23; relflip 0.38-0.47 | P3_FINAL / RELFEAT / RELFLIP_CONFIRM |
| common-set direction | inconsistent incl. s23 reversal | P3_FINAL.json |

## E. Downstream (fresh + dev as labeled)
| action invalid lam2 | 0.75 -> 0.625 fresh; dev decomposition in P2 | E1; P2_DECOMP.json |
| P2 affine fails | lex/affine <= clean; static/joint worse | P2_OPERATING.json |
| P2 equal coverage | repair quality higher at higher cost | P2_OPERATING.json |

## F. Football (confirm-sd: 3-seed same-direction, no pristine holdout)
transitions s11 0.277->0.129, s23 0.383->0.184, s47 0.336->0.176; events and
actions same-direction (HEADLINE_FREEZE). P1 inversion NOT observed.

## G. Pixel (confirm: 3 train seeds x 5 styles)
emerg 0.537->0.351 same-style; style gains as listed; color-swap collapse noted.

## H. Exploratory / appendix / negative
random-screen 84.4% (dev screen rate); N04 62/66 discovery; X03 scoped
negative; E7 12x5 negative; keepbal fair negative; risk-demo negative;
incidence law dead (WP1); football P1 null.
