# MASTER_CLAIM_LEDGER (2026-09-22): the ONLY source for paper numbers

| claim | number | numerator | denominator | dataset | split | seed/model | confirm/dev/exploratory | artifact | allowed wording | forbidden wording |
|---|---|---|---|---|---|---|---|---|---|---|
| headline conditional comp miss | 151/176=85.8% | 151 | 176 | coord | holdout_909 fresh | s11 clean | confirm | E1 result.json | atomic-correct conditional composition miss | paired repair delta |
| repair emergent conditional | 109/141=77.3% | 109 | 141 | coord | holdout_909 fresh | s11 flipmine | confirm | E1 result.json | non-paired denominators 176 vs 141 | 109/176; paired fix |
| C1 single-flip miss | 130/248=52.4% -> 86/248=34.7% | 130->86 | 248 | coord | holdout_909 fresh | clean/flipmine PAIRED | confirm | E1 result.json | single-flip C1 miss | compositional C1 miss |
| C0 false alarm | 0.262->0.238 | — | 248 | coord | holdout_909 fresh | both | confirm | E1 result.json | within +-5pp, no inflation | — |
| static error fresh | 0.180->0.107 | — | 512 | coord | holdout_909 fresh | both | confirm | E1 result.json | improves; no accuracy cost | no pointwise improvement (universal) |
| targeted transition miss | 28/64=43.8% -> 20/64=31.3% | 28->20 | 64 | coord | holdout_909 fresh | both | confirm | E1 result.json | targeted fresh paths | averaged with 84.4% |
| random-screen single-turn | 108/128=84.4% | 108 | 128 | coord | dev screen | s11 clean | discovery | N01 | random-edit screen rate | tangential law; generalized rate |
| action invalid lam2 fresh | 0.75->0.625 | — | 144 | coord | holdout_909 fresh | both | confirm | E1 result.json | hard fresh cases | — |
| P1 update reversal | Q0 0.53-0.68 -> Q3 1.00 | — | ~200-740/model | coord | dev bank + confirm1007 | 6 models | CONFIRM | p1b_res + p1b_confirm | confidence reverses meaning under true change | causes blindness; universal; football |
| P1 static direction | Q0 0.36 -> Q3 0.02 | — | ~5000/model | coord | dev bank | 6 models | dev | P1B.json | classic calibration direction | — |
| P1 selection top25 | flip 1.00, preserve-update 0.00 | 101/271 | retained | coord | dev+confirm | s11 clean | CONFIRM | P1B.json | conditional on retained set | overall deployment risk |
| P1 repair residual | clean 1.00 -> repair 0.53-0.92 | — | ~140-190 | coord | dev+confirm | paired | CONFIRM | P1G.json | means improve, residual dominates | same stubborn errors |
| P3 R_full | 0.03-0.10 | — | n110=110-120 | coord | dev bank | 3 seeds | dev | P3_FINAL.json | full repair rare | repair fixes composition |
| P3 migration M | 0.36-0.40 | — | n110 | coord | dev bank | 3 seeds | dev | P3_FINAL.json | migration dominates | — |
| P3 J | 0.05-0.06 -> 0.06-0.08 | — | 237 | coord | dev bank | 3 seeds | dev | P3_FINAL.json | barely moves | — |
| P3 common reversal | s23 0.920 -> 0.947 | — | 75 | coord | dev bank | s23 | dev | P3_FINAL.json | direction inconsistent | stable gain |
| relfeat J | dev 0.13-0.21; confirm 0.16-0.23 | — | 237/509 | coord | dev+confirm1007 | 3 seeds | CONFIRM | RELFEAT.json | hand-designed relation-aware input | architectures solve it |
| relflip J/R_full/M (dev) | J 0.38-0.42; R_full 0.36-0.44; M 0.32-0.42 | — | 237 | coord | dev bank | 3 seeds | dev | RELFLIP_MIG.json | moves toward full consistency | universal method |
| relflip J/R_full/M (confirm) | J 0.44-0.47; R_full 0.45-0.50; M 0.25-0.30 | — | 509 | coord | confirm1007 | 3 seeds | CONFIRM | RELFLIP_CONFIRM.json | full repair exceeds migration | universal method |
| P1 grouped-CV conf AUC | 0.93-0.95 vs noconf 0.53-0.61 | — | ~500-1300 | coord | dev+confirm | 6 models | CONFIRM | P1B.json | difficulty carries ~nothing held-out | causal mechanism |
| P1 temperature | T*~=8, ranks invariant | — | dev singles | coord | dev bank | s11 clean | dev | P1B.json | calibration != update-risk calibration | saturation artifact |
| keepbal fair rerun | J/R_full/M inconsistent 3 arms | — | 237 | coord | dev bank | 3 seeds | negative | P3D_PILOT(fair) | preserve-balancing insufficient | method |
| P2 affine edge | s11 fit at grid edge (1.0,-8.0) | — | 96 eval | coord | U02 scenes | s11 | dev | P2_OPERATING.json | limitation stated | silent widening |
| P1 selection rho | top25 R=rho exactly (flip 1.0, prv-update 0.0) | 101/271 | retained | coord | dev+confirm | s11 clean | CONFIRM | P1B.json | environment-composition decides | overall deployment risk |
| P2 affine fails | lex/affine <= clean; static/joint worse | — | 96 eval scenes | coord | U02 scenes dev/eval | 3 seeds | dev | P2_OPERATING.json | not pure operating-point shift | — |
| P2 equal coverage | repair quality higher at higher cost | — | matched levels | coord | U02 scenes | 3 seeds | dev | P2_OPERATING.json | quality\|same-coverage | — |
| football transitions | s11 0.277->0.129 etc. | — | per match | football | cross-field | 3 seeds | confirm-sd | HEADLINE_FREEZE | same-direction, no pristine holdout | single-read proof |
| football events/actions | events + actions same-direction | — | per match | football | cross-field | 3 seeds | confirm-sd | HEADLINE_FREEZE | limits stated | P1 inversion |
| pixel emerg | 0.537->0.351; 5 styles same-signed | — | 547/style | pixel | unseen styles | 3 train seeds | confirm | u03b_* | relative gain stable; absolute style-sensitive | P1/migration claims |
| U02 coverage | refusal 0.43->0.118 | — | 93 scenes | coord | dev/test | s11 | discovery | u02 result.json | feasible-set coverage | non-refusal precision gain |
| X03 scoped negative | 0.544/0.491/0.484; anti 0.147 | — | 450 (anti 150) | coord | fresh bindings | s11 | park | x03 result.json | rank-4 patch at this layer | fundamentally cannot compose |
| E7 falsification | 12 arms x 5 seeds all lose/tie | — | 60 trainings | coord | dev | 5 seeds | negative | E7 verdict | slopes not binding | — |
| football P1 | floor/null after correction | — | 6 matches x 3 seeds | football | views | 6 models | negative | p1football_*.json | excluded from P1 | generalized inversion |
