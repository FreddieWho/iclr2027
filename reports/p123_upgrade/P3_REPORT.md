# P3_REPORT (2026-09-23): paired repair comparison on one quartet bank

Bank bank_dev512: 237 quartets, frozen inputs+hashes, 6 models
(s11/s23/s47 x clean/flipmine). No retraining, no resampling.

## Four cells (atomic-pass status -> AB paired changes)
| seed | both (n) | clean_only (n) | repair_only (n) | neither (n) |
|---|---|---|---|---|
| s11 | 72: AB 0.931->0.847 (fix 7, break 1) | 51: 0.843->0.059 (fix 40) | 25: 0.000->0.800 (break 20) | 89: ~0.05->~0.0 |
| s23 | 75: 0.920->0.947 (fix 3, break 5) | 57: 0.895->0.053 (fix 48) | 33: 0.030->0.697 (break 22) | 72: ~0.05 |
| s47 | 71: 0.916->0.817 (fix 11, break 4) | 52: 0.865->0.058 (fix 42) | 16: 0.062->0.688 (break 10) | 98: ~0.04 |

## Triple denominators (AB error)
- own-vs-own: 0.89-0.91 -> 0.79-0.87 (denominators 123-132 vs 87-108).
- baseline-fixed (clean set): 0.89-0.91 -> 0.50-0.56. Genuine paired
  repair effect ~35-40pp, 3/3 seeds. This is the number the headline
  85.8%->77.3% was gesturing at.
- common-correct: s11 0.931->0.847, s23 0.920->0.947 (slight reversal),
  s47 0.916->0.817. No sorting reversal that matters.
- overall AB err: 0.48-0.52 -> 0.32-0.43; atomic pass: clean 0.52-0.56 vs
  repair 0.37-0.46 ON THIS BANK (preserve-only atomics; cf. fresh E1 where
  repair improves all-edit atomics 0.81->0.88 — population-dependent,
  reported as open discrepancy, not a claim).

## Reading
Repair does two things at once: (a) genuine AB fixes on stable quartets
(~35pp paired), (b) denominator churn — exits hard quartets (loses their
atomics) and enters quartets where AB then fails. Own-vs-own mixes both;
baseline-fixed isolates (a). No model sorting change: repair wins on all
three denominators (s23 common trivial reversal excepted).

## Unit test (synthetic, metric property)
Identical AB predictions + different atomic sets -> own-rates 1.00 vs 0.40;
baseline-fixed invariant. Recorded in P3_PAIRED.json.
