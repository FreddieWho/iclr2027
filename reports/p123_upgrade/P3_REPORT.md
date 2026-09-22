# P3_REPORT (2026-09-23): paired repair comparison changes the conclusion

## Paired same-quartet bank (dev512, 237 quartets, s11/s23/s47 x clean/flipmine)
Four cells by atomic-pass status; AB paired changes within cells:
- both (71-75): clean ABerr 0.92-0.93 vs repair 0.82-0.95. s23 REVERSES
  (0.920 -> 0.947, repair worse on the common set).
- clean_only (51-57): clean ABerr 0.84-0.89, repair 0.05-0.06 (fixed 40-48,
  broken 0). Repair nails AB while failing its own atomics.
- repair_only (16-33): clean ABerr 0.0-0.06, repair 0.69-0.80 (broken 20-22).
- neither (72-98): both ~0-5%.
Denominators: own_clean 0.894-0.909 (n=123-132) vs own_repair 0.793-0.870
(n=87-108); baseline-fixed repair-on-clean-set 0.496-0.561; common
0.916-0.931 vs 0.817-0.947 (direction inconsistent across seeds).
Overall AB err: clean 0.48-0.52 vs repair 0.32-0.43; atomic pass drops
0.52-0.56 -> 0.37-0.46.
Unit test (synthetic, identical AB predictions, atomics vary): own-miss
1.00 vs 0.25 — denominator-only movement demonstrated.

## Reading
The repair's apparent gain is substantially denominator change (who counts)
plus a boldness shift (AB right while atomics wrong), not same-sample update
fixing. Common-correct shows no stable gain with one reversal. This is a
substantively different repair conclusion from the headline framing, obtained
without new training.

## Literature (NOVELTY_MATRIX.md)
"First conditional" is dead (Press 2022; Composition Collapse 2026 double-gate).
Defensible delta: controlled-change update protocol + denominator discipline +
repair decomposition + consequence chain.
