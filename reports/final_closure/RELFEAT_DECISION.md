# RELFEAT_DECISION (2026-09-22): RELFEAT_J_SUPPORTED + P3-METHOD

## Gate §14 (relfeat clean vs raw clean, dev bank)
J: s11 0.055->0.131, s23 0.051->0.169, s47 0.055->0.211. 3/3 up, mean
Delta-J +0.11 (dev) / +0.12..0.18 (confirm). Static 0.15-0.17 -> 0.06
(IMPROVED, not degraded). Atomic pass 0.52-0.56 -> 0.68-0.71; AB_end flat;
cond miss 0.89-0.91 -> 0.70-0.81. Gain source: primarily atomic, AB secondary.
Verdict: RELFEAT_J_SUPPORTED. Confirm replication: J 0.16-0.23 vs 0.04-0.06.

## Conditional extension §15 (relfeat+flipmine)
Dev: J 0.38/0.42/0.40; R_full 0.36-0.44 (CI far above flipmine-raw 0.03-0.10);
M 0.32-0.42. Confirm (509 quartets, parent-disjoint): J 0.44-0.47;
R_full 0.45-0.50; M 0.25-0.30. Full repair now EXCEEDS migration (reversed
vs raw-flipmine pattern). Verdict: P3-METHOD.
Claim boundary (§16): explicit label-free relational features substantially
improve full joint consistency; relfeat is hand-designed relation-aware
input, not a general architecture claim. No further relation architectures.
