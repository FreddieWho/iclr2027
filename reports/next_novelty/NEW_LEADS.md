# NEW_LEADS (2026-09-22)
## L1. relfeat joint consistency (exploratory, single-seed)
relfeat J=0.131 vs clean 0.055 on dev bank (s11 only). Relational features
may genuinely help joint consistency. Needs: multi-seed relfeat + matched
ablation (feature-only vs training-only effects). Do NOT claim without it.
## L2. Score-rule quality gain mechanism
Score-rule forced-choice success improves 3/3 seeds (+5/+16/+14pp) while
lexico common-act degrades. What does repair ranking learn? Candidate-level
logit-shift analysis (cheap, no training).
## L3. R_full variance across seeds (0.03-0.10 clean-repair, pilot arms)
Full-repair counts are tiny (3-18 quartets); CIs wide. Any future method
claim on R_full needs larger quartet banks.
## L4. P1-D risk score: negative recorded
Simple confidence saturates fixed-budget capture; fancier scores add nothing
here. Do not reopen without a new setting (e.g., low base-rate regime).
