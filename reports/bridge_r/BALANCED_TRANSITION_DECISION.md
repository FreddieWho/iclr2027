# Balanced Transition Decision: DINO_REPAIR_NEGATIVE — STOP (2026-09-20)

## Gate check (corrected full-data rerun)

- P1 (balanced atomic ≥0.85, ≥2/3 seeds): 0.689/0.722/0.631 → FAIL.
- P2–P4 moot: Layer-1 prerequisite failed; composition numbers are
  uninterpretable by protocol §23.

## Verdict

**`DINO_REPAIR_NEGATIVE`** — balanced transition supervision cannot
install even atomic competence on frozen DINOv3-L R0, let alone unseen
compositional updating. Foundation repair branch stops here. No LoRA,
no finetuning, no adapter/layer search (§30–§31). No Phase 2 (no MLP/CNN
re-runs, §48). Paper body untouched. Holdout 895 still sealed.

(Supersedes the ed8b7e0 numbers, which were trained on one shard due
to a `load_pairs` bug; the verdict is unchanged on valid data.)

## Ten questions (§50, 通俗中文)

1. atomic 能做起来吗？不能——最好单次 0.760（static s11），门限 0.85 全败。
2. static 多少？0.76/0.71/0.70，稳定但不够。
3. flip-only 多少？0.63/0.64/0.68，最差。
4. balanced 多少？0.69/0.72/0.63，最好但远不够。
5. unseen A+B 谁最好？前提不成立，无人及格，无法排名。
6. flip-only 过度敏感了吗？有方向性迹象（FA 0.16 vs static 0.11，M_single 同步下降），但因无人及格，按协议不许解读为结论。
7. balanced 做到"该变则变、不该变则守"了吗？没有。
8. adaptation 还是 composition repair？连 adaptation 都没成立。
9. 值得重构 MLP/CNN repair 吗？不值得。
10. verdict？**DINO_REPAIR_NEGATIVE**。
