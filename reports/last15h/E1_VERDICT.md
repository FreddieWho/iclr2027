# E1 开封结论（holdout_909，现铸 fresh，2026-09-18）

源：`artifacts/last15h/E1/holdout_909/result.json`。零训练，固定 checkpoint，单次不间断 run。

## 逐项判定（对照 E1_PREREG.md 判定线）
- F1（turn_miss）：clean 28/64 = 0.438，CI [0.32,0.56]；flipmine 0.313；积分 0.286→0.163。
  → 方向✓，量级为 headline 一半。调查结论（见下）：分母差异，非证伪。
  → flipmine 排序✓（miss 与积分双排序保持）。
- F2（cond miss）：clean 1/24 = 0.042（headline 0.181 在 CI [0.007,0.20] 内）；flipmine 5/29。
  → 方向✓，弱确认（n 小）。FA：clean 0.258（headline 0.195 在 CI 内）/ flipmine 0.023。
- F3（emergent）：clean 条件 151/176 = 0.858（n 大，强）；flipmine 0.773；matched singles 0.610/0.421。
  → ✓ 强确认，全部排序保持（emergent > matched，clean > flip）。
- F4（lam2 invalid）：clean 0.750 / flipmine 0.625；lam0 0.250/0.243。
  → ✓（>0.5 且 flipmine<clean 双成立）。
- F6（C0/C1）：C1 clean 0.524 / flip 0.347；C0 clean 0.262 / flip 0.238（差 2.4pp）。
  → ✓（C1 方向＋C0 ±5pp 内双成立）。
- 静态：clean err 0.180 / flipmine 0.107 / mlpB 0.186。

## F1 量级差调查（预注册失败规则触发项，结论：分母差异）
- headline 分母：`candidates_for_scene` 随机编辑 3072 条 → 仅 128 条单转折（4% 选中率）→ 其中 84% missing。
  随机编辑凑出的单转折多为擦边/切向穿越——正是模型最瞎的形态。
- E1 分母：瞄准式 sweep（朝边界直推到底），64/64 成径——果断穿越，模型跟上率更高。
- 结论：两个数都真，分母不同。headline 0.844 必须限定为"随机编辑单转折路径"；
  瞄准路径 miss 0.44（CI [0.32,0.56]）是同一现象在另一分母下的值。论文措辞按此限定，
  不算证伪（方向＋flipmine 排序＋积分排序三者全保持）。

## 总判
E1 通过（5/5 方向成立，0 方向反转；唯一量级差已归因分母并限定 headline）。
holdout_909 即刻归档为已用封存，后续不再跑新指标（one-shot 原则）。
