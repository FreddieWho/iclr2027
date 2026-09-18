# E1 预注册（开封只开一次，2026-09-18）

## 对象
现铸 holdout_909（n=512，seed 909，同 generator，min-margin 0.02，平衡 256/256）。
铸后除本 battery 外任何脚本不得读它；holdout_303 保持不动（R05 已碰，不再用）。

## 模型（固定 checkpoint，零训练）
- clean = r04b_s11/clean，flipmine = r04b_s11/flipmine（headline 对）
- mlpB = models/mlpB_h32f16（仅静态基线）

## Battery（协议与原文完全同构，RNG seed 逐项记录）
1. static：512 场干净 acc（三模型）。
2. flips：逐场首个有效翻转（candidates_for_scene，margin≥0.03，R02 口径）→ miss|flip（clean/flipmine）＋n。
3. n06：交叉场 → 48-action 库，lam{0,2} → invalid/success/feasible（clean/flipmine），rng 6。
4. n04：atomic_edits 全协议（atomic_acc、combo 无条件/条件、emergent_n/miss/cond、位移匹配 singles），rng 4。
5. n09：C1=flips（margin≥0.005 重算）＋C0 范数匹配 → C1_miss/C0_FA（clean/flipmine），rng 9。
6. n01：aimed_single_turns（want 64，seed 909）→ turn_miss/integral（clean/flipmine）。预期 ~20 条，宽 CI。
7. n03：build_oscillating（want 128 events）＋build_control（128）→ cond miss/full/FA（clean/flipmine），rng 3。

## 判定线（方向性，预注册）
- F1：clean turn_miss > 0.5（headline 0.844；n 小只看方向＋量级合理＋CI）。
- F2：clean cond miss > 0（headline 0.181）。
- F3：clean emerg cond miss > 0.5（headline 0.939）。
- F4：lam2 invalid clean > 0.5 且 flipmine < clean（headline 0.770/0.683）。
- F6：C1_miss flipmine < clean 且 C0 差 ±5pp 内（headline 0.417/0.574，0.265/0.252）。
- 失败规则：方向反转＋CI 不重叠 → 启动调查（holdout quirk vs eval-overfit），不自动判论文死刑；n 小导致的宽 CI 如实报，不硬拗。

## 不测项
F5（像素，另训 CNN）、T1–T3（死亡/元声明）、T5R3 侧（无 pristine holdout）。
