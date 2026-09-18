# Headline 名单冻结（E1 门槛，2026-09-18 用户拍板）

冻结后任何数字改动须另行授权；E1 只确认本单数字，不开新指标。

## 坐标主发现（FIGURE_SOURCE.json）
- F1-missing：108/128 = 0.844（107 全程零转折），源 N01/round1
- F2-event：条件漏检 23/127 = 0.181，全路径 1/189，对照 FA 0.195，源 N03/round1d
- F3-emergent：条件 62/66 = 0.939（占组合 1.7%），等位移对照 0.653（n=98），源 N04/round1c
- F4-action：lam2 invalid clean 0.770/flipmine 0.683；lam0 三方 ~0.31；lam5 0.935/0.863，源 N06/round1＋round1b
- F5-pixel：miss 0.528（n=53）/0.690（n=71），源 N08/round1b＋round1c
- F6-decomp：flipmine eval_C1 0.417 vs clean 0.574；eval_C0 0.265 vs 0.252，源 N09/round1b
- T1-bracket-dead：v2 积分 s11 0.270 vs 0.229；s23 0.276 vs 0.231
- T2-noise-floor：同配置重跑差 17pp
- T3-fairness：miss@FA≤0.01：concat 0.795 / siamdiff 0.733 / compare 0.782

## 真数据 E2（M_SELECTION.md）
- E2-N01 转折：turn_miss frozen→cover：s11 0.277→0.129 / s23 0.383→0.184 / s47 0.336→0.176；积分与精度全同向
- E2-N03 事件：cond miss frozen→cover：s11 0.255→0.092 / s23 0.362→0.180 / s47 0.347→0.168
- E2-N06-hard 行动：lam2 invalid frozen→cover：s11 0.309→0.102 / s23 0.410→0.184 / s47 0.348→0.141；lam0 三方全 ~0

## E1 确认范围（坐标 only）
上单中坐标可复算项：F1（turn_miss 方向＋量级）、F2（cond miss 方向）、F3（emergent 条件 miss 方向）、F4（lam2 invalid 方向＋flipmine<clean 排序）、F6（C1 方向＋C0 持平）。F5（像素，另训 CNN，不在 E1 内）、T1–T3（方法死亡/元声明，不需 holdout 确认）。
T5R3 侧无 pristine holdout（J03WQQ 用过），E1 不覆盖，真数据侧以 3 种子＋跨场为准，局限如实写。
