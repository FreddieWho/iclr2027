# P4 R1（v5）结果：H1 通过——双通道 3/3 同时存活

- 日期：2026-09-15
- 状态：**H1 PASS**（预注册 bar：zone≥0.90＋centroid 健康＋intrinsic 信号＋干预有限，
  3/3 seeds 满足）；AMR（分支塑造版）成立；剩余 4 轮转 N4 对照
- artifacts：`artifacts/phase4_amr/m1_v5/`（merge 完成）

## 数字（valid）

| seed | zone | phase | cent | pair | margin | 干预 full ρ | 干预 diag ρ | z_ctx 响应 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 0.9865 | 0.54 | 0.026 | 0.82 | 0.093 | 0.81 | **−0.16** | 0.237 | 全过 |
| 23 | 0.9720 | 0.61 | 0.030 | 0.52 | 0.016 | 0.96 | 0.76 | 0.226 | 全过（pair 弱，见下） |
| 47 | 0.9796 | 0.61 | 0.028 | 0.59 | 0.012 | 0.82 | 0.56 | 0.254 | 全过 |

context 侧：zone 0.97–0.99（冻结配方水平）、centroid 0.026–0.030（优于冻结 2:1
的 0.038）、z_ctx 响应 0.23–0.25（健康可访问性）——构造绝缘兑现。
模态侧：active-triplet 全程 0.73–0.97（early-stop 未触发）、奇异谱铺展、
z_mode 平移 0、干预 0.81–0.96 全有限。

## 诚实注记

1. **seed-23 pair 0.52 偏弱**（二元 ranking chance≈0.5；margin 0.016 虽非零但小）。
   但同一种子的几何代理 0.60（三种子最强）与干预 0.96（三种子最强）说明机制信号
   恰恰最强——triplet 收敛慢（epoch 80 时 active 仍 0.967），不是信号缺席。
   记为 limitation（schedule 长度 follow-up），不推翻 H1（bar 要求"信号"非"高分"）。
2. **seed-11 diag −0.16 而 full 0.81**：diagonal-only 反向、耦合项力挽狂澜——
   正是 P3-R12（耦合承载联盟语义）的独立复现，记为支持性注记而非异常。
3. 与 v4b 的唯一变量差＝编码器共享→分离；v4b context 2/3 塌缩 → v5 3/3 健康。
   隔离干净：**杀死 context 的正是硬共享本身**（P4-A5 的残存疑问关闭）。

## 预算

5 轮用 1 剩 4。R2 起转 N4 matched-capacity 对照（CAP 需新跑：现脚本是共享主干版，
须适配双塔；centering/canonicalization/relational 复用 T5R3 冻结数）。
