# P4-N4 v4 结果：模态通道 3/3 学会，context 通道被路由反噬（H1 未通过）

- 日期：2026-09-15
- 状态：v4 未通过 H1（context 侧 2/3 塌缩、1/3 侵蚀），触发预注册 Route-3 备用（v4b）；
  这不是 M1 死刑——模态通路 9 个失败种子后首次 3/3 存活
- artifacts：`artifacts/phase4_amr/m1_v4/`（merge 完成）

## 数字（valid）

| seed | zone | phase | cent MAE | pair | margin | 干预 full ρ | 干预 diag ρ | z_ctx 响应 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 0.13 | 0.18 | 1.13 | 0.65 | 0.005 | 0.93 | 0.32 | 0.006（恒定） | context 塌缩 |
| 23 | 0.57 | 0.28 | 0.19 | 0.86 | 0.076 | 0.82 | 0.64 | 0.116 | context 侵蚀 |
| 47 | 0.11 | 0.41 | 3.57 | 0.63 | 0.024 | 0.99 | 0.54 | 0.000（恒定） | context 塌缩＋head 发散 |

模态侧健康指标：active-triplet 全程 >0（early-stop 未触发）、奇异谱铺展
（top5 如 94/71/42/39/25，无零秩）、z_mode 平移响应 0.0000（构造不变性 intact）、
vicvar ~1e-3（floor 满足）、μ≈0.34–0.37。

## 解读：共享主干竞争的反转

- v1–v3：路由压力（或其对称结构）杀死模态通道。
- v4：W_ROUTE=12（init parity 校准）下路由赢得主干，context 通道被侵蚀/杀死。
- 同一竞争主题（P3-R7/R9）的反向实现，且被量化：12× 放大足以让路由吃掉 context。
- 模态侧干预 full ρ 0.82–0.99 是全项目 dev 最强几何→响应预测——Slepian＋精确谱＋
  VICReg 机器的有效性得到证明，问题纯粹是分配权重，不是机制无效。

## H1 判据

FAIL：context 非劣性不满足（2/3 塌缩）。按 D-20260905-P4-007 的备用条款，
开 v4b（Route-3：W_ROUTE 12→2 ＋ 20 epoch route-free warm-start ＋ 20 epoch ramp，
其余冻结）。v4b 若仍失败 → M1 存废交用户，不再尝试。
