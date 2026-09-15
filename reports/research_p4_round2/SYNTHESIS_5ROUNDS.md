# Research 综合 2：D/E/F → 5 轮预算映射（2026-09-15）

来源：`reports/research_p4_round2/` D 软共享 / E 平衡＋两阶段 / F 成功先例。
综合原则：与 15 次失败证据的相容性 × 证据强度 × 预算经济性。

## 跨轨道收敛

1. **方向性冲突，不是尺度问题**（E）：GradNorm/uncertainty-weighting 按构造
   在本 regime 失败；v4（parity 放大）与 v4b（降权 warm-start）双失败与此一致。
   CAGrad > PCGrad（E P2），但两者都只做梯度手术，不碰表征冲突。
2. **单向绝缘让 context 健康变成构造性**（D）：双塔＋路由限制在 mode 侧＋
   stop-grad，context 健康不再靠运气。H1 的 context bar 从此是实现检查，不是抽奖。
3. **成功先例的共同点全是我们没试过的**（F）：momentum teacher、centering＋
   sharpening 输出端平衡、aux 权重 0.03–0.1＋延迟 40%、sequential fallback。
   但 F 的最小成功尺度（CIFAR-ViT、数百 epoch）远大于我们——直接照搬风险高，
   列为 R3＋ 候选。
4. **永远不要阶跃进入**（E）：ramp 已是 v4b 验证过的正确做法，保留。

## 5 轮映射（H1 bar 不变；通过即停以省预算）

- **R1（v5，本轮）：全解耦双塔**（D Design-1 lite）。ctx 塔（raw＋冻结 context 损失）
  与 mode 塔（centered＋v4 全套机器）零共享参数；与 v4b 的唯一变量差 = 编码器
  共享→分离。故意**不做 gate/bidirectional stitch**：冻结评估接口
  （pair_scores 只传 band feats）要求 train/eval 同构；gate 的"学习共享"问题
  延至 R3＋（R1 通过后的 follow-up）。若 R1 通过 H1 → 成功，剩余轮次转 N4 对照。
- **R2（v6）：Freeze-then-route M2-lite**（E P1）。冻结健康主干（T5R-2:1 或 v4b
  warm-start checkpoint），只训路由头。仅 R1 失败才启动；回答"路由本身是否有价值"。
- **R3 候选（RI 通过/失败后二选一）**：CAGrad 联合救援（E P2，c=0.2）或
  momentum-teacher＋predictor（F #1，τ 0.99→1.0）。由 R1/R2 死因决定。
- **R4–R5**：胜者精化 / 消融 / N4 对照（CAP 等），或提前停止省预算。

## R1 判定逻辑（预注册）

- 通过 H1 ⇒ 路由在受保护分支上有效 ⇒ AMR 方法成立（分支塑造版），进 N4。
- mode 仍塌缩 ⇒ 路由目标本身 broken（连受保护分支都救不活）⇒ 转 R2（测路由头价值）。
- context 异常（理论上不可能，构造绝缘）⇒ 实现 bug，修 bug 不计入轮次。
