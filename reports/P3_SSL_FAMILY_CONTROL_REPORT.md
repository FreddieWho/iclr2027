# P3 SSL 目标族对照报告：现象跨目标族复现

日期：2026-09-05
状态：完成（control，非 selection；train-only 训练，train/valid 评估，无保留场/外部读取）
产物：`artifacts/phase3/ssl_family_control_v1/`（3 checkpoints + records + summary + manifest + SHA256SUMS）

## 设计

隔离单一变量——**目标族**：保持冻结 2:1 配方的一切（TaskModel 架构、train 数据、
循环流、280/140 配额、80 epochs、batch 64、seeds 11/23/47），只把 intrinsic 的
三元组 margin 损失（contrastive 族）换成 Barlow-Twins 互相关损失
（redundancy-reduction 族，λ=5e-3），同样的 (query, positive) 视图，负样本弃用。
并行化：3 seed worker × 8 线程（线程数记入 manifest）。

## 结果（valid）

| seed | zone F1 | half F1 | cent MAE | pair acc | geometry | z_mode 平移 | 干预 full ρ | 干预 diag ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.9695 | 0.608 | 0.0737 | 0.9518 | 0.7322 | ~0 | 0.636 | 0.407 |
| 23 | 0.9661 | 0.543 | 0.1201 | 0.8771 | 0.7497 | ~0 | 0.669 | 0.574 |
| 47 | 0.9302 | 0.590 | 0.0813 | 0.9174 | 0.7515 | ~0 | 0.467 | 0.374 |

对照对比族 2:1（dev，同协议量级）：geometry ≈0.73；干预 full ρ（ε=0.25）≈0.50–0.60。

## 结论

1. **目标族通用性成立**：Barlow-Twins（无负样本、无 margin）训练出的编码器同样
   表现出 support 敏感、且可由归一化局部几何预测的响应（full ρ 0.47–0.67，
   均值 0.59 ≈ 对比族 0.56）。现象**不是三元组损失的配方假象**。
2. full > diag 三个 seed 一致——耦合结构在 BT 族同样存在（与 P3-R12 一致）。
3. 任务侧指标同量级（zone 0.93–0.97、pair 0.88–0.95、geometry 0.73–0.75）——
   BT 族同样能学到可用的 formation 几何。
4. seed 方差与对比族相当（0.47–0.67），如实报告。

## 诚实边界

- 通用性目前跨**目标族**成立，仍限同一架构（Phase-GAT TaskModel）与同一数据视图；
  架构通用性（EGNN/Set Transformer 类）未测——列入 Limitations/future work。
- 干预集为 dev 40 快照构建（与 ε-sweep 的 60 快照不同实例、同分布），
  量级比较有效，逐值比较无效。

## 对 PLAN 假设的影响

显著强化 P3 机制层的外部效度：support 条件化几何→响应关系是训练后多构件
编码器的一类**跨目标族稳健现象**，为论文回应"这是你训练配方的特产吗"提供直接证据。
