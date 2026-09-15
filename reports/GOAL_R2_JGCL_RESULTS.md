# Goal R2（JGCL 单几何）结果：H1 通过，但处处弱于教师系——无大数字

- 日期：2026-09-16；目标第 2 训练轮（台账 2/10）
- 设计：单 InfoNCE（自然对＋Slepian 视图，189 batch 内负例，τ=0.1），
  triplet/teacher/predictor/VICReg/KoLeo/mask 全拆；v6 架构逐位复用；
  210 ctx＋210 InfoNCE 等算力；warm-start＋ramp；emb-std 守卫
- 状态：3/3 TRAINED，无塌缩无 NaN；**H1 PASS**（形式），但锁内"大数字"分支未触发

## 数字（valid）vs v6/R1

| seed | zone | pair | margin | geom | 干预 full | 干预 diag |
|---|---|---|---|---|---|---|
| JGCL-11 | 0.9791 | 0.6367 | 0.0641 | 0.2791 | 0.7236 | 0.4788 |
| JGCL-23 | 0.9771 | 0.6304 | 0.0523 | 0.2441 | 0.6320 | 0.4010 |
| JGCL-47 | 0.9785 | 0.6345 | 0.0470 | 0.2465 | 0.6921 | 0.4167 |
| v6 均值 | 0.979 | 0.84 | 0.16 | 0.54 | 0.61 | 0.54 |
| R1 均值 | 0.974 | 0.83 | 0.16 | 0.56 | 0.68 | 0.54 |

context 侧与 v6/R1 同级健康（zone 0.978、cent 0.037、z_ctx 0.24）；
奇异谱平坦；emb-std 0.78（守卫从未接近触发）。

## 解读：tug-of-war 拆掉了，锐度也没了

1. **InfoNCE 损失深度收敛**（train 0.10 vs chance 4.56）但 pair 只有 0.63、
   margin 0.05——对比目标把 ranking 做"对"了（正例排前），却没把间隔磨"开"。
   triplet 的 margin-seeking 恰恰是锐度的来源：**判别性恰是 InfoNCE 在小样本下
   学不到的东西**（Track G SimCLR 小 batch 教训的实例化）。
2. **几何代理塌到 0.24–0.28**：contrastive 按 pair-membership 组织嵌入空间，
   连续几何关系被打散——margin 体系（triplet）反而保留了序几何。
   机制 insight：序保持需要显式 margin 压力，ranking 正确性不够。
3. **种子方差极小**（pair 极差 0.007）：InfoNCE 收敛到同一解——可复现性好，
   但绝对值天花板低。有趣但非卖点。
4. 干预 0.63–0.72 有限中等——几何→响应通路随几何代理一起变弱，自洽。

## 对目标的影响

- 预注册三分支：大数字分支（margin 锐于 triplet）**未触发**；H1 失败分支未触发；
  塌缩分支未触发。结论：单几何是**有效但全面更弱**的替代——tug-of-war 本体
  不是锐度的敌人，triplet 的 margin 压力才是锐度的来源。
- 训练搜索出现收益递减拐点：R1（mask 非必要）、R2（单几何更弱）连续回答
  "什么不重要"。建议银行剩余 8 轮，转 N4 行文＋论文（见 D-20260916-GOAL06）。
