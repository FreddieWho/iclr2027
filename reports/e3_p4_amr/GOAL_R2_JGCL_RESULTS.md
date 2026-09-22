# Goal R2（JGCL 单几何）结果：H1 通过，但全面弱于教师-triplet——无大数字

- 日期：2026-09-16；目标第 2 训练轮（台账 2/10）
- 设计：单 InfoNCE（自然对＋Slepian 视图正例，189 batch 内负例，τ=0.1），
  triplet/teacher/predictor/VICReg/KoLeo/mask 全拆；v6 架构逐位复用；
  210 ctx＋210 InfoNCE 等算力；warm-start＋ramp；emb-std 守卫
- 状态：3/3 TRAINED，无塌缩无 NaN；**H1 PASS（形式）**，但大数字分支未触发

## 数字（valid）vs v6/R1

| seed | zone | pair | margin | geom | 干预 full | 干预 diag |
|---|---|---|---|---|---|---|
| JGCL-11 | 0.9791 | 0.6367 | 0.0641 | 0.2791 | 0.7236 | 0.4788 |
| JGCL-23 | 0.9771 | 0.6304 | 0.0523 | 0.2441 | 0.6320 | 0.4010 |
| JGCL-47 | 0.9785 | 0.6345 | 0.0470 | 0.2465 | 0.6921 | 0.4167 |
| v6 均值 | 0.979 | 0.84 | 0.16 | 0.54 | 0.61 | 0.54 |
| R1 均值 | 0.974 | 0.83 | 0.16 | 0.56 | 0.68 | 0.54 |

context 侧同级健康（zone 0.978、cent 0.037、z_ctx 0.24）；InfoNCE 深度收敛
（0.10 vs chance 4.56）；emb-std 0.78（守卫从未接近触发）；奇异谱平坦。

## 解读：tug-of-war 拆掉了，锐度也没了

1. **H1 通过但弱**：pair −0.2、margin −0.11、几何代理 −0.3（vs v6/R1）；
   仅干预 full 打平（0.68 vs 0.61–0.68）。锁内"大数字"分支（margin 锐于
   triplet 系）**未触发**。
2. **机制结论**：对比目标把 ranking 做"对"了（正例排前，loss 0.10），却没把
   间隔磨"开"——triplet 的显式 margin 压力才是锐度的来源，
   InfoNCE 的 ranking 平均化给不出 margin。这是 Track G 小-batch SimCLR
   教训的实例化，有机制价值但属"什么不行"类结论。
3. **种子方差极小**（pair 极差 0.007，R1 同类现象复现）：单几何收敛到同一解——
   可复现性好、天花板低。有趣但非卖点。
4. 几何代理塌到 0.24–0.28：contrastive 按 pair-membership 组织空间，连续几何
   关系被打散——与 margin 结论自洽。
