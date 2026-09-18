# B01 Bridge pilot 报告（frozen foundation + probe，2026-09-18）

问题：headline 85.8% 是否只是小模型 artifact？ frozen DINOv2-B / CLIP ViT-B
+ probe 在同一 conditional metric 下还剩多少？

## 设计
- 224px nuisance-locked 渲染（S0风格，同quartet同seed新鲜RNG），与 u03b_fixed 同源
- frozen backbone（零训练），probe 在 230 train-quartets 上训，200 eval quartets 上测
- 预注册判据：atomic≥0.90 才展开全矩阵

## 结果
| backbone | probe | atomic | cond_n | cond_miss |
|---|---|---|---|---|
| DINOv2-B | LR | 0.850 | 149 | 0.195 |
| DINOv2-B | LR-CV/scale | 0.845 | 147 | 0.245 |
| DINOv2-B | MLP-128 | 0.865 | 156 | 0.288 |
| DINOv2-B | MLP-256-128 | 0.858 | 153 | 0.314 |
| CLIP ViT-B | LR | 0.790 | 131 | 0.351 |
| CLIP ViT-B | LR-CV | 0.805 | 137 | 0.365 |
| CLIP ViT-B | MLP-128 | 0.798 | 134 | 0.373 |
| CLIP ViT-B | MLP-256-128 | 0.805 | 137 | 0.431 |

## 解读
1. **Scale 极大缓解（85.8%→20-35%），但没有消除。** 两 backbone 同向残余，MLP
   probe train_acc=1.0 而 test atomic 不动 → 瓶颈在 frozen 表示侧，不在读出容量。
2. **未达展开线。** atomic 最高 0.865 < 0.90 go-bar，全矩阵不开。严格说这是 NOGO。
3. **OOD 保留意见。** 抽象线段图对 foundation 系远分布；但 conditional 设计下
   atomic 80%+ 已成立，残余 miss 不能拿 OOD 一笔勾销——部件能读，更新仍漏。
4. 定位：Level 2→2.5 证据（"scale helps a lot, does not eliminate"），够不上 Level 3；
   标题降级仍要做。

Artifacts: `artifacts/bridge/b01_{dinov2,dinov2C,clip}/{result,feats}.json.npz`
（dinov2 首跑无 feats 缓存；dinov2C 同 seed 重提特征一致）。
脚本：`experiments/bridge/b01_pilot.py`（渲染+frozen提特征+LR），`b02_probe.py`（缓存特征上 CV/MLP）。
环境注记：需 `LD_LIBRARY_PATH=/opt/anaconda3/lib`（系统libstdc++缺GLIBCXX_3.4.29）；
预处理手工 resize+normalize（transformers processor 需 torchvision，未装）。
