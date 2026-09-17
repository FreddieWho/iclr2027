# last15h 终局选择（2026-09-18，10/10 首轮＋全部 round2 开奖）

## 一句话
静态分类正确 ≠ 语义转折正确。六个独立零训练测量收敛到同一事实；五个新方法想法全死，旧 flipmine 是唯一站得住的修复——"试了五种，只有旧方法站得住"如实写进稿。

## 主张选择
- **主发现（lead）**：单转折路径上 clean 模型 84% 全程零转折（108/128，N01 round1；其中 107 全程无模型转折）；匹配转折误差散布无系统偏。
- **切面 2（路径事件，B 发现＋A 主图）**：未见父场景条件漏检 23/127≈0.18（N03 round1d 分池）；全路径逐帧全对 1/189；对照误报 0.195。
- **切面 3（组合协同，现象）**：emergent 翻转（占组合 1.7%）条件 miss 62/66≈0.94；等位移对照 0.653，组合特异 +29pp（N04 round1c）。
- **切面 4（像素复现，现象）**：小 CNN clean 0.906/0.920，像素翻转 miss 0.53（n=53）/0.69（n=71)，可见性门控后（N08 round1b/c）。
- **切面 5（端点比较崩塌，观察）**：classify_compare miss 0.78 vs pair-F1 0.9@FA0.14；FA 对齐后三方无差（N10 round1b）。
- **机制观察**：模型梯度与语义法向 cos≈0 且无判别力（N02）；未见保号编辑误报 ~0.25 全模型一致（N09 round1b；N03 FA 0.195 独立收敛）。

## 方法选择
- **flipmine-as-is**（旧方法，新证据加固）：未见 C1 −16pp（0.574→0.417）、C0 +1pp（N09 round1b）；行动选择 success +9pp 但机制是置信非判断（N06 λc 消融）。
- 新方法全死，死因归档：N01v1/v2（积分输 4pp 双种子）、N02（cos 无判别）、N04 课程（噪声地板内）、N05（种子彩票）、N07（覆盖≈随机）、N09 primal-dual（无冲突可解）、N10-joint（未开：先验不足＋N01v2 已死）。
- 转折位置精度是未解机制，写进 future work（bracket 双侧标签＋方向 margin 均不够）。

## 后果模块
- N06：固定动作库 clean invalid 0.77（λc=2），λc=0 时三方 ~0.31（判断无差），λc=5 时 0.86–0.94（代价压垮）。headline 机制：置信非判断。
- N03 主图：首尾正确截图＋中间事件错开（数据源 N03/round1d）。

## 组合判定
终局 ≈ 组合 A（N01 观察＋N03＋N06），方法位 flipmine-as-is，N09 分解图作方法视角图，N08 作次级像素复现图。B（像素课程）不开（N01v2 已死，连带暂缓）。

## 失败边界（进稿 Limitations）
1. 小 MLP 重训噪声地板 ±15pp（同种子同代码两次运行差 17pp，N04 round2 vs round2b）——一切单种子训练对比不可 headline。
2. emergent 占组合 1.7%（罕见）；N03 全路径 metric 严苛（单帧错即败）。
3. N06 动作库弱（139/256 可行）；λc=2 主数，0/5 为机制消融。
4. 像素翻转分母薄（53/71）；CNN 两种子幅度差。
5. 未碰：holdout_303（留终局确认）、T5R3 版转折（新方法全死后未开，避免烧预算）、J03WQQ（红线如旧）。

## 可运行命令（复现）
```
python3 experiments/last15h/n01_scan.py --out artifacts/last15h/N01/round1
python3 experiments/last15h/n03_events.py --out artifacts/last15h/N03/round1d --pools train_101 eval_202 recheck_404 recheck_505 recheck_606 --n_want 200
python3 experiments/last15h/n04_compose.py --out artifacts/last15h/N04/round1c
python3 experiments/last15h/n06_actions.py --out artifacts/last15h/N06/round1 --lam_cost 2   # 0/5 消融同式
python3 experiments/last15h/n08_visual.py --out artifacts/last15h/N08/round1c --seed 23 --n_scan 800
python3 experiments/last15h/n09_frontier.py --out artifacts/last15h/N09/round1b
python3 experiments/last15h/n10_change.py --out artifacts/last15h/N10/round1b
python3 experiments/last15h/n01_brackets.py --out artifacts/last15h/N01/round2c --v2 --seeds 11 23
python3 experiments/last15h/n04_curriculum.py --out artifacts/last15h/N04/round2b --seeds 11 23 47
python3 experiments/last15h/n05_transfer.py --out artifacts/last15h/N05/round1b --seeds 23 47 11
python3 experiments/last15h/n07_cover.py --out artifacts/last15h/N07/round1
python3 experiments/last15h/n02_angle.py --out artifacts/last15h/N02/round1
```
线程：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`（N08 用 4）。CPU 全程。
