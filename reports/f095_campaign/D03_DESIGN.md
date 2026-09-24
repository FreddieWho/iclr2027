# D03｜设计（同起点总体＋候选顺序审计，2026-09-23）

## 数据（三份总体，同一起点配对）
- 父场景：d01fresh401 块（1536 场景，seed401）中新采样 seed 下取用；注：该块被 D01-4N/16N 训练见过，但本路线只复测四臂（r04b＋relflip，train_101 训练）＋D02 平均——对被测模型是未见父场景（如实记录）。
- Bank-O（普通编辑）：模型无关采样编辑＋oracle 标注。
- Bank-E：真实原子保持/联合翻转子集。
- Bank-EA：E 中按种子 raw_clean 原子全对子集（沿用 U1 的 per-seed B110 口径）。
- 同 parent 尽量同时配 preserve＋flip 编辑；配不成记不可配对率，不筛难例。
- 顺序三臂：first-K（原候选顺序前12合法）、random-K（固定种子打乱）、stratified-K（按编辑族分层）；同数量/幅度分布/oracle 查询预算；记 family 占比、y0→y1 方向、面积 margin（不叫距离）。

## 施工
1. 审计层：进入资格禁模型 logit/损失；保留全部尝试＋剔除原因（sampling_flow.csv）。
2. 同一起点、一套 dev 置信阈值：起点误差 / preserve 终点误差 / flip 终点误差 / 双起点正确条件更新风险。
3. parent 等权＋编辑等权双报告（主估计跑前选 parent 等权）；方向分层；有采样概率才 IPW。
4. 三银行 × 四臂＋D02 平均复测（复用 bank npz 格式 Qx/Qe/Qmeta＋既有 eval），**不重训**；仅当顺序效应明确（任一主指标 first-K vs random-K 差超种子噪声带）才加一组随机顺序训练对照。

## 交付
`d03_build_bank.py`、manifest、sampling_flow.csv、分层逐 parent 结果、P1 estimand 表。旧 holdout 不动；新构造 exploratory。
