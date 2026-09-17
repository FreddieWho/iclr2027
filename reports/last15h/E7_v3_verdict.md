# E7 v3 falsification 记录（2026-09-18，积分第一指标，5 种子）

基线 flipmine 本轮：0.248 / 0.248 / 0.259 / 0.218 / 0.235（均值 0.242；历史 0.229 同量级）。

## Batch1：flipmix 全死
- flipmix_u 积分：0.280 / 0.290 / 0.270 / 0.288 / 0.267（均值 0.279，5 种子全输基线）
- flipmix_c 积分：0.289 / 0.286 / 0.261 / 0.268 / 0.265（均值 0.274，5 种子全输基线）
- turn_miss 亦无改善（0.35–0.46 vs 基线 0.38–0.44）；|terr| 反而更差（~0.21 vs ~0.18）
- **判决：B-1 死。**稠密 oracle 标签不如稀疏 flip 标签——监督密度≠位置精度，与 v1 病理（过锐化沟槽）一致。B-2（位置匹配）、B-4（local 配对）失去前提，不再开。

## Batch2a：vanilla-mixup（阴性对照，符合预期）
- 积分：0.391 / 0.345 / 0.311 / 0.383 / 0.349（均值 0.356）；|terr| ~0.27-0.32 全场最差。
- **对照验证通过**：线性先验确与 sharp turn 冲突（B 轨核心论断的负对照成立），mixup 线关闭。

## Batch2b：antiflat（首版协议错误，重跑中）
- 首版犯规：antiflat/sdf 臂漏加 flipmine 基线（clean-only 起跑），且 antiflat_01/10 两 λ 输出 bitwise 一致——诊断：在 bracket 内斜率仅 0.14（< g0=1.0 floor，本应触发惩罚），协议修掉后重跑。
- 旧数（clean-base，仅供参考）：积分 ~0.33-0.36 ≈ clean 水平，无奇迹。

## Batch3a：tangent（首版 detached，作废重跑中）
- 首版 tangent 输出与 flipmine bitwise 一致——根因：slope 经 `float()`＋`torch.no_grad` 双重 detach，梯度从未流过。属代码 bug，非科学结论。
- **新纪律：一切正则臂必须先验梯度流通（grad mass > 0），再谈 falsify。** antiflat/coupled/globalgrad 同病，已修（autograd 图内有限差分 `slope_at_grad`，逐点循环→每路径 2 forward），5 臂重跑中。
- sdf（同 logit MSE，无 slope）不受影响，其 0.38–0.44 数的确是"有害"，待 flip-base 版。

## Batch4a：BAN（有效，死）
- 积分：0.326 / 0.336 / 0.349 / 0.339 / 0.326（均值 0.335 vs 基线 0.242，+9pp）；turn_miss 0.44–0.55，|terr| ~0.28。
- **判决：BAN 死。**自蒸馏（T=2）大幅有害——软目标稀释边界信号，与 C-4 的"高温稀释转折"预测方向一致（但这只证伪了软，不能证低 T 有益；低温 sweep 不再开——主线已够多）。
- （同 batch 的 coupled/globalgrad 为 detach 版 no-op，作废，重跑中。）

## Rerun batch（flip-base）分析：版本号 skew 事件
- antiflat_01：与 flipmine bitwise 一致（旧代码 void，符合预期，已被 b190a1bca 取代）。
- antiflat_10：0.304/0.287/0.288/0.274/0.301（全差于 flipmine）——旧代码下它**必须**等于 flipmine（已由 antiflat_01 证同进程确定性），故它实际跑的是中途落盘的 slope 修复版代码。版本 skew 由矛盾反证确认。
- 教训：长顺序 subprocess 循环＋中途改代码＝版本 skew。b190a1bca 启动后已冻结 e7_v3.py，不再改。
- antiflat_10（修复版，初步）：差于基线，kill-leaning，待 b190a1bca 干净 5 种子确认。
- sdf（flip-base，不受 slope 代码影响，有效）：0.329/0.291/0.318/0.301/0.327（均值 0.313 vs 基线 0.242，5 种子全差）。
- **判决：D-1（同 logit SDF 回归）死。**kill 范围诚实限定：死的是"经分类器 logit 回归距离"，独立 SDF/DSNT 头（D-2）未测，不连坐。

## 终局记分牌（积分均值，5 种子；基线 flipmine 0.242）
- 持平：noise_only 0.241、coupled 0.244
- 全死（差）：flipmix_c 0.274、globalgrad 0.277、flipmix_u 0.279、antiflat_01 0.284、tangent 0.285、antiflat_10 0.291、sdf 0.313、ban 0.335、vanilla_mixup 0.356
- **12 臂 60 训练，无一胜出。flipmine-as-is 是尝试过的局部最优。**

## 诊断量（训后 bracket 内平均斜率）
antiflat_01 5.8 / antiflat_10 3.7 / tangent 4.0 / coupled 44.1（退化）/ globalgrad 2.7——斜率确实拉上去了，积分反更差。
按 A-1 预注册 falsifier：**斜率不是 binding constraint**。\(ds/dt\) 拉上去，转折照样错位——瓶颈在表示或归纳偏置，不在监督。coupled 的 44.1 顺带证实 ratio loss 不稳定（brief 已预警）。

## E7 判决
- v3 网全灭；E7 方法位空缺维持，进 future work（"转折位置需要别的机制——斜率/距离/稠密标签/蒸馏皆败，下一个假设应是表示侧而非监督侧"）。
- 正资产：① flipmine-as-is 的 favorable 对照网（史上最强"旧方法站得住"证据：5 机制＋3 对照全败）；② 三条工程纪律（梯度流通先验、λ sweep 前确认发火、长循环冻结代码）；③ Finlay 引文纠错已记 E7_background.md。
