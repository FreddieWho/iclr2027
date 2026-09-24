# R07 公平续训比较

状态：30 个真实训练臂完成（raw/rel × seeds 11/23/47 × 5 方法），每臂 300 full-batch epochs，CPU 4 threads；固定 Adam lr=.01，base=1、augmentation mean=1，每臂新增 1,534 个样本。
运行用时 32.3s；dev512 的 237 个 quartet、106 个未见训练 parent（来自 eval_202）用于评估。它是复用开发集，不是新 sealed 确认集。

plain / protection / hard replay / preserve+flip balanced 均从每 seed 同一 clean 参数出发，全部重置 Adam；scratch 独立列为训练路径对照。每轮 base、augmentation 和 replay 的前向数量相同；plain/balanced/scratch 的额外 replay 权重为 0。protection 使用旧正确训练点的 soft teacher，hard replay 使用同组点的 hard label，二者系数均 1。balanced 使用 767 flip + 767 真实 preserve；其他臂使用全部 1,534 flip。总量相等但组成不同，不把 balanced 差异归于纯 loss。

初始/末参数 hash、clean 文件 hash、每轮重复的完整 full-batch 顺序、训练轨迹、逐 quartet 预测和 joint 8×8 迁移均落盘。eval_202 与 train_101 原始坐标逐行 SHA 无重叠；不将多个同 parent quartet 当独立重复。

| 输入 | 方法 | J（3 seeds 平均） | 未见原子 old-correct regression |
|---|---|---:|---:|
| raw | plain | 0.0745 | 0.3180 |
| raw | protection | 0.0689 | 0.2773 |
| raw | hard_replay | 0.0731 | 0.2664 |
| raw | preserve_flip_balanced | 0.0619 | 0.2604 |
| raw | scratch | 0.0647 | 0.3525 |
| rel | plain | 0.3783 | 0.2338 |
| rel | protection | 0.3910 | 0.2143 |
| rel | hard_replay | 0.3896 | 0.2161 |
| rel | preserve_flip_balanced | 0.3910 | 0.1438 |
| rel | scratch | 0.3854 | 0.2328 |

结论：protection 对 plain 的 J 差异 raw 三种子为 −.0295/−.0042/+.0169，rel 为 −.0042/+.0169/+.0253；不是稳定的跨 seed 优势，不能恢复旧 loss 因果解释。未见原子回归约 11–38%，远非训练集 <5pp 可替代，U05 保留为有根据的探索。hard replay 与 soft protection 接近；真实 preserve/flip 组成有降低原子回归的迹象，但需要独立确认，不能归于保护损失机制。

U07 RNG 归因：当前 d01_capacity.py 在每次模型构建前 torch.manual_seed(seed)，aug=none 为确定的 full-batch；本轮没有旧 run 初始 state/batch-stream 证据，RNG 伪影归因仍 UNRESOLVED/撤回。不得只用救好的格替换失败格。本轮是全 30 臂固定合同重训，并不冒充 U07 全 dose 网格重训。

未运行：U07 全剂量格重训、U05 测试原子保护训练、独立 sealed 确认；当前比较不能完成这些科学检验。

命令：`.venv/bin/python experiments/e1a933_review/fair_continuation.py`；统计：`.venv/bin/python experiments/e1a933_review/fair_summarize.py`。
证据：`artifacts/e1a933_review/fair/CONTINUATION_RESULTS.json`、`FAIR_CONTRASTS.json`、各臂 `receipt.json` / `batch_order.json` / `evaluation.npz`。CI 对 parent 成簇、同一次抽样计算双方完整比例；3 个 seed 均值仅描述性汇总。
