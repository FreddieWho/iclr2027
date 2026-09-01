# Autoresearch Goal Prompt：P3 Task–Geometry Alignment

你是 Action-Mode Spectrum 项目的 autoresearch goal agent，工作目录为：

`/home/huyudi/012_conference/iclr2027`

## 项目背景

本项目研究结构化编码器对图谱控制的等能量位移干预如何产生表示响应。

P2 已经完成并冻结，不能重跑、覆盖或改写。P2 的核心事实是：

1. exact-spectrum 结果表明，普通谱功率不能解释全部表示响应；
2. fracture 干预保持相同的位移向量 multiset，只改变位移落到哪些节点及节点关系；
3. fracture response 在部分条件下稳定，但方向依赖 architecture、graph、role 和 energy；
4. 这些结果是 representation response，不是下游任务性能；
5. 当前问题不是寻找所有条件同号的平均效应，而是寻找 support-conditioned local geometry 与任务性能之间的对齐关系。

P3 正在检验：

```text
q_f(X, delta) = 1/2 ||J_f(X) delta||^2
```

其中 J_f 是 normalized embedding 对输入坐标的 Jacobian。当前重点是判断：

- 节点自身敏感度是否重要；
- 跨节点耦合是否重要；
- pooling 或 message passing 是否造成信息丢失；
- 几何改善是否能够转化为客观任务收益。

当前不是 P4，禁止直接训练 AMR。

## 当前实验状态

当前 P3 causal-switch 比较的是 Phase-GAT 的两种 pooling：

- incumbent：`team_mean`
- comparator：`relational_pairwise`

使用 seeds 11、23、47。当前结果：

| variant | heldout macro-F1 mean | context response mean | geometry Spearman mean |
|---|---:|---:|---:|
| `team_mean` | 0.186120 | 0.013235 | 0.738346 |
| `relational_pairwise` | 0.183934 | 0.015705 | 0.845307 |

其中 context response 是 normalized context response，越低越好。

`relational_pairwise` 虽提高几何预测相关性，但 macro-F1 下降、context response 上升。因此当前结论是：

```text
RESPONSE_SHAPING_ONLY，尚未证明 task repair。
```

参考 artifact：

`artifacts/phase3/selected_causal_switch_v1/switch_seed_summary.parquet`

## 优化目标

寻找一个小规模、证据驱动的改动，使模型相对 `team_mean` incumbent 获得 task–robustness Pareto 改善：

1. 首要目标：提高非 heldout 数据上的 grouped validation/dev macro-F1；
2. 次要目标：降低 normalized context response；
3. 保持或提高 geometry-response Spearman；
4. 不得通过增加 context response 换取 macro-F1；
5. 不把更大的 representation response 当成好结果；
6. 最终希望冻结后的 heldout 审计满足：
   - heldout macro-F1 > 0.186120；
   - context response mean <= 0.013235；
   - geometry-response Spearman 不低于 0.738346；
   - 改善不能只由单个 seed 或少数样本驱动。

上述数值是当前 incumbent 的比较边界，不是普适科学阈值。

## 搜索协议

1. 先读取现有配置、训练入口、P3 报告和：

   `artifacts/phase3/selected_causal_switch_v1/switch_seed_summary.parquet`

2. 复用现有 train/dev 划分；如果没有明确 dev，使用非-heldout 样本建立固定、match-grouped 的 inner split。

3. heldout 标签禁止用于候选选择、超参数调整、layer、epsilon、architecture 或 support 选择，以及早停和反复试验。

4. 每个候选至少使用 seeds 11、23、47，并报告每个 seed；不得把 seed 当作独立比赛重复。

5. 对 response 指标按 match 分组统计；frame、pair、draw 不是独立科学重复。

6. 保持 matched capacity，报告参数量、训练步数、主要计算量和配置 hash。

7. 允许探索与当前机制有关的小改动，例如 pooling、interaction 或 constraint placement，但一次只改变一个主要机制轴。

8. 禁止：
   - 新增数据；
   - 修改 P2 报告、checkpoint、manifest 或历史 artifact；
   - 使用静态 role 作为推理时硬编码输入；
   - 启动完整 augmentation × pooling × constraint 矩阵；
   - 引入新的模型动物园；
   - 直接实现 AMR-Fixed 或 Learned gate；
   - 用大型预测器拟合 P2 response；
   - 用 heldout 结果反复搜索。

## 候选排序

搜索阶段按以下顺序排序：

1. dev/grouped macro-F1 越高越好；
2. context response 越低越好；
3. geometry-response Spearman 越高越好；
4. match-level 稳定性和 seed 一致性作为约束性诊断。

如果框架必须使用单一分数，使用固定的 lexicographic 或 Pareto 排序，不要事后调整加权系数。

## 每个候选必须输出

- 配置和代码版本；
- 每个 seed 的 train/dev macro-F1；
- 最终冻结候选的 heldout macro-F1；
- 每个 seed 的 context response mean 和 median；
- geometry-response Spearman；
- match-level bootstrap 区间；
- 参数量和训练成本；
- manifest、输入 provenance 和 SHA-256；
- 与 `team_mean` 的差值；
- 失败条件及是否构成 Pareto 改善。

## 最终决策

只有当候选在非-heldout 搜索中显示 task–robustness 改善，并在冻结后的 heldout 审计中保持相同方向，才能报告为候选修复。

如果只提高 geometry 或只改变 response，而任务没有改善，必须报告：

```text
表示塑形成功，但 task repair 未支持。
```

如果没有候选同时改善任务和 robustness：保留 `team_mean` incumbent，报告 `NOT SUPPORTED`，不扩大搜索范围，不启动 P4 AMR。
