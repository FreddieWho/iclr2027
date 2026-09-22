> 历史 checkpoint 说明：本文件记录 P2 rigid formal v2 闭合时的状态。fracture continuity 已由独立 operator 完成，P2-H1 异质性诊断也已完成；当前状态以 `reports/EXPLORATION_CHECKPOINT_2_FRACTURE.md` 和 `reports/P2_HETEROGENEITY_DIAGNOSIS.md` 为准。

# Exploration Checkpoint 2：刚性联盟干预是否超出普通频率偏置

日期：2026-08-31  
阶段状态：`P2_RIGID_C2_COMPLETE`  
探索性解释：`mixed_or_graph_specific`  
fracture 连续性分析：`NOT_RUN_SEPARATE_OPERATOR`

> provenance 更新：v1 数值保留但标记为 `UNVERIFIED_CODE_PROVENANCE`。权威结果来自重新从零计算的 v2；v2 锁定执行代码和环境，并在 response、statistics、report 前递归验证 manifest→receipt→output 的 size/SHA256。v1 与 v2 的 40 个 matching、90 个 response 和 97 个统计核心 Parquet 均逐文件一致。

## 一句话结论

模型确实会区分“频谱功率完全相同、但空间组织不同”的位移；但“按球员角色定义的联盟总是比同频随机联盟更特殊”并没有稳定成立。结果明显依赖模型架构、probe graph、角色和频段匹配严格度，因此当前最合适的结论是：**存在空间组织/谱相位敏感性，但角色联盟效应是混合且有条件的。**

这不是预注册检验，也没有用固定显著性阈值自动做结论；判断综合了效应方向、10 场比赛、bootstrap 区间、leave-one-match-out、三类模型、两种图和支持集敏感性。

## 实际完成了什么

- 使用全部 250 个 canonical 样本，覆盖 10 场比赛，每场 25 帧；
- 构造 45,600 个 rigid matched sets、182,400 个计划 arm；
- 21,389 个集合获得完整的 semantic anchor 加三类 control；其余逐条保留失败原因，没有插补或记零；
- 对 DeepSets-AE、GAT-small-AE、Phase-GAT 各 3 个 seed，共 9 个冻结模型完成 90 个 response shard；
- 九个模型的 250 样本 baseline parity 全部 `PASS`，最大差为 0；
- 生成 1,184,742 条有效 arm response 和 1,596,348 条 common/pairwise/strict-support 配对记录；
- 先按方向、coalition、team、role、sample 聚合，再以 match 为独立单位；模型 seed 只在 architecture 内平均。

本次全部在本地 CPU 完成。matching 分片累计约 16.8 分钟，模型推理约 7.0 分钟，统计约 11.9 分钟；峰值 RSS 约 1.04 GB，不需要 GPU。

## Matching 是否足以使用

| energy | Delaunay 完整集 | kNN-4 完整集 | 定位 |
|---:|---:|---:|---|
| 0.25 | 4,141 / 5,700（72.6%） | 4,026 / 5,700（70.6%） | 主要低能量 |
| 0.50 | 3,697 / 5,700（64.9%） | 3,613 / 5,700（63.4%） | 主要低能量 |
| 1.00 | 2,611 / 5,700（45.8%） | 2,533 / 5,700（44.4%） | 探索非线性 |
| 2.00 | 393 / 5,700（6.9%） | 375 / 5,700（6.6%） | 压力测试，不进入低能量结论 |

最大能量误差为 `5.02e-14`，exact-spectrum 最大逐模态功率误差为 `0`，没有 graph invalid。主要缺失来自 semantic/control 越界和 topology-frequency 匹配失败。`energy=2.0` 支持率太低，只能作为边界压力测试。

## 主要科学结果

主指标为：

\[
\mathrm{SCG}=\frac{d_z(\mathrm{semantic})-d_z(\mathrm{control})}{\epsilon}.
\]

正值表示 semantic rigid coalition 引起的 embedding 变化更大。下面的范围是低能量 common-four 主分析中，不同 architecture 和 graph 的 match-level 均值范围。

| control | energy 0.25 | energy 0.50 | 解释 |
|---|---:|---:|---|
| arbitrary same-team | `2.7e-6` 至 `8.5e-5` | `-4.1e-6` 至 `1.49e-4` | GAT 两族偏正，DeepSets 近零 |
| topology-frequency matched | `-1.7e-6` 至 `5.0e-5` | `-1.0e-5` 至 `9.8e-5` | 架构和图依赖明显 |
| exact-spectrum sign-randomized | `4.1e-5` 至 `1.35e-4` | `8.4e-5` 至 `2.58e-4` | 12/12 个 architecture×graph×energy 组合均为正 |

### 1. 普通“频率功率”不能解释全部差异

exact-spectrum control 保持每个 Laplacian 模态的功率、总能量和 Rayleigh quotient，只改变谱符号/空间组织。低能量下，三个 architecture、两种 graph、两档 energy 的 12 个组合全部得到正 SCG；match-bootstrap 区间下界全部大于 0，逐场正向比例为 0.8–1.0，leave-one-match-out 后方向仍为正。

因此，当前模型不是只看“各频率有多少能量”；空间排列或谱相位也会影响表示响应。

### 2. 角色联盟并非普遍优于 topology-frequency control

与 topology-frequency matched control 相比：

- GAT-small-AE 和 Phase-GAT 多数设置为正；
- DeepSets-AE 多数接近 0，在 kNN-4 下可转为负；
- Delaunay 通常比 kNN-4 更偏正；
- 将已选 control 再限制到 `band-power L1 ≤ 0.20` 后，低能量仍覆盖每个 graph/energy 约 1,760–2,023 个集合和全部 10 场比赛，但 12 个 bootstrap 区间全部跨过 0。

所以不能把结果概括成“角色联盟具有普遍、架构无关的额外效应”。宽松 `0.30` matching 下的正效应对更严格频段平衡并不稳健。

### 3. role 异质性不是小细节

在 topology-frequency common-four 低能量分析中：

- defender 在 12 个 architecture×graph×energy 组合中均为负；
- midfielder 在 12 个组合中均为正；
- forward 为混合方向。

总体平均会把这些相反模式压在一起。当前“角色”来自静态元数据，不能直接解释为动态战术真值；但它提示下一阶段应研究结构类型差异，而不是只追求一个全局平均数。

### 4. heldout 只提供描述性支持

heldout 只有两场比赛。exact-spectrum 对照在 heldout 的低能量 architecture×graph 汇总均值仍为正，但 topology-frequency 结果继续混合。该结果支持“现象没有只出现在训练比赛”，却不足以提供精确的外推误差。

## 当前允许与不允许的说法

允许：

> 在这些冻结点集模型中，即使逐模态功率完全相同，改变干预的空间组织仍会系统性改变表示响应；但角色定义刚性联盟相对 topology-frequency control 的额外响应依赖架构、图、角色和匹配严格度。

不允许：

- “模型已经理解或忽略了足球战术组织”；
- “角色联盟效应在所有模型中成立”；
- “P1 现象已经完全排除了低频偏置”；
- “SCG 改善了预测性能或任务表现”；
- “fracture operator 或中国书法已经复现同一结果”。

低能量 raw cosine-distance 差值约为 `1e-5` 至 `1.3e-4`，绝对量很小；当前没有下游任务证据说明这一差异具有实际性能意义。

## 风险与下一步

- common-four 是 response-blind 的完整支持子集，但边界缺失仍可能改变可推广对象；
- exact-spectrum control 通常是稠密位移，它严格控制频谱功率，却不控制稀疏支持或局部拓扑，回答的是“谱相位/空间排列是否重要”；
- topology-frequency control 更接近局部联盟对照，但 `0.30` 与严格 `0.20` 子集结果不同；
- 只有 10 场比赛、2 场 heldout，模型是冻结 autoencoder 表示，未测下游任务。

由于结果属于 `mixed_or_graph_specific`，下一条最有信息量的主线是单独实施 fracture continuity：保持非零位移向量 multiset 完全相同，只重新分配端点支持。该分析必须使用新的 operator ID、intervention seed 和独立报告，不能把 P1 fracture 或本次 rigid 结果直接合并。

## 证据路径

- 正式配置：`configs/phase2_rigid_formal_v2.yaml`
- matching manifest：`artifacts/phase2/p2_rigid_formal_v2/matching/matching_manifest.json`
- response manifest：`artifacts/phase2/p2_rigid_formal_v2/response/response_manifest.json`
- 统计 manifest：`artifacts/phase2/p2_rigid_formal_v2/statistics/statistics_manifest.json`
- 主汇总：`artifacts/phase2/p2_rigid_formal_v2/statistics/scg_summary.parquet`
- match-level 结果：`artifacts/phase2/p2_rigid_formal_v2/statistics/match_level_effects.parquet`
- 报告图：`artifacts/phase2/p2_rigid_formal_v2/figures/`
- 实现：`scripts/run_phase2_rigid_formal.py`、`scripts/p2_statistics.py`
- 测试：`tests/test_phase2_rigid_formal.py`、`tests/test_p2_statistics.py`

完成后的 operational 边界：不要对本 run 单独执行部分 stage 的 `--resume`，因为 v2 runner 的顶层 `pipeline_summary.json` 可能被较早 stage 状态重写；核心 manifests/receipts 不会被覆盖，最终 `MANIFEST_SHA256.txt` 可检测 summary 漂移。
