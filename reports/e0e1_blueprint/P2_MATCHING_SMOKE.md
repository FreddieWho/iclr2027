# P2-A/B matching-only smoke

日期：2026-08-31  
状态：`INCOMPLETE_MATCHING_ONLY`；构造约束校验通过，但匹配完整性不足；P2 模型推理和科学解释仍为 `NOT_RUN`

## 运行范围

本轮只验证 P2 的干预构造、匹配和审计链，不加载 checkpoint，不训练模型，不租用或访问 GPU。输入为 P1 已存在的 3 个足球点集样本；使用 P1 weighted kNN4 和 weighted Delaunay 两个 probe graph，四个能量（0.25、0.5、1.0、2.0）和四个固定方向。

干预臂包括：

- `semantic_rigid`：角色组内同队节点刚性同向平移；
- `arbitrary_same_team`：同队、同支持大小的任意控制；
- `topology_frequency_matched`：通过拓扑/频谱 caliper 的同队控制；
- `spectrum_exact_sign_randomized`：逐 eigenmode 功率保持、符号随机化控制。

## 工程结果

| 项目 | 结果 |
|---|---:|
| 样本 | 3 |
| matched-set | 576 |
| 干预臂 | 2304 |
| 有效行 | 1378 |
| 显式无效/未匹配行 | 926 |
| 四臂完整 matched-set | 98 |
| 图构造失败 | 0 |
| 有效行最大能量误差 | `7.1e-15` |
| exact-spectrum 最大逐模态功率误差 | `0` |

无效行全部保留，并写明原因；主要原因是高能量平移越过坐标边界，以及 topology/frequency caliper 下没有合格候选。它们不是被删除的数据。

## 解释边界

本轮只能说明 P2-A/B 的代码和数据契约可运行，能量约束和 exact-spectrum 约束通过；不能说明组织效应存在、SCG 为正、模型响应一致或 C2 成立。首轮完整匹配率为 `98/576`，是后续正式推理前必须关注的覆盖率信号，尤其要单独检查 `epsilon=2.0` 和严格 caliper 的影响。

下一步是 P2-C：单个 DeepSets checkpoint 的响应 smoke。P2-D 的 250 样本、10 场比赛、9 个模型正式运行尚未启动；在 P2-C 实测 wall time/RAM 前不租用 GPU。

## 产物

- [pipeline_summary.json](../artifacts/phase2/smoke_matching/pipeline_summary.json)
- [matching_validation.json](../artifacts/phase2/smoke_matching/matching_validation.json)
- [matched_intervention_manifest.parquet](../artifacts/phase2/smoke_matching/matched_intervention_manifest.parquet)
- [matched_set_manifest.parquet](../artifacts/phase2/smoke_matching/matched_set_manifest.parquet)
- [matched_set_summary.parquet](../artifacts/phase2/smoke_matching/matched_set_summary.parquet)
- [matching_balance.parquet](../artifacts/phase2/smoke_matching/matching_balance.parquet)
- [graph_manifest.json](../artifacts/phase2/smoke_matching/graph_manifest.json)

本轮未读取、更新或重建 `infra/bioinf-data-index/`，也未引入任何生信数据。
