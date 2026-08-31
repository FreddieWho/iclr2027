# Make Me a Hanzi 受控复现报告

状态：探索性 pilot 已执行

## 技术路线

- 数据：Make Me a Hanzi 的 `medians`、`dictionary.matches`；不使用自然书法图像。
- 部件：`matches` 的顶层 IDS 结构标注；这里的 `semantic_component` 不是字源学语义标签。
- 节点：每个笔画是 5 个有序中线点，首尾点保留，内部点按等弧长采样。
- 特征：坐标、切向、弧位置、首尾标记、局部曲率、笔画顺序；部件标签只用于干预。
- 图：笔画链边 + 跨笔画几何 KNN + 连通性补边；使用对称归一化 Laplacian，方向进入特征而非有向谱。
- 主模型：DeepSets-AE、GAT-AE；ResNet-18 只作为需要本地权重的辅助矢量渲染接口。

## 执行证据

- 字符数：800；节点数范围：20–120。
- 干预行数：182753；有效：180495；边界无效：2258。
- 干预校验：`pass`；最大能量误差：`7.009730573770412e-10`。
- 点集模型状态：[{"model_id": "deepsets_ae_seed11", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "deepsets_ae_seed23", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "deepsets_ae_seed47", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "gat_ae_seed11", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "gat_ae_seed23", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "gat_ae_seed47", "status": "complete", "n_responses": 180495, "device": "cuda", "amp_enabled": true, "intervention_chunk_size": 128, "oom_batch_splits": 0}]
- 运行时：{"requested_device": "cuda", "resolved_device": "cuda", "amp_requested": "on", "amp_enabled": true, "gpu_profile": "none", "cuda_available": true, "gpu_name": "NVIDIA GeForce RTX 3080", "gpu_total_memory_bytes": 10488315904, "cuda_capability": "8.6"}；推理 batch：256；训练 batch：16。
- 优化路径：干预 manifest 与响应均按固定小块流式写盘；按节点数分组的 dense batch、`inference_mode`、可选 CUDA FP16；CUDA 未请求或不可用时保持 CPU 路径。
- 用时：929.91 秒。

## 结果与边界

本报告不把 smoke 或候选曲线写成跨域确认。只有在完整 pilot、字符级 bootstrap 和严格同频对照完成后，才解释 ERRR、频段曲线或 notch 候选。

### ERRR

| model_id           |   n_pairs |   errr_common_gt_semantic |   mean_common_minus_semantic |
|:-------------------|----------:|--------------------------:|-----------------------------:|
| deepsets_ae_seed11 |      9132 |                  0.408016 |                 -0.000241535 |
| deepsets_ae_seed23 |      9132 |                  0.252957 |                 -0.00394354  |
| deepsets_ae_seed47 |      9132 |                  0.466601 |                  0.000691156 |
| gat_ae_seed11      |      9132 |                  0.801029 |                  0.00815159  |
| gat_ae_seed23      |      9132 |                  0.462987 |                 -7.12502e-05 |
| gat_ae_seed47      |      9132 |                  0.577201 |                  0.00257107  |

### 模式可访问性

| model_id           | split   |   n_rows |   mode_rank_response_correlation |
|:-------------------|:--------|---------:|---------------------------------:|
| deepsets_ae_seed11 | heldout |     7300 |                       0.43052    |
| deepsets_ae_seed11 | dev     |     7250 |                       0.421215   |
| deepsets_ae_seed11 | train   |    32835 |                       0.405181   |
| deepsets_ae_seed23 | heldout |     7300 |                       0.475049   |
| deepsets_ae_seed23 | dev     |     7250 |                       0.444421   |
| deepsets_ae_seed23 | train   |    32835 |                       0.444712   |
| deepsets_ae_seed47 | heldout |     7300 |                       0.478231   |
| deepsets_ae_seed47 | dev     |     7250 |                       0.493395   |
| deepsets_ae_seed47 | train   |    32835 |                       0.475169   |
| gat_ae_seed11      | heldout |     7300 |                       0.0854479  |
| gat_ae_seed11      | dev     |     7250 |                       0.060919   |
| gat_ae_seed11      | train   |    32835 |                       0.0590239  |
| gat_ae_seed23      | heldout |     7300 |                       0.0299741  |
| gat_ae_seed23      | dev     |     7250 |                       0.00954824 |
| gat_ae_seed23      | train   |    32835 |                       0.0178078  |
| gat_ae_seed47      | heldout |     7300 |                      -0.0308751  |
| gat_ae_seed47      | dev     |     7250 |                      -0.0570454  |
| gat_ae_seed47      | train   |    32835 |                      -0.0642867  |

### 图表

- `/home/vipuser/codex-jobs/calligraphy-3080-formal-oomfix-20260830/figures/spectral_transfer.png`
- `/home/vipuser/codex-jobs/calligraphy-3080-formal-oomfix-20260830/figures/errr.png`

### 未执行/不确定项

- ResNet-18 若没有本地预训练权重，只能做 `interface_only_untrained` smoke，不能作为科学证据。
- 当前 pilot 使用受控中线渲染，不等同于自然书法风格；局部干预的渲染边界已单独记录。
- 所有结论仍属于发现性候选，需后续严格控制和任务验证。
