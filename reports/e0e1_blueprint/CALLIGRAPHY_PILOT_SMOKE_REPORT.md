# Make Me a Hanzi 受控复现报告

状态：smoke，仅接口和小样本核验

## 技术路线

- 数据：Make Me a Hanzi 的 `medians`、`dictionary.matches`；不使用自然书法图像。
- 部件：`matches` 的顶层 IDS 结构标注；这里的 `semantic_component` 不是字源学语义标签。
- 节点：每个笔画是 5 个有序中线点，首尾点保留，内部点按等弧长采样。
- 特征：坐标、切向、弧位置、首尾标记、局部曲率、笔画顺序；部件标签只用于干预。
- 图：笔画链边 + 跨笔画几何 KNN + 连通性补边；使用对称归一化 Laplacian，方向进入特征而非有向谱。
- 主模型：DeepSets-AE、GAT-AE；ResNet-18 只作为需要本地权重的辅助矢量渲染接口。

## 执行证据

- 字符数：24；节点数范围：35–105。
- 干预行数：2689；有效：2659；边界无效：30。
- 干预校验：`pass`；最大能量误差：`5.215815557235715e-10`。
- 点集模型状态：[{"model_id": "deepsets_ae_seed11", "status": "complete", "n_responses": 2659, "device": "cpu", "amp_enabled": false, "intervention_chunk_size": 128, "oom_batch_splits": 0}, {"model_id": "gat_ae_seed11", "status": "complete", "n_responses": 2659, "device": "cpu", "amp_enabled": false, "intervention_chunk_size": 128, "oom_batch_splits": 0}]
- 运行时：{"requested_device": "cpu", "resolved_device": "cpu", "amp_requested": "off", "amp_enabled": false, "gpu_profile": "none", "cuda_available": false, "gpu_name": null}；推理 batch：64；训练 batch：16。
- 优化路径：干预 manifest 与响应均按固定小块流式写盘；按节点数分组的 dense batch、`inference_mode`、可选 CUDA FP16；CUDA 未请求或不可用时保持 CPU 路径。
- 用时：12.22 秒。

## 结果与边界

本报告不把 smoke 或候选曲线写成跨域确认。只有在完整 pilot、字符级 bootstrap 和严格同频对照完成后，才解释 ERRR、频段曲线或 notch 候选。

### ERRR

| model_id           |   n_pairs |   errr_common_gt_semantic |   mean_common_minus_semantic |
|:-------------------|----------:|--------------------------:|-----------------------------:|
| deepsets_ae_seed11 |        91 |                  0.692308 |                   0.00070293 |
| gat_ae_seed11      |        91 |                  0.923077 |                   0.00136281 |

### 模式可访问性

| model_id           | split   |   n_rows |   mode_rank_response_correlation |
|:-------------------|:--------|---------:|---------------------------------:|
| deepsets_ae_seed11 | heldout |      236 |                         0.480638 |
| deepsets_ae_seed11 | dev     |      221 |                         0.474721 |
| deepsets_ae_seed11 | train   |      864 |                         0.391896 |
| gat_ae_seed11      | heldout |      236 |                         0.510433 |
| gat_ae_seed11      | dev     |      221 |                         0.454989 |
| gat_ae_seed11      | train   |      864 |                         0.33186  |

### 图表

- `/tmp/iclr2027_calligraphy_oom_smoke_current/figures/spectral_transfer.png`
- `/tmp/iclr2027_calligraphy_oom_smoke_current/figures/errr.png`

### 未执行/不确定项

- ResNet-18 若没有本地预训练权重，只能做 `interface_only_untrained` smoke，不能作为科学证据。
- 当前 pilot 使用受控中线渲染，不等同于自然书法风格；局部干预的渲染边界已单独记录。
- 所有结论仍属于发现性候选，需后续严格控制和任务验证。
