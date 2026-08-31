# 书法受控 probe：计算优化 checkpoint

日期：2026-08-29

## 状态

优化代码已完成并通过本地验证；正式 800 字符计算尚未启动，等待用户提供具体 GPU 算力地址。

本次只优化执行路径，不改变字符划分、五点笔画表示、固定基线图/谱、干预定义、能量误差定义或两类主模型的结构。

## 已完成的优化

- 按节点数分组做 dense batch，减少逐字符/逐干预 forward。
- 干预数组保留为运行时缓存，公开 parquet 仍只保存 JSON 审计字段；写盘前清除 `DataFrame.attrs`，避免 ndarray 混入 parquet 元数据。
- 推理使用 `torch.inference_mode()`，且只运行 encoder，不运行不参与响应计算的 decoder。
- DeepSets 推理/训练不再分配无用邻接矩阵；GAT 保留固定基线邻接。
- 支持 `--device auto|cpu|cuda`、`--amp auto|on|off`、`--gpu-profile none|v100`、推理/训练 batch 参数。
- 显式请求 CUDA 但不可用时直接失败，不静默改跑 CPU。
- V100 AMP 路径中 attention logits、mask、`log(adjacency)` 和 softmax 保持 FP32；模型张量可使用 FP16。训练 loss 在 FP32 中归约。
- 默认训练 logical batch 在 CPU/CUDA 都保持 16，避免为提速悄然改变 Adam 更新轨迹；CUDA 默认只放大推理 batch 到 256。
- 响应表保留 `intervention_id` 和 `pair_id`，便于检查完整性。

## 本地验收

环境：`torch 2.11.0+cu130`，本机 `torch.cuda.is_available() == false`，因此没有伪造 V100 实测。

优化版 smoke：24 字符、1 seed、2 epochs、CPU、推理 batch 64、训练 batch 16。

| 指标 | 优化前参考 smoke | 优化后 smoke |
|---|---:|---:|
| 用时 | 148.76 s | 88.12 s |
| 相对用时 | 1.00× | 0.59× |
| 加速比 | — | 1.69× |
| 干预行数 | 2,689 | 2,689 |
| 有效干预 | 2,659 | 2,659 |

优化后干预校验为 `pass`，最大能量误差 `5.22e-10`。将旧/新响应按样本、干预类型、能量、seed、部件和 mode 对齐后：

- DeepSets 最大 `response_distance` 绝对差 `8.57e-08`；
- GAT 最大 `response_distance` 绝对差 `5.96e-08`；
- 所有响应行的干预键集合一致，优化后每个模型的 `intervention_id` 唯一；
- 9 个定向测试通过。

当前 smoke 的简单线性外推为约 30.6 小时，仅作成本预警，不是正式耗时承诺。该外推不能替代 GPU 实测。

## V100 启动配置

拿到算力地址并确认环境后，先运行小 smoke 做 CUDA/FP32/AMP 对照：

```bash
env LD_LIBRARY_PATH=/opt/anaconda3/lib PYTHONPATH=. \
python scripts/run_calligraphy_probe.py \
  --smoke --device cuda --gpu-profile v100 --amp on \
  --batch-size 256 --train-batch-size 16 \
  --out artifacts/calligraphy_pilot/v100_smoke
```

若 AMP 与 CPU FP32 的响应偏差超过项目约定阈值，正式运行应改为 `--amp off`；不能放宽科学阈值迁就 FP16。V100 smoke 通过后再评估是否启动正式 800 字符任务。

## 尚未完成/明确阻塞

- 当前机器没有 CUDA，V100 吞吐、显存峰值和 AMP 数值等价尚未实测。
- 当前环境缺少 `torchvision`；ResNet-18 仍是辅助分支阻塞状态，不能用随机权重代替科学证据。
- 本 checkpoint 不表示正式 pilot 已完成，也不把 smoke 的 ERRR 或频段曲线当成跨域确认。
- 本次没有访问或更新 `infra/bioinf-data-index/`，项目与生信工作流保持隔离。

证据目录：[artifacts/calligraphy_pilot/optimized_smoke](../artifacts/calligraphy_pilot/optimized_smoke)

