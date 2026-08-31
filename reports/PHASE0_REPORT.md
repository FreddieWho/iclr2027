# Phase 0 Report — Data and Action-Mode Probe Smoke

## 1. 状态

- **阶段**：P0_BOOTSTRAP
- **状态**：completed
- **研究模式**：exploratory_discovery
- **数据域**：SkillCorner football tracking
- **比赛**：`1886347`
- **随机种子**：`20260827`

本报告只确认数据接口、干预生成和 embedding 流水线可运行，不对任何 Action-Mode 现象作科学结论，也不把探索假设当作预先承诺。

## 2. 已完成的计算

1. 从 `1886347_tracking_extrapolated.jsonl` 选择覆盖率最高、两队各 10 名的 20 个外场球员。
2. 从同一场比赛抽取 1,000 个等间隔且节点完整的合法帧。
3. 对每帧构建 4-nearest-neighbor 对称关系图和归一化 Laplacian。
4. 对每帧计算 20 个谱模态，并划分为 6 个频段。
5. 为每个样本生成 10 条干预：
   - `common`
   - `single`
   - `semantic_coalition`
   - `random_coalition`
   - `band_0`–`band_5`
6. 用 40 epoch 的 tiny coordinate autoencoder 生成 32 维坐标 embedding。
7. 使用 ImageNet-1K 预训练、冻结的 torchvision ResNet-18 对 1,000 张 minimap 提取 512 维视觉 embedding。
8. 生成六帧样本图和第一阶段 minimap 图。

## 3. 结果与质量检查

| 检查项 | 结果 |
|---|---:|
| canonical samples | 1,000 |
| 节点数/样本 | 20 |
| intervention records | 10,000 |
| spectrum sanity records | 6,000 |
| 频段数 | 6 |
| 最大等能量误差 | `4.44e-16` |
| 最低目标频段纯度 | `1.00`（浮点误差范围内） |
| 坐标 embedding | `(1000, 32)`，全为有限值 |
| 冻结视觉 embedding | `(1000, 512)`，全为有限值 |
| 图对称性与 Laplacian 对称性 | PASS |
| 重复生成契约单测 | 4/4 PASS |

数据 manifest 验证中，两个 P0 必需数据集 `skillcorner` 和 `makemeahanzi` 为 OK；其他已下载数据按 optional/fallback 角色记录。

## 4. 产物

- `artifacts/phase0/canonical_samples.npz`
- `artifacts/phase0/canonical_samples.jsonl`
- `artifacts/phase0/intervention_manifest.parquet`
- `artifacts/phase0/spectrum_sanity.csv`
- `artifacts/phase0/sample_grid.png`
- `artifacts/phase0/phase0_first_minimap.png`
- `artifacts/phase0/coordinate_embedding.npy`
- `artifacts/phase0/coordinate_autoencoder.pt`
- `artifacts/phase0/frozen_resnet18_embedding.npy`
- `artifacts/phase0/pipeline_summary.json`
- `data/checksums/verification.json`

## 5. 探索检查点 0

**结论：通过基础流水线检查，允许进入 P1 的现象探索。**

这里的“通过”仅表示基础数据与探针产物具备可用性，不表示 H1–H9 中任何假设成立，也不构成结果门槛。

### 当前仍需注意

- 本阶段使用单场比赛和固定 20 节点选择，只能作为 smoke-scale 输入。
- `semantic_coalition` 在 P0 中是确定性结构支持样例；同频语义–随机严格配平属于 P2 的重点比较。
- P0 只生成干预，不测模型对干预的响应，因此尚未支持共同模式优先、组织盲区或跨域预测等科学 claim。
- 当前机器 CUDA/MPS 不可用；冻结 ResNet-18 已在 CPU 完成，不影响本阶段接口检查。

## 6. 可复现命令

```bash
./.venv/bin/python scripts/run_phase0_pipeline.py
bash scripts/run_phase0_smoke.sh
./.venv/bin/python -m pytest -q tests/test_phase0_pipeline.py
```
