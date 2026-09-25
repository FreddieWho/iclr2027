# 路线2：视觉机制（v2合同修复完成 / 有界阴性）

## 当前状态

v2已在授权GPU上完成12个arm/seed run和3个clean-only baseline run。J3定义与native RGB输入合同均已修复并由归档预测独立重算。最终结论是direct与interaction的J3差异方向不稳定，属于有界阴性/机制未显示优势，不是视觉交互方法成功。

代码：`experiments/e832_focus/route2_visual/`。数据：`artifacts/e832_focus/route2/data/`。正式 runner 会先验证 manifest/data SHA，再检查 CUDA；无 CUDA 时只写状态和 unrun list。

## 精确执行命令

```bash
cd /home/huyudi/012_conference/iclr2027
python experiments/e832_focus/route2_visual/data_generator.py --out artifacts/e832_focus/route2/data
python -m unittest discover -s experiments/e832_focus/route2_visual -p 'test_*.py' -v
python experiments/e832_focus/route2_visual/visual_mechanism.py --smoke --out artifacts/e832_focus/route2/cpu_smoke.json
python experiments/e832_focus/route2_visual/gpu_preflight.py --data artifacts/e832_focus/route2/data --out artifacts/e832_focus/route2/gpu_bundle
python experiments/e832_focus/route2_visual/gpu_run.py --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json --data artifacts/e832_focus/route2/data --out artifacts/e832_focus/route2/gpu_run
```

等价的单命令 CPU 流程是：

```bash
python experiments/e832_focus/route2_visual/run_route2.py
```

它执行 fresh data generation、focused tests、`PILOT_ONLY` smoke、manifest preflight，并再次调用 runner 写出无 CUDA 的阻塞状态。

## 新数据（不是旧 exposed bank）

生成器只调用 `make_relational_scenes`、`oracle_at`、`atomic_edits`、`mine_quartets` 和 corrected canonical renderer；没有加载既有 scene/image artifact，也没有读取 sealed/dev512/old 159 bank。父 seed 在生成前冻结为 train `832701`、dev `832702`、test `832703`；observation seeds 为 `832711/832712/832713`，quartet mining/observation 为 `832721/832722`。renderer 是 native 64 RGB，模型输入 bilinear resize 到 224。

| split | parents | singleton images | clean | single-edit | AB training images |
|---|---:|---:|---:|---:|---:|
| train | 128 | 323 | 128 | 195 | 0 |
| dev | 32 | 81 | 32 | 49 | 0 |
| test | 128 | 319 | 128 | 191 | 0 |

test-only quartets：28 quartets、112 images、19 parents；没有 train/dev quartet 或 AB 标签。三组物理 parent key 两两不交，image SHA 集合也两两不交。observation recipe（背景、颜色、single-edit margin/像素变化筛选、quartet 保留规则）在 `data/data_manifest.json` 中逐项记录。

数据 archive SHA256：`8e5128e51499b32bc5082a010a27d0e19542a1979325c31969780594f1e221ee`。`data_manifest.json` SHA256：`fd148616753df82d607c68ead94c867a44a1c95af5bf1a746117223fc64e63c9`。manifest 同样锁定这两个 hash；runner 每次启动都会重算 archive、所有数组 hash、split disjointness 和 attached data-manifest hash。

## 四臂 formal contract（v2已运行）

- `direct`：shared ResNet18 global mean pooled image feature + linear head。
- `additive`：red/blue visible masked pools 分别经过同一个 shared linear head，再相加。
- `representation`：red/blue pools 拼接 + linear head。
- `interaction`：red/blue pools 拼接 + nonlinear `1024 -> 128 -> 1` head。

四臂使用相同 ResNet18 backbone、相同 native64 输入 resize 到 224、相同 train exposure；只使用 singleton train labels，checkpoint 只按 singleton dev BCE 选择。heads 参数量：direct/additive `513`，representation `1025`，interaction `131329`；backbone 参数量和实际 train/inference runtime 会写入每个完成 run。MAC 未伪造，明确记录 `MISSING_REASON`（没有 pinned validated MAC profiler）。每个 run 会保存 train/dev/test/quartet image exposure indices 及 SHA；推理 `forward(images)` 不接受坐标、parent/index、oracle label、AB truth、edit answer 或 combination ID。

v2 runner使用seeds `803/805/806`、20 epochs、batch 32、224 input，并输出修正后的quartet `P/A/B/AB/J3/J4`。full-repair/migration使用单独的clean-only direct baseline预测计算；不使用direct-reference flow冒充clean→repair flow。

## Final v2 result（2026-09-25）

v2修复了两个评估合同缺陷后重跑：

1. `J3` 现在按 A、B、AB 三状态同时正确计算；旧runner的`J3`字段曾错误地等于atomic-joint，预测文件本身未损坏。
2. 正式runner现在把native RGB原图传入模型，由模型内部resize/normalize；不再把normalized tensor当作原图生成红/蓝mask。

fresh bank仍为28个test quartets / 19 parents，3 seeds，12/12 run完成。归档预测重算结果：

| arm | J3 seed 803/805/806 | mean |
|---|---|---:|
| direct | 0.571 / 0.607 / 0.536 | 0.571 |
| additive | 0.393 / 0.321 / 0.536 | 0.417 |
| representation | 0.214 / 0.321 / 0.286 | 0.274 |
| interaction | 0.750 / 0.464 / 0.536 | 0.583 |

interaction相对direct的seed差为 +0.179、−0.143、0.000，方向不稳定；不能写成稳定的跨对象交互收益。clean-only direct baseline的J3为0.107、0.250、0.107（均值0.155），所以single-edit训练整体改善了联合状态，但interaction结构没有稳定超越direct。该路线结论是**有界阴性/机制未显示优势**，不是视觉交互方法成功。

结果与flow：`artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`。原始v1结果保留但标为合同错误，不进入科学结论。没有继续优化，因为没有发现新的可修复合同或实现因素。

## Release status

用户要求停止后通过AI Galaxy退租。AI Galaxy MCP显示账户有1台running instance，但`plan_release(instance_name=223.109.239.36)`返回“instance is not owned by this MCP state store”。因此当前无法通过该MCP执行退租；需要作者/账户所有者从AI Galaxy控制台释放，或提供该MCP-owned instance name。未使用SSH shutdown替代退租。


已运行：CPU fresh data generation（含 deterministic regeneration 检查）、9个 focused tests、native64→224 shape/finite检查、四臂 head 行为、no-hidden-input signature、missing-color zero-mask、metric denominator/data contract tests，以及 v2 12个GPU arm/seed和3个clean-only baseline。CPU smoke仍是`PILOT_ONLY`，不是性能结果。

v2正式结果：direct/additive/representation/interaction的mean J3分别为0.571/0.417/0.274/0.583；interaction-direct的seed差为+0.179/-0.143/0.000。结果路径：`artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`。原始v1结果因J3字段和mask输入合同错误保留但不作科学结论。

## 历史v1 GPU命令（已被v2取代）

以下命令仅保留作历史复现入口；科学结果应使用v2的`gpu_run_v2`与`static_baseline_v2`：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/e832_focus/route2_visual/gpu_run.py \
  --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json \
  --data artifacts/e832_focus/route2/data \
  --out artifacts/e832_focus/route2/gpu_run
```

如果输出目录已有部分 arm，使用一个新的 `--out`，或明确加 `--overwrite`。v2正式解释以本报告Final v2段和corrected_results为准。
