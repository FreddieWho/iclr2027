# N02 MAC 比复算（只读，无训练）

日期：2026-09-25。未改既有结果文件，未读封存池，未租 GPU。

来源：`artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results/structure_matrix.json`。四格参数都是 11,177,025。MAC 是 conv+linear MACs/图。

| 配置 | conv+linear MACs/图 | 除以 standard_64 148,046,336 |
|---|---:|---:|
| standard_64 | 148,046,336 | 1 |
| lowstride_64 | 563,282,432 | 3.8047711765 |
| standard_224 | 1,813,561,856 | 12.2499610933 |
| lowstride_224 | 6,900,204,032 | 46.6084080055 |

报告里的 12.25× 是 1,813,561,856 / 148,046,336 的四舍五入，不是新阈值。同一文件里 standard_64 的 layer1 是 16×16，standard_224 的 layer1 是 56×56；(56×56)/(16×16) = 12.25 整。网格面积与 MAC 在这个骨干上一起变，本矩阵分不开。

配对 J 的分母与加权以 `artifacts/e1a933_review/N02_interpretation/contrast_summary.csv` 为准：`value_parent` 是 parent 等权，`value_row` 是行等权。`all` 为 178 quartet / 85 parent；`small_edit` 为 5 quartet / 3 parent（parent 135 有 1 行，205 与 471 各 2 行）；`other_edit` 为 173 quartet / 84 parent。3+84≠85，因为 parent 135 与 205 同时出现在两层（`contrast_parent_deltas.csv`，对比 `B_standard_pretrained_flip minus A_standard_pretrained_flip`，quantity=J，seed=803）。不要把 small_edit 写成 parent 划分。

`questions.json` 的跨 seed `value`/`ci` 与 `value_parent` 的 cross_seed_mean 一致；其 `per_seed` 与 `value_row` 一致，不是 parent 等权。引用逐 seed 时用 `contrast_summary.csv:value_parent`。
