# configs/ 说明

全部 config 锁保持平铺、字节不动（冻结证据）。

⚠️ **路径字段对应 scripts/ 分区前布局**：2026-09-22 scripts/ 已按时代物理分区
（`scripts/INDEX.md`），各 config 内 `source_code.path`/`sha256` 记录的是当时的
路径与字节（历史真实，可经 git 历史复验）；投稿前最终重锁见根 `TODO.md` 顶部清单。

| 文件 | 时代 | 用途 |
|---|---|---|
| `project.yaml` | E2（止更） | 项目元信息（⚠️ 字段为 P3 时代快照，当前状态见根 STATUS.md） |
| `data_manifest.yaml`、`experiment_matrix.yaml` | E0–E1 | 数据清单 / 实验矩阵 |
| `phase2.yaml`、`phase2_matching_v2*.yaml`、`phase2_rigid_formal_v{1,2}.yaml`、`phase2_fracture_continuity_v1.yaml` | E1 | P2 系列锁（含脚本字节 sha256） |
| `phase3_support_geometry_v1.yaml`、`phase3_task_semantic_repair_v1.yaml` | E2 | P3/T5R 设计锁 |
| `t5r4_round{1,2}.yaml`、`t5r5_hidden_confirmation.yaml` | E2 | T5R 轮次锁 |
