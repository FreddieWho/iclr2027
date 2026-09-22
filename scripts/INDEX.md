# scripts/ INDEX — 脚本按时代分区（2026-09-22 物理重组）

> **布局史**：scripts/ 长期平铺；2026-09-22 首批整理时发现 8 个 configs + 7 个 AMR lock
> 以 sha256 字节级锁定脚本（锁死路径与内容），故暂保持平铺。同日用户授权
> **"sha256 可在整理完后重新部署"**，遂执行物理分区。冻结 config 内的
> `source_code.path/sha256` 字段从此对应**分区前布局与原字节**（历史真实，
> 可经 git 历史复验）；活树的最终重锁在投稿前部署（见根 `TODO.md` 顶部清单）。
> 机械迁移改动仅限：`parents[1]→[2]`（目录加深一级）、跨时代 `sys.path`/load_module
> 路径补全；无任何逻辑变更，tests 166/166 通过。

## 分区总览

| 子目录 | 时代 | 内容 | 文件数 |
|---|---|---|---|
| `figures/` | E6 终局/当前 | `fig_p1_p3.py`——Fig2（P1 双曲线）＋Fig3（P3 迁移条），仅读冻结 JSON | 1 |
| `p0p1/` | E0–E1 蓝图/P0/P1 | 环境/数据/bootstrap、phase0/phase1 管线、书法 probe、谱萌芽 | 11 |
| `p2/` | E1 P2 fracture | p2_* 核心模块＋run_phase2_* 管线（被 `configs/phase2_*.yaml` 历史锁定） | 12 |
| `p3_t5r/` | E2 P3-T5R | p3_* 几何/定位、T5R 运行与审计链、IDSSE/SoccerTrack 数据制备、扫掠 | 28 |
| `p4_amr/` | E3 P4-AMR（已放弃） | amr_model v1–v6、run_amr_*、audit_amr_*_lock、谱缓存/μ 阈值分析 | 21 |

## 跨时代引用关系（已在新布局下接通）

- `p4_amr/amr_model*` → `from run_t5r3_sanity import GraphEncoder, team_pool`
  （经 `sys.path.insert(parent.parent / "p3_t5r")` 接通 E2 编码器）
- `p3_t5r/` 扫掠脚本 → `load_module(ROOT/"scripts"/"p3_t5r"/...)` 与
  `ROOT/"scripts"/"p2"/p2_fracture_controls.py`
- `p2/` 管线内部互调（`sys.path.insert(SCRIPT_DIR)`，同代同目录）
- `tests/` → `from scripts.p2.<mod> import`、`load_module("scripts/<era>/xxx.py")`、
  `sys.path.insert(ROOT/"scripts"/"<era>")`

## 注意

- `p3_t5r/audit_current_state_consistency.py` 硬编码引用根目录旧治理文档
  （STATUS.md/CLAIM_LEDGER.md 原路径），随治理退休与分区已失效，保留作历史工具。
- `p4_amr/audit_amr_*_lock.py` 中 `code_hashes` 的键（`"scripts/amr_model_v6.py"` 等）
  是冻结锁数据，未改动；重跑这些审计需检出分区前提交（`bc3b169` 及以前）。
- 冻结 configs（`configs/phase2_*`、`phase3_*`、`t5r*`）内的脚本路径字段对应
  分区前布局，勿按字段直接寻址；现行位置以本 INDEX 为准。
