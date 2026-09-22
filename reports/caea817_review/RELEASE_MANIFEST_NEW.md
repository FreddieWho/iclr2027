# RELEASE_MANIFEST_NEW — 结构重组后新清单与旧→新路径映射

> R9.5。本清单**不覆盖**任何历史 manifest（`artifacts/phase2/*/MANIFEST_SHA256.txt`、
> `docs/legacy/MANIFEST_SHA256.txt`、各 lock 内嵌 hash 均保持原字节与旧路径语义）。
> 路径映射来源：重组提交 `3d16a7f` 的 git rename 记录（`git show --name-status 3d16a7f -- scripts/`）。
> 冻结 config/lock 内的 `source_code.path` 字段仍记旧路径——属历史真实，
> 时代审计请检出 `3d16a7f` 之前的提交。

## 1. 旧 → 新路径映射（scripts/ 平铺 → 时代分区）

| 旧路径（历史 manifest 语义） | 新路径（当前工作树） |
|---|---|
| `scripts/fig_p1_p3.py` | `scripts/figures/fig_p1_p3.py` |
| `scripts/{bootstrap_env,download_public_data,initialize_research_repo,run_calligraphy_probe,run_phase0_pipeline,run_phase0_smoke,run_phase1_pipeline,spectral_smoke,verify_data,write_point_mainline_checkpoint,build_p0_overview}.py/.sh` | `scripts/p0p1/…` |
| `scripts/p2_*.py`、`scripts/run_phase2_*.py`（12 个） | `scripts/p2/…` |
| `scripts/{p3_*,audit_*t5r*,prepare_idsse_t5r2,prepare_soccertrack_t5r6,run_p3_*,run_t5r*,build_headline_table,compute_t5r5_prelock_audit,download_soccertrack_v2}.py`（28 个） | `scripts/p3_t5r/…` |
| `scripts/{amr_model*,run_amr_*,audit_amr_*,analyze_mu_threshold,build_spectral_cache_v4,plot_mu_threshold_svg}.py`（21 个） | `scripts/p4_amr/…` |

（完整 73 行 rename 清单见 `git show --name-status 3d16a7f`。）

## 2. 整改新增产物（本次 caea817 整改，均为新文件不覆盖历史）

| 产物 | 路径 | 来源 |
|---|---|---|
| P1G 修正版 | `artifacts/next_novelty/p1g_confirm/P1G_CORRECTED.json` | `experiments/next_novelty/p1g_repair.py --mode confirm` |
| P1 配对 OOF 重分析 | `artifacts/next_novelty/p1b_res_v2/`、`p1b_confirm_v2/` | 整改版 `p1b_analysis.py` |
| P3 v2（8×8 矩阵+identity 修复） | `artifacts/next_novelty/p3_v2/P3_FINAL.json` | `experiments/ccm_audit/p3_final_v2.py` |
| relflip 确认复算 | `artifacts/next_novelty/relflip_v2/RELFLIP_CONFIRM.json` | `experiments/ccm_audit/relflip_confirm_eval.py` |
| P2 v2（待全 seed 落盘） | `artifacts/next_novelty/p2_v2/`（s11 已有） | `experiments/repair_decomposition/p2_operating_v2.py` |
| Fig4 四臂图 | `paper/figures/fig4_fourarm.{pdf,png}` | `scripts/figures/fig_fourarm.py` |
| 官方样式 | `paper/iclr2027_conference.{sty,bst}`、`paper/{fancyhdr,natbib}.sty`、`paper/math_commands.tex` | media.iclr.cc 官方包（2026-09-22） |
| 审计文档 | `reports/caea817_review/{RECTIFICATION_LOG,CLAIM_SOURCE_MAP.csv,CONFIRMATION_USAGE,REPRODUCE}.md` + 本文件 | 本轮整改 |

## 3. 两类校验分开记录（R9.5/R9.7）

- **设计完整性测试**（历史冻结证据）：`tests/test_phase2_rigid_formal.py`（冻结 config 本体 sha 对 manifest＋设计断言）、`tests/test_p1g_corrected.py`（冻结 schema/阈值逐位）。这些测试验证"归档锁的完整性"，不验证当前工作树脚本字节=锁内字节（那是 `3d16a7f` 之前提交的性质）。
- **可执行依赖校验**（当前工作树可跑）：`tests/test_p1b_groupedcv.py`、`tests/test_lane_b_v2.py`、全量 `pytest tests/`——验证整改后代码产出与归档一致（如 p3_v2 15/15 点估计逐位一致）。

## 4. 验收状态
- 旧 SHA 保留：✅（历史 manifest/lock 未动一字节；新产物一律新路径）。
- 新 manifest：✅（本文件 + `artifacts/SHA256SUMS_UNTRACKED.txt` 维护大文件侧）。
- 待办：P2 v2 全 seed JSON 落盘后回填第 2 节状态。
