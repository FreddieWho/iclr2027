# scripts/ INDEX — 脚本按时代速查（物理布局保持平铺）

> **为什么不平铺改分区**（2026-09-22 结构整理的决定）：8 个 configs（phase2/phase3/t5r 系列）
> 与 7 个 AMR config lock 以 **sha256 字节级锁死**脚本内容与路径（如
> `configs/phase2_rigid_formal_v2.yaml` 的 `source_code.*.sha256`，且配置自身 sha 又录入
> `artifacts/phase2/p2_rigid_formal_v2/MANIFEST_SHA256.txt`）。物理移动会迫使修改脚本字节
> （`parents[1]→[2]`、跨目录 import），从而不可逆地破坏全部时代锁的现场可复验性。
> 因此脚本**保持平铺、字节不动**，时代归属由本索引承担。
> 若要重跑某时代的冻结审计链，请使用该时代布局（git 历史中 2026-09-22 前的任意提交）。

## E6 终局/当前（论文图表）

- `fig_p1_p3.py` — Fig2（P1 双曲线）＋Fig3（P3 迁移条），仅读冻结 JSON（REPRODUCE_FINAL §图）

## E0–E1 蓝图/P0/P1（Action-Mode 旧线）

- 环境/数据：`bootstrap_env.sh`、`download_public_data.sh`、`verify_data.py`、`initialize_research_repo.sh`
- P0/P1 管线：`run_phase0_smoke.sh`、`run_phase0_pipeline.py`、`run_phase1_pipeline.py`、`build_p0_overview.py`
- 书法/谱萌芽：`run_calligraphy_probe.py`、`spectral_smoke.py`、`write_point_mainline_checkpoint.py`

## E1 P2（fracture continuity，冻结证据链）

- 核心：`p2_statistics.py`、`p2_fracture_controls.py`、`p2_fracture_statistics.py`、
  `p2_matched_controls.py`、`p2_matching_diagnostics.py`、`p2_point_model_adapter.py`、
  `p2_heterogeneity_diagnosis.py`
- 管线：`run_phase2_pipeline.py`、`run_phase2_fracture_formal.py`、`run_phase2_rigid_formal.py`、
  `run_phase2_response_smoke.py`、`run_phase2_resource_gate.py`
- ⚠️ 本组全部被 `configs/phase2_*.yaml` 字节级锁定，禁止任何改动。

## E2 P3-T5R（几何可预测＋任务语义修复＋外部确认）

- 核心：`p3_support_geometry.py`、`p3_candidate_search.py`、`p3_causal_switch.py`、
  `p3_node_localization.py`、`p3_task_localization.py`
- 数据：`prepare_idsse_t5r2.py`、`prepare_soccertrack_t5r6.py`、`download_soccertrack_v2.py`
- 运行：`run_t5r3_sanity.py`、`run_t5r4_round{1,2}.py`、`run_t5r5_hidden_confirmation.py`、
  `run_t5r6_external_confirmation.py`
- 扫掠/对照：`run_p3_epsilon_sweep.py`、`run_p3_support_scaling.py`、`run_predictor_baselines.py`、
  `run_ratio_sweep.py`、`run_ssl_family_control.py`、`build_headline_table.py`
- 审计：`audit_idsse_t5r2.py`、`audit_t5r3_sanity.py`、`audit_t5r4_round{1,2}.py`、
  `audit_t5r5_candidate_lock.py`、`audit_t5r5_hidden_confirmation.py`、`audit_t5r6_confirmation.py`、
  `compute_t5r5_prelock_audit.py`、`audit_current_state_consistency.py`
  （⚠️ 最后一个硬编码引用根目录旧治理文档路径，随治理退休已失效，保留作历史工具）

## E3 P4-AMR（方法线，已放弃；全部被 `artifacts/phase4_amr/*config_lock.json` 锁定）

- 模型：`amr_model.py`、`amr_model_v4.py`、`amr_model_v5.py`、`amr_model_v5cap.py`、`amr_model_v6.py`
- 运行：`run_amr_m1{,_v4,_v5,_v5cap,_v6,_jgcl}.py`、`run_amr_cap_baseline.py`、`build_spectral_cache_v4.py`
- 分析：`analyze_mu_threshold.py`、`plot_mu_threshold_svg.py`
- 审计：`audit_amr_m1{,_v4,_v5,_v5cap,_v6,_jgcl}_lock.py`
  （⚠️ 锁文件的 `code_hashes` 键为 `"scripts/amr_model_v6.py"` 等历史路径字符串，
  属冻结数据，重跑审计需检出该时代提交）

## 跨时代引用关系（移动会破坏的耦合）

- p4_amr 各 `amr_model*` → `from run_t5r3_sanity import GraphEncoder, team_pool`（E2 编码器）
- p3_t5r 扫掠脚本 → `load_module(ROOT/"scripts"/"run_t5r3_sanity.py" …)` 与 p2 模块
- p2 管线内部 → 同代 `p2_*` 互调（`sys.path.insert(SCRIPT_DIR)`）
- tests/ → `from scripts.p2_* import`、`load_module("scripts/xxx.py")`、`sys.path.insert(ROOT/"scripts")`
