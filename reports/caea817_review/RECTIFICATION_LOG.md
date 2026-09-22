# RECTIFICATION_LOG — caea817 审计整改执行记录

基准：`FreddieWho/iclr2027@caea817` → 本轮落点见各节。执行：Lane A（统计）→ 超时后收尾完成；Lane B（证据）→ 超时后收尾完成；Lane C（写作）→ 一次完成；R8（文献）→ 总控直接核实。格式：问题 → 变更 → 实际结果 → 受影响 claim → 处置。

## R1 — P1G 冻结阈值 bug ✅ 保留（修正版替代）

- 问题：`p1g_repair.py` 用 `freeze.get(f"s{seed}")` 读取，实际 schema 为 `s{seed}_clean` 键；确认模式静默退回当前数据分位数（实际 thr 18.29/18.72/21.57 vs 冻结 20.7555/22.1445/20.7557）。
- 变更：生产级 loader（真实 schema，缺键/类型/模型 SHA/bank 错配硬失败）；`P1G_CORRECTED.json` 新落盘（旧 `P1G.json` 留历史）；主比较=全 baseline-fixed 群体，`paired_delta` 更名为 `common_start_correct_secondary`。
- 实际结果（confirm1007）：s11 n=137 repair_end 0.555 Δ0.445 [0.298,0.598]；s23 n=100 repair_end 0.320 Δ0.680 [0.471,0.848]；s47 n=194 repair_end 0.624 Δ0.376 [0.188,0.570]；repair 新增起点错误 26/26/52（上报不筛除）。
- 受影响 claim：P1 repair residual 旧值 0.53–0.84 **作废**，新值 **0.32–0.62**（三 CI 全不含 0）；ledger 行已更新，旧值标注为 bug 产物。
- 测试：`tests/test_p1g_corrected.py` 6 项（schema/缺键/阈值逐位/主比较群体/不覆盖旧文件）全过。

## R2 — P1 统计 ✅ 保留（区间与措辞修正）

- grouped_cv：改为 paired OOF AUC-difference（同 parent draws），命名准确；in_sample 单列。结果：dev +0.30–+0.48，confirm +0.20–+0.37（CI 全不含 0，条件于该 CV 拟合）；pooled OOF 0.77–0.96 vs 0.35–0.63（dev），0.77–0.95 vs 0.52–0.63（confirm）。
- 输入/群体：m 起终点来源核实（无 0 伪装）；编辑族分布输出（8 族，非随机总体）；matched_check 问题已处理（stratified 口径）；row/distinct-start/quartet/parent 四计数，CI unit=parent；分层截取敏感性方向稳定；等 parent 权重 0.84/0.82。
- 概念：静态曲线=未筛起点、更新曲线=条件起点正确；top25 实际覆盖率仅 13.5%/18.1%（dev 参考分位，必须加注）；R(τ,ρ) 定义+口径 note；[1,1] 不作零不确定性解读；Δf 保持后验描述（本仓库无端点比值计算）。
- 测试：`tests/test_p1b_groupedcv.py` 4 项全过；全量 176 passed。

## R3 — relflip 主线 ✅ 保留并提升

- `relfeat_eval.py` 参数化（`--bank/--model-map/--out`）+ provenance（bank/model SHA、hand-feature 边界双声明）；默认行为与原 dev 一致。
- `p3_final_v2.py` 修复 identity 校验容差 bug（round-4dp 后 1e-6 比较误报；改未取均值 1e-9）；重跑后 p3_v2 点估计与冻结版 15/15 逐位一致。
- relflip_v2 复算保真：R_full/M 点估计与冻结版一致（CI 因 bootstrap draw 略宽，预期内）。
- 四臂同 bank（n_q=509）配对，缺失臂显式 null；hand-feature 边界（6 距离+4 半径，节点身份为输入）代码+JSON 双声明。
- 测试：`tests/test_lane_b_v2.py` 10 passed + 2 skipped（待 P2 v2 JSON，缺产物不假绿）。

## R4 — P3 边界 ✅ 保留（收窄后）

- 8×8 全转移矩阵归档（A/B/AB 正确性语义）；M/R_endpoint = 85%/94%/79%（重验通过，identity 残差全 0）。
- 配对 ΔJ CI（同 parent 重采样）：[-0.039,0.061]/[-0.051,0.065]/[-0.032,0.078]，全跨零 → "小幅变化/未证大收益"成立。
- keepbal 范围限定在固定 BCE 方案；110→001 排序诊断已做（Spearman 0.42–0.43，非纯阈值亦非强机制证据）。

## R5 — P2 对照 🟡 大半（待 P2 v2 JSON 落盘）

- 已完成：dev-mask 修复、raw logit 排序、双 estimand 拆分（policy_quality_difference vs common_act_difference）、dev-vs-eval coverage gap、完整阈值枚举、s11 贴边标记、score-shape vs best-dev 分离。
- 待定：`P2_OPERATING_V2.json` 后台重算运行中（s11 已出：clean 0.1979 / affine_grid 0.1667 / thresh_enum 0.0208 / flip 0.25；affine 点估计不高于 clean）。
- 正文措辞已按"点估计不高于 clean、不排除全部 operating-point 解释"锁定；s23/s47 落盘后复核 CI 句。

## R6 — 主稿重组 ✅（Lane C，3 占位符已由本轮填完）

- 形式化（E/CCM/J/H/R 恒等式、feed-forward 声明、Table 1 setup 全代码核实）入 §2；7 段结构；6.3 措辞表 12 行全落实；失败目录移附录；Discussion/Conclusion/Appendix 新建。
- 占位符填充：%FILL-P1-CV%、%FILL-P1G-RESIDUAL%、%FILL-P2-INTERFACE%（u02_fixed 冻结数，v2 JSON 落盘后复核）。
- 遗留 wiring（main.tex、fig4、编译）→ Wave-2。

## R8 — 文献 ✅（总控直接核实）

- Kamath ECCV 2024 / Jones ICLR 2021 / Villar NeurIPS 2021 / Donti NeurIPS 2017 四组官方页核实无误，已入 bib 并接 \citep。
- Jacobsen 句已改为 task-relevant 方向不变性表述（与 ICLR 2019 原文一致）。
- "task-oriented prediction" 无 canonical 邻域，已从正文删除（核查员建议方案 a）。
- Press（Findings EMNLP 2023）/ Composition Collapse（Yu et al. 2026 preprint）保持不争首创立场。

## 未解决 / 待 Wave-2

1. `P2_OPERATING_V2.json` s23/s47 落盘 → 复核 §4b affine CI 句。
2. R7：fig4 四臂图生成、main.tex 接线、官方样式编译、匿名包。
3. R9：REPRODUCE.md（bash for 循环等）、确认池使用表（本目录 CONFIRMATION_USAGE.md 初版已建）、新 manifest + 旧→新路径映射、PROJECT_MAP FOOTBALL_CONF 标注。
