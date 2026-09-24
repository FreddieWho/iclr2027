# f095 campaign 工作区（2026-09-23 开工）

基准：`f095dfa`（= 当前 HEAD，已核对一致）。
总控：`docs/f095dfa_review_pack/01_AGENT_MASTER.md`；评估 `00_PROJECT_REVIEW.md`。
用户授权（2026-09-23）：A机制基线首批（D01+D02+U01，U03零训练，D03/D09支撑）＋足球优先D06＋VLM费用批准＋提交分支策略。

## 目录约定（总控§8）

- `experiments/f095_campaign/`：新 run 入口（建议、待建；不声称仓库已有）。
- `artifacts/f095_campaign/<route>/`：run_manifest、per_parent、summary、training_curves。
- `reports/f095_campaign/`：每条路线的问题/差异/合同/结果表/主claim/不能说/next。
- 每个 run 保留：repo_ref、model_hash、data_hash、arm/seed、train/eval父场景、query/forward预算、预测时间点、可见信息、失败解析、primary/secondary、raw counts。
- 逐 quartet 最小字段：parent_id/qid/split/arm/seed/orbit_g；y0/yA/yB/yAB；score0/A/B/AB；pred0/A/B/AB；identity labels 与 role map；edit_family/norm；是否参与任何训练。

## 并行纪律

一次最多两条训练 worker＋一条纯分析 worker（总控§6B）。重桥（D05/U06/U09）放独立分支，不阻塞主稿。

## 已核对的现场事实（2026-09-23）

- 总控§2文件全部存在；`relations.py` 在 `docs/iclr2027_discovery_campaign_20260917/core/`（非 experiments/core）。
- U03：`threshold_certificate.py` 原语已就绪（9项设计测试通过）；但 U1_PERQUARTET.csv 只有 A/B/AB logits、无 state-0，需补 state-0 前向才能组 [n,4] 证书（ adaptation，见 DECISION_NOTE 模板）。
- D01：`coord_mlp.py` 支持 --hidden/--feat，原锚点 hidden=64/feat=32/in_dim=8。
- D02：合法群 8 元素（A↔B、C↔D、AB↔CD）待实现为 `symmetry.py` 公共库；平均预测 8×前向成本需记账。
- D06：`data/raw/sports/` 下有 idsse-data/metrica/skillcorner/SNGAR/soccertrack_v2；复用 tracking＋已有 graph 实现，不复用 zone head 输出。
- §5(1)：`paper/sections/04_mechanism.tex` L111 四臂表 `AB endpoint err.` 混列已确认存在（raw/relfeat 为 AB 准确率，raw+flip 为错误率）。
