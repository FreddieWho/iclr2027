# CAMPAIGN_STATE.md — discovery campaign 首轮（2026-09-17 UTC）

工程起点，非研究结论。用户授权：MASTER_AGENT_PROMPT 总控命令；优先级覆盖旧 TODO/B30 本轮建议；
不覆盖数据许可/费用授权/真实记录要求。旧审计·强基线闭合·AMR 配方竞赛不先行。

## 基线与资产（bootstrap 实测）

- commit `f86dbdf` == 包基线，无需回滚，无新增结果需要避重。
- 8/8 已知入口 FOUND；checkpoints 候选 69 个（含 phase1 9 个、t5r3_sanity_v3 12 个、t5r4/t5r5/t5r6、AMR）。
- core kernels：`test_kernels.py` 24/24 通过；`smoke.py` 工程恒等式通过（边际差 0、臂能量误差 ~1e-17）。
- 算力：CPU only（无 nvidia-smi，torch 2.11 CPU）。无新付费资源授权 → 首轮全部 CPU 可行规模。

## 受控场景（已生成，纯程序数据）

- `artifacts/discovery_campaign/scenes/train_101`：512 场景（256+/256-），seed 101
- `artifacts/discovery_campaign/scenes/eval_202`：512 场景，seed 202
- `artifacts/discovery_campaign/scenes/holdout_303`：256 场景，seed 303（R05/胜出路线评价预留）
- 任务：四端点 AB/CD 真相交（proper crossing），oracle=`core.relations.segment_relation`，min_margin=0.02。
- 铁律：同一 parent 的一切变体/编辑/渲染永不跨 split。

## 首轮执行顺序（可顺序，不等 worker）

1. W1/R01：小坐标 MLP（2–3 个）＋同边际模式库（n=4 端点，group (4,) 或 (2,2)），比较
   Q_ref（等权）/ Q_coherent（角色固定协同模板）/ Q_source（源端贪心覆盖选权）；
   终点=任务风险（oracle 重算标签，label-preserving 与 label-flipped 分表），不是 Spearman。
2. W2/R02：同一 MLP 上搜“oracle 标签变、表示/预测不变”协同编辑；报双条件成功率＋分母。
3. W5/R05：冻结 MLP 倒数第二层特征，关系诊断对 vs 同预算随机对，metric initializer，
   在 holdout 上做检索/关系排序评价。
4. W6/R06（低资源并行）：numpy 弹簧模拟器＋小 MPNN 一步预测器，扰动依赖对长期 rollout 影响试跑。
5. T5R3 旧 checkpoint 适配器：仅做轻量加载探针（state keys/前向可行性），不阻塞 1–4；
   若适配成本 > 一轮试验成本，R01 首轮结论只写坐标模型，适配器记入下一轮。

## 看到什么→为什么改（活记录）

- R01 v1 全平（±1%）→ 怀疑 bank 太穷（3 模式），改联合分布族丰富度（bank-8，35 模式）：仍全平 → 判真阴性，关闭坐标分支（2026-09-17）。
- R02 首轮即见 48% 可翻转/67% 失察，但 stealth 分析反杀 metamer 叙事（翻转表示动得更大）→ 主张收紧为“预测未跟随”，转 R05 检验接口假设。
- R05 度量零迁移 → 改 kNN 读出 → 与头逐例一致 → 改隐藏层 → 更差 → 判编码器级缺失，关闭（2026-09-17）。
- R06 定律全等 → 查 oracle 发散 → 系统压缩（发散≈eps，不随 horizon 涨）→ 实现无错、物理选错，round2 转散射系统。
- 首轮结论：主发现 R02，方法 R04（未跑），动态 R06-round2；详见 reports/discovery_campaign/ROUND1_SUMMARY.md。

## 次轮活记录（2026-09-18）

- R02 复核后台开火：recheck_404/505/606（512 场景×3）× mlpA/B/C 共 9 搜。
- R04 首试：bank-8 全局 pi（逐 batch DRO，合法掩码冻结共享）vs 等权 vs 逐场景最坏臂 vs 干净；300 epoch 后台训练＋r02 式跟随评价。
- R06 散射：首版 rollout 漏 append（恒差 bug，已修）；修正后真发散（eps0.02→300 步 ×24）；全量 200/50/50 后台（MPNN 完全图 vs MLP）。
- T5R3 探针通过：TaskModel strict 加载 OK，前向 9ms/快照 CPU，Jacobian 可用；R01-T5R3 适配成本低，排队等 R02/R04/R06 结果。
- R04 首训作废重跑：pert 前向误包 no_grad 致四模型权重 bit-identical（损失不同但梯度只过 clean）；已修（grad 版 lp＋detached pi），smoke 验证四权重分化，坏数据已删、r04 全量重训中。
