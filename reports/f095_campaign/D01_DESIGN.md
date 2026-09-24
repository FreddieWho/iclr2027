# D01｜容量×数据×收敛设计（待执行）

- 历史锚点（已有，无需重训）：train_101（48K）＋ mlpA h64f32 ＋ 300ep/Adam1e-2 ＋ clean/flipmine，3 种子 checkpoint 即 `artifacts/discovery_campaign/r04b_s{11,23,47}/{clean,flipmine}` 与 relfeat 对照。第一步先用 `gen_fourarm_table.py`＋U1_SUMMARY 复现锚点数值（已完成：12/12 一致）。
- 新矩阵（5 格，共享交点 256×N）：宽度 {64, 256, 512}×N（feat 32/64/64）；数据 {N, 4N, 16N}×宽256（嵌套父场景，oracle 生成，manifest 记 parent 清单与 seed）。解析 oracle 列为可解性上限，不算学习模型。
- 每格 3 独立 seed ×（clean + 同 flip 挖掘）＝ 6 训；共 30 训。flip 挖掘复用 `r04b_methods.mine_flips` 同 seed 协议（cap12/margin0.03）；4N/16N 候选池按 parent 分层嵌套。
- 统一 trainer（待实现 `d01_capacity.py`）：显式存初始化、预处理、数据清单、优化轨迹、参数量/前向次数/墙钟；static/single-dev 选停止点与有限 lr 网格；报告等步数与各自收敛两套结果。
- 评价：static/dev 误差、A/B 边际、atomic-pass、CCM、全体 AB 错误、J、R_full/M/误迁移、D02 orbit 指标；训练 seed 单列；风险-原子/数据量/参数量三图。
- 处置线：强收敛普通模型消除大半失效→收窄"普遍缺陷"叙事；同 atomic 下 J/更新风险仍差且关系臂稳→U01/U02 可信起点；未收敛/生成差异→未决。
- 输出：`artifacts/f095_campaign/D01/{run_manifest,per_parent,summary,training_curves}`＋`reports/f095_campaign/D01.md`；主文 1 图/表。
