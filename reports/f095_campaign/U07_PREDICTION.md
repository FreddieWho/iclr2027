# U07 冻结预测（2026-09-23，训练前）

来源：N 格已见模式（raw：clean .055→flip .068，增益小；sixdist：clean ~.15→flip ~.43，增益大）。
预测（4N 未见格，容差取符号＋量级）：
- P1：sixdist-4N-f25 的 J ≥ sixdist-4N-clean ＋ 0.10（结构降低覆盖需求：25% 覆盖即吃到大半 flip 增益）。
- P2：sixdist-4N-f25 的 J ≥ raw-4N-flip-full（结构×25%覆盖 ≥ 无结构×100%覆盖）。
- P3（单调性）：sixdist-4N-clean ≥ sixdist-N-clean（数据单调，不反转）。
命中判据：P1–P3 全中→"结构降低覆盖需求" SUPPORTED；P1 失败（f25≈clean）→覆盖需求未降，N 模式只是全量 flip 特例；任一反转→如实报告失败预测。
