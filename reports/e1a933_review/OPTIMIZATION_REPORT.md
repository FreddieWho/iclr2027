# O01–O06 实际优化状态

2026-09-24快照；视觉实时状态读取其receipt，不从本静态文档推断完成。

| 路线 | 状态 | 实际执行及边界 |
|---|---|---|
| O01 | 有限矩阵完成，扩展PARTIAL | 108候选、36个dev选定格；新池1229quartet/250 eligible parent，原抽512 eval parent；4N raw100 J=.334–.475、distance100=.819–.889。6个规模×seed比较distance25>raw100。不是节省75%oracle查询；额外优化器/完整warm-start剂量矩阵未跑 |
| O02 | 完成 | 18 typed重训和旧四臂逐parent CI；T1正I/T2负I；**匹配容量（6681 vs 6529–6849）+ 匹配优化预算对照已完成**：typed 劣势仍存（T1 +0.486…+0.529、T2 +0.701…+0.755），机制为拟合失败（训练目标差 ~10³）；见 `O02_BUDGET_MATCH.md` | 可选扩展（难度/手性任务）未跑，施工卡标为可选 |
| O03 | MATRIX_COMPLETE | 同1760训练步预算，pretrained/random的static→flip平均J提升14.47/20.34pp；逐parent曝光仍不同，不归为纯内容因果 |
| O04 | 完成 | 12冻结encoder×6所选head，144凸候选均数值收敛；旧bank再分析，S可随head变化；**新任务前瞻干预已完成（主判定 MISS，P1 7/8、P2 8/8）**，**D10 下游回声已完成**；见 `O04_FORWARD_INTERVENTION.md` | typed encoder 类在新任务上的同口径结论未做（发布检查点无独立特征头） |
| O05 | 12臂完成 / 外推范围PARTIAL | 三场train/dev/test隔离；真实轨迹受控缺测学习与自然时间事件检验完成，test2794快照/134 E；学习模型未超过恒速+解析基线；自然缺测机制及真实传球成功未测试 |
| O06 | 272请求MATRIX_COMPLETE | 用户已有GPU实例上运行固定Qwen2.5-VL-3B；复用64 parent quartet+16sanity，J=0/64，sanity8/16，无拒答/解析失败；原子能力不足，不能解读为组合专属机制或所有VLM失败 |

详细证据：O01_DATA_OPTIMIZATION_REPORT.md、U10_CLAIM_CORRECTION.md、VISION_FACTORIAL_CORRECTED.md、O04_READOUT_REPORT.md、FOOTBALL_E_FEASIBILITY_V2.md。R07另完成30公平续训臂，不能冒充O01所有warm-start剂量格。

视觉训练与N01共享baseline，完整21臂已远端GPU完成并回收校验。N03另6臂已完成；N02完整48臂仍在运行。新增O05/O06均无需新租机/付费API；结果和解释见VISION_GPU_RESULTS.md、O05_LEARNED_PARTIAL_TRACKING_GPU.md、O06_NATIVE_VLM_GPU.md。
