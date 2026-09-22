# LEADS

2026/9/5
## L-001  smoke 场 117092/117093 可作 parser 调试但禁入正式统计
manifest 明确排除。其余 8 场的 GSR 结构若与 smoke 一致，parser 可先在 smoke 上验证。
最小验证：只解析 smoke 场结构、不跑指标，约 0.5h。
状态：已放弃（2026-09-18 清理：parser 随 T5R2 审计通过，无残留工作）

2026/9/5
## L-002  ball 轨迹与 BAS 事件暂未使用
当前任务只需要 20 人位置；ball/BAS 可能支撑未来的事件对齐分析，但会引入新语义。
最小验证：暂不做，等 T5R6 主线闭合后再评估。
状态：已消耗（2026-09-22试点：主线弱null，S1埋，S2争夺存在+7pp两场复现保留appendix级，S3观察；见reports/l002_ball/）。WP-D核查：matched SMD结构性失败，残差示运动解释70-90%，S2降级为混杂观察，lead完全消耗。数据审计完成：canonical frames含ball行（WOY 14万行全场连续）＋events表（Pass 743/Tackle 195/outcome标签齐）；SoccerTrack侧无ball/event。试点方向：ball连续变化＋真实事件标签喂场景级分析，见正文。

2026/9/5
## L-005  GRF（Google Research Football）模拟干预臂
外部检索建议：模拟环境的干预有 ground truth，可彻底消除"输入干预导致内部表征 OOD"的审稿批评。
若成立可替代/补强 IDSSE 受限访问的可复现性负债，也是 rebuttal 阶段的现成预案。
最小验证：GRF 导出轨迹→跑冻结箅子与预测器，约 1-2 天。
状态：待挖掘（本期投稿冲刺不启动，rebuttal 或下一篇用）

2026/9/18
## L-006  训练类主张的 ~10 seeds/arm 确认门（E6）
小 MLP 重训噪声地板 ±15pp（同配置两次运行差 17pp，见 reports/last15h/N04.md），单种子训练对比不可 headline。
若开：唯一能让 N04 课程 / N05 迁移类主张上正文的通道；先导须 ≥3 种子同向且均值效应 ≥12pp 才值得烧。
最小验证：30 次小训练（3 臂 × 10 种子，同 recipe，eval-only 分母），约数小时 CPU。
状态：待挖掘（等 E2 之后有候选主张再开；同一时间只开一条）

2026/9/18
## L-007  转折位置精度的全新机制（E7，v3）
Bracket 双侧标签（v1）＋方向 margin（v2）在积分尺子下双种子一致输 flipmine 4pp："敢转"易、"转准"难，位置精度需要别的机制。
若成立就是主发现里唯一未解机制问题的新方法位；先做零训练诊断（转折处梯度/曲率结构分析），再谈训练，避免重蹈 v1/v2。
最小验证：坐标单转折路径诊断小试，零训练，约 1h。
状态：已执行（2026-09-22 零训练诊断，阴性）：冻结 flipmine 三种子上判据字面触发，
但消融显示全部可预测成分来自预测邻接量（min_abs_logit）；隐空间几何零增量，输入几何零预测力。
判决=边界：位置误差在现有轨迹结构中无可解释来源；下一篇若重启从噪声分解开始。见 reports/caea817_review/L007_DIAGNOSTIC_REPORT.md。
追杀（2026-09-22）：文献线另提 5 候选，已处决其二——掠射角×带宽交互（w/a 相关全<0.2且符号乱）与 kink 密度（方向全对但全<0.3线，不追）；分歧分解由训练线覆盖，离路褶皱/max-margin park。见 L007_FOLLOWUP_REPORT.md。
终局（2026-09-22）：三张训练牌全打完且全阴——P1 锁定动力学（中位锁定287.5 vs 收敛300，drift/late）；P2 LOO（top-k/随机比值0.85/0.84/0.17，数据决定论死）；P3 几何钉位（A仅2/4、C 0/4，sham亦动±6–11%）。转折精度问题正式封存，见 L007_TRAIN_REPORT.md；仅保留 rebuttal 引用价值。
**审计修正（2026-09-22，重要）**：上述五条判决经 GPT6astra 审计后**全部需要降级**——P3 违反 train-only 契约（用 eval 派生样本训练）、P5 弱阳性在完整目标下失效、P1 锁定统计与文字契约不符（287.5 是全局伪影，逐路径重算为中段 125–175）、LOO 是 6 次单删而非删 6 个、TracIn 权重错；另发现并列秩 Spearman（影响 kink 等离散特征）。修正后：P3 v2（train-only＋匹配 sham）0/8 不成立、剂量 LOO 不成立、C2 尺度阴性、P5 v3 无支持实例、P1 v2 未决。详见 CORRECTION_AFTER_GPT6ASTRA.md 与 L007_CORRECTED_RESULTS.md。
**本轮净收益（2026-09-22）**：撤回七条排除性声明，换取两条更硬的结果——C1 等价表示敏感性（91–95% 路径重标号下不等价）与 C1c 交互对称群稳健性（dev 24/24 ＋ confirm 24/24 全正；J 的标号敏感性按臂单调递减），后者已写入正文 §4 与 §7 限制 (xi)。

2026/9/19
## L-008  DINOv3 S/B/L 同家族 size sweep
门禁未解（HF gated manual，无 token；GitHub 只给 Meta 接受页，绕不过；S/B/L ID 已确认）。
若许可就绪，同一批 v2 数据、同一协议 S→B→L，B1 先行、holdout 一次性，禁 VLM prompting/finetune。
最小验证：先通 S 的权重下载＋dev atomic。
状态：已并入主线（2026-09-19 用户解禁：hf CLI 落地 S/B/L 快照；提特征 11 分片并行执行中，见 TODO #16）

2026/9/20
## L-009  自然图局部正控（Bridge-R Task 3，park）
验证对象不是"大模型通病"，而是"OOD抽象任务的可及性梯度能否走到自然图上"：同家族尺寸梯度＋固定线性配方下，自然刺激的changed区对比与可读性是否复现抽象图模式。
若成立才能把 verdict 从 WEAK_OR_OOD 往一般 scale 组织问题升级；若不成立则梯度锁死在抽象探针内，同样值钱。
最小验证：先冻结自然刺激管线（刺激/标签/渲染口径/混杂控制）＋Am04预注册，holdout继续封存；只在"升级verdict"或"审稿人要求正控"时开。
状态：待挖掘（Task 2 closure已定：只测OOD抽象梯度，不测一般scale问题；本期投稿不启动）

2026/9/20
## L-010  semantic-delta救援方向（已关闭）
假设"Δz比z1更易读出语义翻转"在DINOv3-L final R0上被证伪：matched 4000/1000对、SMD全<0.08，D0准确率0.497≈随机，不及两个低级control；confirm未开。
关闭原因不是数据不平衡，而是表示变化本身不携带线性可读的翻转信息。
状态：已放弃（dev门未过即停，不再向复杂readout fishing）

2026/9/20
## L-011  frozen foundation repair（已关闭）
Balanced Transition三臂pilot（DINOv3-L frozen R0＋残差adapter，3种子等预算）：balanced atomic最高0.684，P1门0.85全败；失败卡在atomic安装不是composition泛化。
状态：已放弃（DINO_REPAIR_NEGATIVE，不做LoRA/finetune/search，不进Phase 2）

2026/9/20
## L-012  foundation repair分支（永久关闭）
Competence-first：frozen DINOv3-L patch tokens＋conv spatial head，atomic仅0.68（门0.85），0/3通过；base好（0.88）edited差（0.65-0.71）签名依旧。AB从未提取。
状态：已放弃（FOUNDATION_COMPETENCE_NOT_ESTABLISHED，不再设新rescue）

2026/9/21
## L-013  composition-path fresh confirm额度（未用，留给审稿人）
WP1的111/111归因是discovery池（eval_202）证据；geometry_confirm从未铸造，one-shot额度完整保留。
若审稿人要求，可用seed 26092201铸新池＋冻结协议做一次性确认。
最小验证：重跑wp1_paths＋attribution，不开新挖掘。
状态：待挖掘（仅审稿驱动）

2026/9/22
## L-014  半径诊断R=r_m/r_s（保留观察，不施工）
miss组R中位~1.9 vs hit组~0.8，三种子同向；分解证明全来自模型半径（oracle难度相同）。讨论段可留一行；不发展margin-matching loss。
状态：待挖掘（仅审稿驱动）

2026/9/22
## L-014b  半径loss试点（已埋）
RADIUS-PILOT-1：radius trans均值0.499 vs flipcov 0.450，三种子全反向；stay FA s23 spike 0.193；R机制检查radius≈flipcov。按门控掩埋，不调参、不开confirm。
状态：已放弃（L-014诊断信号本身保留）

2026/9/22
## L-015  标号敏感性 / 置换等变方向（C1 衍生，本轮新增）
现象：同一几何对象、同一 oracle 答案，仅改四个端点的命名，raw 坐标模型在 91–95% 的
评估路径上给出不同根位（标号诱导 σ≈0.13 vs oracle 误差均绝对值≈0.17），70–79% 路径
翻转"是否检出转折"；四元组 J 层面 42–48% 状态翻转，raw 臂 J 摆动 75–133% 均值，
而关系特征＋flip 只摆 15–29%（dev＋confirm 双池三种子同序）。
为什么可能有价值：这是全文唯一一条**正面解释"关系输入为何有效"**的线索——
10 维关系特征在重标号下只是彼此置换，因此携带了 raw 坐标所缺的任务对称性。
若成立，可支撑"把任务对称性注入输入/架构"的设计原则，而不只是"这组手工特征管用"。
最小验证：(a) 零训练已完成（C1/C1c，见 C1C_RELABELING_ROBUSTNESS.md 与
C1C_CONFIRM_RESULT.md）；(b) 训练版本——置换增广训练或置换等变架构（如对四点的
DeepSets/注意力池化）能否在抬高 J 的同时消除标号依赖？约 2–4 臂 × 3–8 种子小 MLP
（单次 300 epoch 仅 0.8 s，成本极低），判据须预注册（建议：J 全路径成功率配对差
CI 排除 0，且标号敏感性相对波动降到 raw 臂水平以下）。
状态：待挖掘（本轮未开训练版本；投稿前不启动）
