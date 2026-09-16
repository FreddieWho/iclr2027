# 执行 Prompt：已完成 Memory Pilot 的修复与交付收口

你负责 `/home/huyudi/012_conference/iclr2027/submission` 已完成工作的修复。先读本文件和同目录 `REVIEW.md`，核对当前状态，然后直接完成授权范围内的本地修复；不要只给计划，不按 TODO 逐项补实验。

## 目标与边界

目标是：消除当前/历史状态混淆，修复已复现的运行器完整性与续跑缺陷，让已有结果能在清晰、可移植的离线交付包中复核。结论仍是 `INCONCLUSIVE_NO_GO_SIGNAL`；不把软件修复当成科学 GO。

- 工作范围仅 `submission/`。不改父仓库 Action-Mode，不修改 `sources/` 原文，不读/展示 `.env` 值，不自动 commit/push、发消息或上传数据。
- 不调用付费或免费模型，不重新压缩、不重读问题、不新增 history，不续跑 V3，不启动 P3/C2/LongMemEval/restore/placebo，不换 endpoint、prompt 或预算去重新生成科学结果。
- 本地已有文件读取、离线复算、脱敏证据副本、代码修复、必要定向回归检查和文档修订已在范围内。依赖无法离线满足时记录具体 `BLOCKED`；不要擅自转成在线安装/下载或替代实验。
- 一份任务一个 owner、一个修复 ledger；新增证据在唯一目录只追加。多人同时工作时不得覆盖他人修改；默认无需派生 agent。
- 先分类：状态/打包为 Operational/non-code，仅做必要检查；F3–F5/F8 是行为变更，针对真实失败做离线回归，不为纯文档加测试，不默认跑广泛测试。
- 保留原运行收据：`work/`、P0/P1/P2 原始结果 JSON/JSONL、原 SQLite、原配置/prompt 与既有审计目录不可覆盖。允许在新的修复目录复制脱敏快照、输出新版派生表与勘误。需要历史配置迁移时创建新配置，不重写冻结配置。
- 历史报告可加显式 superseded banner 和当前索引链接；历史 JSON 的原始值不改。主报告只更新出处、边界或经证据支持的勘误。

## 当前事实（执行前复核，不依赖本文件盲目改值）

当前结果入口是 `reports/V4_FAST_PILOT_REPORT.md`、`reports/V4_ROUTE_DECISION.json`、`artifacts/v4_p2_compact_v2_science_result_20260914.json`。

P2 runtime 为 `configs/v4_compact_v2_continuation_runtime_20260914.json`，SHA-256 `db7b7392f2e9ecf3a0270177170baa03efb71006927d222c3f9b34714589cc28`。
P2 history 汇总 `artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl` SHA-256 `9a5502b2e00f4f9fcaf2daffeaa820f28e0eeba25cca4eef7d03fc4c3e18497e`。
P2 memory `work/v4_fast_decision_20260914_compact_v2/p2/live/memories.jsonl` SHA-256 `3d11ba615f468b3603a47201e4a382006c3016ac82fea8bdf4795d9f8590330e`。

现存 P0/P1/P2 共 1,776 评分行，独立检查未发现 exact score 或 P2 汇总字段差异。P2 24 histories、每个 8 题；R1 有 11 个三路径完整 history，R2 有 12 个。52/72 memory 自然停止，全部超过 8192 软目标，20/72 是技术缺失。三路径 complete-case SD 分别为 −4.545 pp 和 −3.125 pp；所有相关 CI 跨零。P1 旧 prompt 不合并；R2 为 reader sensitivity；P3 `NOT_RUN`。

独立复算目录 `docs/ICLR_MemoryV4_Independent_Reanalysis_20260914/` 的 12 个结果能复现，但 ZIP 旧、目录单独拷出缺源文件。具体缺陷证据见 `verification.json`、`runner_gaps.json`、`native_length_check.json`、`rewrite_control_check.json`。

## 执行顺序

### 0. 固定输入和变更范围

1. 读本地 AGENTS、当前 V4 主报告、continuation amendment 与上述证据。`git status --short -- .` 只关注 submission；不要清理父仓库未跟踪文件。
2. 新建 `reports/repair_<实际日期时间>/`，记录当前 Git HEAD、工作树差异、输入哈希、修复项 F1–F8、状态和产物路径。存在同名目录则新建后缀，不覆盖。
3. 核对关键哈希。若源数据与本审查快照不同，记录差异并查明来源；不能用旧预期去覆盖新合法工作。无法解释的源变动阻断依赖它的复算，其他独立修复继续。

### 1. F3/F5：评分与分析的输入必须可验证

修改 `scripts/run_v4_fast_pilot.py` 相关函数：

- `score_phase` 在读取 query/建立 provider/费用预留之前检查：冻结 manifest 状态、实际 memory 文件 SHA、各正文哈希、实际 history 文件 SHA、history/condition 唯一键及合同覆盖；不能仅信任布尔标志或 self-reported hash。
- reader manifest 必须绑定实际 memory 文件、query、history、reader prompt 和代码版本。用明确兼容性规则支持合法 R2 effort 解析，不要求不同 reader 配置逐字相同。
- 评分 COMPLETE 必须基于预期唯一 `(reader, history, condition, question_id)` 键完全覆盖，不能用 `len(rows)>=expected`；重复、跨阶段混入和未知键均明确报错。
- `aggregate_history_utilities`/分析入口接收或可靠读取冻结 query 合同，以合同中的预期题数和类型成员作分母。缺题、重复题、schema-invalid 保留技术缺失，不能缩小分母或填 0。
- 保留旧分析的三路径 complete-case 主估计及类型分析原纳入规则；pair-specific 作为单独敏感性分析，不悄悄改变原估计对象。

验收：用纯本地 fixture 证明改 memory/history、哈希不符、重复键、每臂只有 1/8 题均在使用错误证据前失败或明确缺失；不产生任何 provider 调用。正常完整样本仍可处理，既有 P2 重算值不变。

### 2. F4：修复调用收据与评分行之间的中断恢复

- 不能因为 call receipt 已完成就跳过缺失的 score rows。
- 从确定已返回的缓存/解析响应恢复缺少的评分；按 score key 幂等追加，已有正确行不重复写。
- 若缓存不足以恢复，报告具体单元 `BLOCKED_MISSING_RESPONSE_RECEIPT` 或同等明确状态；不可自动重发不确定请求、假填答案或标 COMPLETE。
- 不把该修复应用为对历史 P2 的自动补跑。现存 P2 完整，不需要更改原结果。

验收：模拟中断在 call receipt 之后、0 行或部分 score rows 之后；恢复成恰好预期键集，再恢复一次内容不变，模型调用计数不增加。没有确定响应时停在明确阻塞状态。

### 3. F8：执行 prompt 与 runtime 绑定

- 替换隐式全局 compressor prompt 选择：实际从新 runtime 的显式路径/哈希解析，写入 manifest 并校验。
- 默认路径不得默默组合旧 runtime 与最新 compact-v2。可以要求调用者显式 `--runtime`；历史 runtime 缺字段时给迁移说明或显式历史兼容参数，不猜测。
- 固定 reader batch/retry/single prompt 的来源/哈希，使后续续跑不能混用修改后的 prompt。
- 保留历史配置原样，新建修复版 runtime/template；本任务不执行它的 live 路径。

验收：两份临时 runtime 指定不同 prompt 时实际选中对应模板；修改模板后 frozen resume 被拒绝；合法 P2 离线分析不受影响。定向测试无需真实 tokenizer 或网络。

### 4. F1/F7：整理当前状态和方法边界

- 新增 `reports/CURRENT_STATUS.json`（或同等唯一入口），列 active protocol/amendment、当前报告/决策/数据/账本、历史被取代关系、已完成矩阵与 `NOT_RUN/BLOCKED` 项；README 首屏直接链接它和当前主报告。
- 给 `MASTER_AGENT_PROMPT.md` 及旧 `PILOT_REPORT`、`V4_PREFLIGHT`、`VALIDATION_REPORT` 等当前语气的旧文件加历史标记和正确导航，防止重新启动旧任务。保存原历史正文/事实，不把旧阶段改写成已经完成。
- 对根旧 `artifacts/model_manifest.json`、`dataset_manifest.json`、`cost_summary.json`、`reports/decision.json`，通过新索引标注范围与 successor，不篡改历史收据。
- 当前报告分开列出 P0 技术校准、P1 旧 prompt 探索、P2 owner-amended 结果、R2 sensitivity、独立离线复算。保留 V3 `INCOMPLETE_PROTOCOL_DEVIATION`、P3/C2/自然评估等 `NOT_RUN`。
- 主报告追加 rewrite 限制：direct 首次 cap=24,576；rewrite 首次 cap=32,768 且角色 prompt 不同，24 个 rewrite 第一阶段均不等于对应 direct memory。这是策略比较，不能当成“同一个 direct memory 只多改写一次”的纯控制。只补解释，不改历史 memory。
- 明确费用 `$8.29376795` 已含 `$0.1529898` 预留；每模型历史保守占用重复分配仅用于各自 cap，不能将模型数相加报告成项目实付。不改旧台账。
- 不声称目前完成了 24 对机制证据审计、完整 novelty、严格等长实验或完整第三方复现；没有自动平台隐藏压缩的可靠证据时，不写已证明无隐藏压缩。

验收：从 README 一次导航即可找到当前结论/输入/运行范围；不会把旧技术 NO_GO 当当前科学结果；所有科学主张和缺失状态与真实 artifact 一致。

### 5. F2/F6：形成可移植、可核验的新交付

- 保留 20260914 原复算目录/ZIP，本次创建新版本；不得只覆盖旧 ZIP 而失去变更记录。
- 新复算包包含新版脚本、报告、changelog、CSV、固定源 JSONL 副本及其 SHA/source commit；源副本与 CSV 逐字段校验。源不符、缺源、重复键均 fail closed。允许 `--source`，但独立 ZIP 的默认路径必须离线可用。
- 为主结果增加最小离线证据包：已有 synthetic histories/queries、P2 memories/必要步骤、R1/R2 rows/manifests、运行配置/prompt 的注明来源副本、复算脚本与 manifest。P0/P1 可用单独历史索引，不合并统计。只复制本地已有必要证据，避免整个 `work/` 和 API 缓存倾倒；排除凭据、provider headers、隐藏 reasoning、第三方原始数据和 tokenizer 大文件。
- 提供一条离线命令从既有逐题答案重算 exact scores、history utility、长度分布、三路径完整和 pair-specific 表、原报告 CI；无需重跑模型。各自 CI 算法/seed 清楚标注，统一 seed 独立复算不得冒充原 runner 全部随机序列。
- 原 C1/R1 runner SHA `55eabd3a…` 与当前 R2 SHA `8f7afcbc…` 分别列明。允许只读查找 Git/备份；找不到旧源码时写 `UNAVAILABLE_ORIGINAL_RUNNER_SNAPSHOT`，保留限制，不能伪造原版。新修复代码有自己的 SHA，不能回填成历史运行代码。
- 保存历史根 checksum 快照，修复当前交付的哈希清单并明确覆盖范围，避免把未覆盖新文件的旧清单称完整发布验证。新包逐文件 SHA、ZIP SHA、文件清单及使用说明一致；不产生自引用哈希。
- Git 当前未跟踪的独立复算材料应列为待发布，不声称已提交。只准备本地交付和建议提交路径，不自动 git add/commit/push，也不新增宽泛忽略规则掩盖缺交付。

验收：在干净临时目录分别解压新复算 ZIP、新最小证据包，用文档命令离线运行；结果与冻结值一致，源码/源文件不匹配时失败。清单全量校验无 unexplained mismatch。若某些证据无法纳入，列明文件、原因及对复核能力的影响，不能仅给 preflight 并声称包已完成。

## 可选完善及明确不做的工作

有余力可补：已有 raw baseline/缺失路径分解附表、环境依赖说明、`PYTHONPATH=scripts` 的定向测试命令。全部来自已有证据。

新 scientific protocol、真实长度操纵、共享起点 rewrite、类型交互新历史、P3/自然语义评分、C2、完整 evidence/novelty 审计只写后续建议，**不在本 prompt 内执行**。原 TODO 中这些项可保留未做并注明理由，不因此阻塞这次工程收口。

## 最终交付与停止条件

交付至少包括：

1. 修复 ledger：F1–F8 各 `FIXED / PARTIAL / BLOCKED`、改动路径、证据、残余影响。
2. 当前状态入口及修订后的导航/主报告边界。
3. 运行器修复、针对上述真实失败的定向离线回归及日志。
4. 新独立复算包、新最小证据包、manifest/checksum、干净目录验证记录。
5. `REPAIR_REPORT.md`：实际完成、未完成/未运行、数值是否变化及原因、科学结论边界、待发布路径。若数值变化，保留原结果并出勘误，禁止静默覆盖。

停止前核对冻结源哈希未变、没有新增模型调用/费用、没有触碰父仓库和 sources、没有把 `NOT_RUN` 改成无证据的通过。只做必要定向验证；检查通过后交付，不继续扩展任务。

最终向用户简短报告修复结论、产物链接和剩余阻塞。只有全部必修项关闭才称工程修复完成；无论工程是否通过，科学结论仍须由证据决定，当前应保留 `INCONCLUSIVE_NO_GO_SIGNAL`。
