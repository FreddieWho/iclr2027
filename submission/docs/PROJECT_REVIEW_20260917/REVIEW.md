# 项目独立检查 — 2026-09-17

结论：**主要 pilot 及离线复算的现有数值可核对；交付一致性与运行器可靠性仍需修复。** 保留 `INCONCLUSIVE_NO_GO_SIGNAL`。本次没有发现需要撤回 P2 数值的证据，也不将工程检查通过解释成科学验证完成。

可执行修复指令：[REPAIR_PROMPT.md](REPAIR_PROMPT.md)。本次只新增检查文件，没有修复项目代码、改写历史结果、调用模型、重跑科学实验或执行 Git 发布。

## 范围与实际检查

任务分级为独立审查与文档交付（Operational/non-code）。检查对象是 `submission/`，不是父仓库 Action-Mode 全项目。根目录合同、V4 改版/continuation、当前科学结果、运行器关键路径、复算目录和 ZIP 均纳入；TODO 不作为验收依据。

| 对象 | 实际检查 | 结论 |
|---|---|---|
| P0 reader 校准 | 48 评分行、唯一键、问题哈希、exact score | 通过；oracle/raw 各 16/16；no-memory 0/16 |
| P1 旧 prompt exploration | 384 评分行及逐题重计分 | 通过；288 valid、96 compressor-invalid；不与 P2 合并 |
| P2 R1/R2 | 768+576 评分行，问题/记忆哈希、有效性、逐题重计分 | 通过；R1 600 valid/160 compressor-invalid/8 schema-invalid；R2 416/160 |
| P2 派生 history 汇总 | 从逐题行重建 48 个 reader-history，核对 U、长度、n_valid 共 432 字段 | 全部一致 |
| P2 memory | 72 行文件/正文哈希；本地固定 tokenizer 重数 native tokens | 全部一致；52 个 natural-stop 均超过 8192 软目标 |
| 既有独立复算 | 在临时目录核对 288 个 CSV 转录字段，执行 12 项、每项 20,000 bootstrap | 输出 CSV 与原文件逐字节一致 |
| 费用 | 只读共享 SQLite，核对当前累计与预留 | 与 P2 汇总一致；见下文保守口径 |
| 发布完整性 | 根 SHA 清单、复算 ZIP CRC/目录差异、Git 跟踪状态 | 存在 F1/F2/F6 问题 |
| V4 运行器 | 定向离线复现，全部替换为 fake provider、临时目录 | 复现 F3/F4/F5；没有真实调用 |
| 现有 V4 单测 | 仅 `test_v4_fast_pilot.py` | 加 `PYTHONPATH=scripts` 后 7/7 通过；未运行广泛测试 |

共核对 1,776 条评分行，其中 1,352 条 valid。`verification.json` 中 1,603 个数据一致性检查均通过；这不包含“发布包完整性通过”的含义，发布问题另列。未重新解析所有 provider 原始响应、逐条复核 V1–V3 的历史探针、重新做完整语义证据审计、联网查新或核验父仓库 Action-Mode 的外部确认状态。因此不是全部上游行为的完整重放。

证据：[verification.json](verification.json)、[native_length_check.json](native_length_check.json)、[runner_gaps.json](runner_gaps.json)、[rewrite_control_check.json](rewrite_control_check.json)。复现脚本和定向测试日志同目录。

## 必须修复

### F1 — 高：当前入口与历史状态混用

- `README.md:3` 仍说没有真实 LLM pilot；`VALIDATION_REPORT.md:12` 仍说未调用真实 LLM。
- `reports/PILOT_REPORT.md`、`reports/V4_PREFLIGHT.md:3` 仍把停在 P0、P1/P2/R2 未运行写成当前状态。
- `reports/ROBUSTNESS.md`、`reports/EVIDENCE_AUDIT.md`、`reports/LENGTH_AUDIT.md` 和根 `artifacts/model_manifest.json`、`dataset_manifest.json`、`cost_summary.json` 也没有统一的历史/当前指引。后者仍是 `NOT_RUN/0`，不能当当前费用。
- `reports/decision.json` 是旧技术 NO_GO；`reports/V4_ROUTE_DECISION.json` 才是 continuation 后未决结果。V3 supersession 文档虽解释了部分关系，根入口没有接好。

影响：读者或下游 agent 可能误读为没有结果，甚至从旧 MASTER prompt 重新启动已结束实验。修复应新增唯一当前状态索引，给旧报告加历史标记/指向；原始 JSON 收据保留，只在索引标明被取代关系，不能将历史未运行记录全部改成完成。

### F2 — 高：复算 ZIP 过期，目录依赖没有闭合

- `docs/ICLR_MemoryV4_Independent_Reanalysis_20260914.zip` 只有 4 文件；README 与 `recompute.py` 均不同于当前目录，缺少 `PHASE_REPORT`、`RELEASE_CHANGELOG`。
- ZIP 内脚本没有当前目录的 `SOURCE_SHA256` / `verify_transcription`。运行旧 ZIP 得到相同统计值，并不代表做过来源校验。
- 当前 `recompute.py:19` 依赖目录外 `artifacts/v4_confirmation_rows_compact_v2_20260914.jsonl`；只复制当前目录运行会 `FileNotFoundError`。这不是源 CSV 数值错误，是包不自足且使用说明不完整。

应把锁定汇总源以来源注明的副本纳入新包，或提供显式 `--source` 且明确完整 checkout 前置条件；若交付独立 ZIP，优先做到离线自足。重新打包并在干净临时目录验收，保留旧 ZIP 为历史件。

### F3 — 高：评分前没有落实冻结输入校验

位置：`scripts/run_v4_fast_pilot.py:804`，后续 `:893` 使用 memory 内已有哈希。

`score_phase` 只检查压缩 manifest 的 `COMPLETE` 和布尔 query-blind 标志，没有将实际 memory 文件、正文、原 history 与压缩 manifest 中的冻结哈希核对。离线将 memory 正文改为 `MODIFIED after freeze`、保留旧哈希后，runner 仍发送该文本给 fake reader 并返回 `COMPLETE`。

修复：在读 query、创建 provider 或产生费用前校验实际文件、正文哈希和预期唯一键；评分 manifest 固定 memory 文件哈希和 reader prompt 哈希。配置与阶段的兼容性须明确，不能因合法 R2 effort 解析而误拒绝同一冻结 memory。

这是真实可触发的完整性漏洞；本次现存 P2 哈希实际匹配，**没有证据显示现有结果遭到篡改**。

### F4 — 高：续跑可能永久遗漏评分行

位置：`scripts/run_v4_fast_pilot.py:865`、`:895`、`:979–1019`。

调用收据先写入，逐题评分后写入；续跑发现 completed call 就跳过整个 condition。离线模拟“调用收据已在、评分行未落盘”，续跑结果为 `INCOMPLETE`，`0/8` 行，且不能恢复。

修复：以预期 score key 的完整性判断完成；从已确认响应缓存/解析收据补缺行，幂等且不产生重复 API 调用。若没有可恢复的响应，明确阻塞该单元，不能把不确定请求当可安全重发。当前四份 reader 产物键完整，本缺陷未在这些产物上形成缺行。

### F5 — 高：统计聚合把“观察到的行数”当应有问题数

位置：`scripts/run_v4_fast_pilot.py:1089–1108`；评分完成判定 `:1023` 还使用 `len(rows) >= expected`。

`aggregate_history_utilities` 以 `expected=len(rows)` 计分，缺行会缩小分母。离线每臂只有 1 个有效问题时，它输出 U=1、`delta_SD=0`，没有拒绝；若被当作 P2，会将本应 8 题的未完成 history 纳入分析。重复键也不能仅靠总行数排除。

修复：从冻结 query 合同得到预期键及每类型分母，校验唯一性和覆盖；缺失/重复不能成为有效完整 history。分类型分析可保持原有 pair-specific 规则，但分母必须来自合同。现存 P2 每臂应有键齐全，原始结果不需要因此改值。

### F6 — 中：证据发布与哈希清单未收口

- 根 `SHA256SUMS.txt`：119 项一致，`reports/PILOT_REPORT.md` 1 项不一致；没有缺失文件。清单日期/覆盖范围也未指明新的 V4 交付。
- 独立复算目录及其 ZIP 当前均未被 Git 跟踪。不能把本地完成表述成已经随 `5646f1d` 发布。
- `.gitignore` 排除 `work/`，因而公开包没有本次可在本地核对的 synthetic histories、queries、memory、reader rows。已有复算文档诚实声明了限制；若要交付可独立核验的完成项目，需要最小离线证据包。
- 科学 artifact 保存 C1/R1 runner 哈希 `55eabd3a…` 和 R2 哈希 `8f7afcbc…`；当前源码匹配后者，`work/` 中没有前者的同名源码快照。前者字节是否能从 Git/其他备份恢复尚待查，不能根据哈希反推源码或把当前版称为原 C1 精确版。

修复：新增版本化本地证据包和 manifest，只纳入需要且可共享的 synthetic 证据及脱敏配置；补一条离线从逐题结果到主表的命令。更新当前交付的 checksum，历史清单保留并说明范围。Git commit/push 属于后续发布，不在本次授权内。

### F7 — 中：主报告应明确 rewrite 控制的实际限制

位置：`scripts/run_v4_fast_pilot.py:584–600`；实测 `memory_steps.jsonl`。

direct 首次写入 cap=24,576；rewrite 首次写入 cap=32,768，且 prompt 的 final/intermediate role 不同。rewrite 独立重生成起点，24 个 history 的第一步 memory 均不等于 direct memory。它不是在同一已实现的 direct memory 上“仅再改写一次”的配对控制。

主报告已正确披露软目标、长度超标和缺失，但对上述额外限制交代不足。应说明现有比较检验的是整套路径策略，不足以识别纯重复改写或纯 bottleneck 机制。不能事后用共享 direct 起点替换历史 rewrite 结果；这只能作为另行授权的新实验设计。

### F8 — 中：运行配置没有实际控制 compressor prompt

位置：`scripts/run_v4_fast_pilot.py:28–30`、`:97–99`、`:534`。

prompt 来自全局 `PROMPT_COMPRESS`，不读取当前 runtime 的 `compressor_C1.prompt_path`。当前默认 runtime 仍是旧 `v4_frozen_runtime.json`，实际全局 prompt 却是 compact-v2。仅指定历史 runtime 无法恢复历史 prompt；新 runtime 写不同 prompt_path 也不会生效。

修复：将 prompt 路径/哈希作为实际执行参数，从 runtime 解析并校验；缺少该声明的历史配置须显式选配或仅用于历史阅读，不能默默套用最新 prompt。固定代码与 reader prompt 来源；保持已有 P2 收据不变。

## 可以完善，但不应变成自动补实验

1. **费用标签更清楚。** SQLite 累计 `$8.29376795` 已含 `$0.1529898` reserved；按模型保守口径各自加了完整旧项目占用 `$1.37211145`，模型数不能相加充当项目实际支出。`separately_reserved`/`before_separate_reserved` 字段命名易引起重复扣减，当前金额与账本一致，未发现超额支出。新报告明确 actual/uncertain/含预留关系即可，历史收据不重写。
2. **环境与定向测试可移植性。** 当前 requirements 未明确包含 V4 的 `tokenizers`/`transformers`；固定 tokenizer 是本机绝对路径。给离线证据复算单独最小依赖说明，给未来 live runner 说明额外依赖和本地资源定位。单独运行 V4 测试需 `PYTHONPATH=scripts`；已有日志保留了首次 import 失败和修正后 7/7 通过，不能称已有测试覆盖了 F3–F5。
3. **科学诊断。** P2 R1 raw 已有 191/192 correct，可补入现有结果表；缺失按路径/阶段分解、类型效应和 pair-specific 结果可统一成描述性附表。不要选择性报告“更好看”的纳入规则，也不要把每题当独立样本。
4. **证据/文献工作。** V4 已将完整 novelty、机制证据审计、C2/自然数据发表级泛化放到 pilot 之后。现存 `NOT_RUN` 不等于工程缺陷；只有准备机制/泛化/创新性论文主张时才需补做，且联网文献核验与新调用分别定范围。未在本次核验的 Action-Mode“已完成外部确认”表述不能作为本项目的审计结论。
5. **后续实验。** 真实长度操纵、共享 direct 起点且首步参数一致的 rewrite、降低技术失效、新 histories 上的类型交互/位置平衡值得另拟方案。本次没有结果要求自动启动它们。P3、C2、自然语义评分与 restore/placebo 继续为 `NOT_RUN`。

## 已确认而不应“修掉”的结果

- R1 complete-case SD：n=11，−4.545 pp，90% CI [−15.909,+4.545]；R2：n=12，−3.125 pp，[−16.667,+9.375]。
- Pair-specific SD：R1 n=13，−1.923 pp，[−13.462,+9.615]；R2 n=14，−0.893 pp，[−12.500,+9.821]。
- 三路径完整与 pair-specific 的估计对象不同，不能相互覆盖；P1/P2 prompt 不同，不能合并。
- R2 是同冻结 memory 的敏感性读取；P2 是 owner-amended route-selection evidence，不是原 P1 同 prompt 的独立确认。
- 当前证据既不支持切换主线，也不证明 Memory 无效或等价。修复完成应意味着可靠交付与可复核，不意味着科学结论升级。
