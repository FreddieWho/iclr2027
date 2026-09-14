# ICLR Memory Pilot V4 — Fast Route Decision Pack

## 目的

本包只服务一个决定：**Compression-Path Memory 是否值得取代 Action-Mode，成为当前 ICLR 主线。**

它不是发表级 benchmark 施工包，也不是为了修复 V3 的所有技术债。当前项目尚未得到任何有效的 direct-vs-staged utility 结果；V3 的 `NO_GO/TECHNICAL` 只说明原长度预检协议没有打开科学实验入口，不能解释为科学假设阴性。

## 这次的核心修正

1. **停止 V3 剩余 35 格长度矩阵。** 不再花调用补一条已经不能回答科学问题的技术曲线。
2. **取消“严格 1K/2K native-token 自然合规”作为 route-selection 的前置门。** V4 使用相同 requested target + 自然结束，并记录真实 final length；通过长度匹配和回归判断 path effect 是否超出 endpoint length 差异。
3. **C1 允许关闭 thinking。** 这是 V4 的新冻结 compressor 配置，不与旧 V3 混合。科学假设依赖的是相同 compressor 下的 path，而不是 reasoning mode。
4. **只保留一个主 final target。** 默认 4,096 native tokens；若 2-history 联调失败，只允许一次切到 8,192。禁止再次扫描五个以上预算。
5. **探索 12 历史 + 新确认 24 历史。** 这是路线决策样本，不冒充发表级证据。
6. **只做 direct、staged2、rewrite 三个核心 memory path。** raw/oracle/no-memory 只做仪器校准。
7. **第二 reader 复读冻结 memory** 作为廉价泛化检查；不立即启动第二 compressor。
8. 最终必须输出 **GO / NO_GO**。TECHNICAL 只在一次允许的技术调整后仍无法产生可评分 memory 时使用。

## 启动顺序

1. 将整个包复制到当前 Memory Pilot 仓库根目录下的 `docs/v4_fast_decision/` 或保留为外部只读指导包。
2. 把 `MASTER_AGENT_PROMPT.md` 交给总控 agent。
3. agent 先阅读：
   - 当前仓库 `PROJECT_PROGRESS_SUMMARY_20260913.md`；
   - 本包 `docs/01_PROGRESS_DIAGNOSIS_AND_CORRECTION.md`；
   - `docs/02_V4_FAST_SCIENCE_PROTOCOL.md`；
   - `docs/04_ROUTE_DECISION_RULES.md`；
   - `docs/05_MIGRATION_FROM_V3.md`；
   - `configs/v4_fast_decision.json`。
4. 不要求用户再次确认本包已经明确授权的协议修改；只有 API key 缺失、provider 不可用或现有项目 cap 不足才允许阻塞。

## 必须交付

- `reports/V4_PREFLIGHT.md`
- `artifacts/v4_exploration_rows.*`
- `artifacts/v4_confirmation_rows.*`
- `reports/V4_FAST_PILOT_REPORT.md`
- `reports/V4_ROUTE_DECISION.json`
- 所有 frozen config / prompt hash / tokenizer revision / cost ledger

## 决策解释

- **GO**：Memory 获得足够强的 route-selection 信号，暂停 Action-Mode 主施工并转入发表级 Memory 验证。
- **NO_GO**：不把更多时间投入 Memory；回 Action-Mode，并使用之前独立审计备忘中的修正路线。
- **NO_GO_TECHNICAL**：只表示当前可执行条件无法评估科学假设；不得写“compression path 不存在”。
