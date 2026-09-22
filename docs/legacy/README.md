# docs/legacy/ — 历史一次性文件存档

本目录收录项目根目录清理（2026-09-22 结构整理）时移出的历史文件。
它们记录的是**路线调整前**（Action-Mode Spectrum / P3 / P4-AMR 时代）的
施工指令与交接快照，**不再反映当前项目状态**，保留仅供审计溯源。

| 文件 | 原位置 | 时代 | 说明 |
|---|---|---|---|
| `MASTER_AGENT_PROMPT.md` | 根目录 | P3-T5R 时代（2026-09-04） | 当时的总控 agent 启动 prompt |
| `AUTORESEARCH_P3_GOAL_PROMPT.md` | 根目录 | P3 时代（2026-09-01） | P3 autoresearch 目标 prompt |
| `PROJECT_PACKAGE_CONSOLIDATED.md` | 根目录 | 蓝图期（2026-09-05 打包） | 旧项目方案包合并版（Action-Mode/AMR 叙事） |
| `P1_FORMAL_TASK_HANDOFF.md` | 根目录 | 书法 P1 时代（2026-08-30） | 书法正式任务远端监测 handoff |
| `QA.md` | 根目录 | 蓝图期（2026-09-02 止更） | 架构/路线持续问答记录 |
| `MANIFEST_SHA256.txt` | 根目录 | 2026-08-30 包 | 当日报价的文件校验清单（对应文件此后已演变） |

⚠️ 注意：
- `scripts/audit_current_state_consistency.py` 是 T5R 时代的一致性审计脚本，
  按原根目录路径硬编码引用上述部分文件；本脚本本身已是历史工具，
  不在终局复算链（`reports/final_closure/REPRODUCE_FINAL.md`）内，移动后其失败无影响。
- 当前项目叙事与权威文档入口见根目录 `README.md` 与 `PROJECT_MAP.md`。
