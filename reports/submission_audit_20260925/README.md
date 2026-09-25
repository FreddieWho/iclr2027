# 最终投稿前证据目录

日期：2026-09-25。基准 `3fc9757`。本轮交付为结论与代码审查、必要修复及低成本补算；不是最终文章。

**M2 修正补算按用户最新指示后置。** 本轮交付其代码修复与旧结论撤回，不接纳后台新数值。其余结果按各组审查记录使用。

## 下一轮写作顺序

1. [FINDINGS.md](FINDINGS.md)：哪些错误已修、哪些解释必须改变。
2. [CLAIMS.csv](CLAIMS.csv) / [CLAIMS.json](CLAIMS.json)：每项结论的代码、证据、分母、允许措辞和限制。
3. [METRIC_CONTRACT.md](METRIC_CONTRACT.md)：J3、J4、CCM、修复流与重采样单位。
4. [GAPS.md](GAPS.md)：不能写成已完成或已证实的内容。
5. [REPRODUCE.md](REPRODUCE.md)：复算与回执入口。
6. [paper_map/INVENTORY.csv](paper_map/INVENTORY.csv)：现稿主文和附录的结论覆盖。

## 可写的主线

- 历史 fresh holdout 的条件组合漏检为 151/176；本轮核已发布汇总，没有重开封存池。
- 原始坐标的端点修复常伴错误迁移；几何输入与部分视觉对照能增加完整修复。分银行、基线子集和训练配方报告。
- 源任务解析器可达 J3=1；学习表示结果属于这个解析基线限定下的诊断。
- 跨任务输入优势不等于正交互跨任务成立；N02 的宽网对照是描述性结果，仍未识别采样机制。
- M2 原“解冻无变化，因此接口不可学／排除容量”的解释已经撤回。

## 分组证据

| 分组 | 审查记录 |
|---|---|
| 核心结论 | [core/AUDIT.md](core/AUDIT.md) |
| 源表示与 M2 实现 | [mechanism/AUDIT.md](mechanism/AUDIT.md) |
| 路线 1–4 与曝光匹配视觉基线 | [routes/AUDIT.md](routes/AUDIT.md) |
| N02 与视觉边界 | [vision/AUDIT.md](vision/AUDIT.md) |
| 控制、血缘、足球及部署 | [controls/AUDIT.md](controls/AUDIT.md) |
| 稿件覆盖与引用 | [paper_map/AUDIT.md](paper_map/AUDIT.md) |

`ASSET_AVAILABILITY.csv` 区分 Git 跟踪文件、本地大资产和目录引用。哈希不等于公开下载。复算所依赖的历史数据和检查点不应被描述成全部已随 Git 发布。

本次交付保留在本地；按用户最新指示，不创建 Git commit、不推送 GitHub。
