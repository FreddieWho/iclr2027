# ICLR 2027｜发现型探索施工包
日期：2026-09-17。仓库基准：`FreddieWho/iclr2027@f86dbdf6107bad3d156b250a3c23bd5dcd8bb9d2`。

## 本包的任务
不是把现有文章补得更严密，而是寻找能改变文章核心主张的结果。允许换任务、换干预分布、换方法、换主要图；已有资产用于降低试错成本，不要求新结果继续服务 AMR。

**首选故事：每个构件受到的扰动分布相同、每次扰动总能量相同，甚至球队质心也不变，仅改变协同方式，模型是否会从可靠变得不可靠？**

反向故事：内部关系已经改变，模型却几乎不改变表示。方法赌注：改变训练见到的联合扰动结构，或在冻结骨干上重新配置可访问的信息，而非再搜索温度、margin、pooling 配比。

本包包含六条研究路线、两个方法分支、共享数据/评价接口、并行任务图、可运行的数学内核与测试、文献差异表、论文替换方案。**六条路线是候选组合，不要求全部塞进一篇论文。**

## 使用
将整个目录放到仓库 `docs/iclr2027_discovery_campaign_20260917/`，向总控 agent 提交 `MASTER_AGENT_PROMPT.md`。不要把包内 AGENTS.md 覆盖为仓库根 AGENTS.md。

```bash
cd /path/to/iclr2027
export PACK="$PWD/docs/iclr2027_discovery_campaign_20260917"
python "$PACK/scripts/bootstrap_run.py" --repo "$PWD" --output "$PWD/artifacts/discovery_campaign/initial_inventory"
python -m unittest discover -s "$PACK/tests" -v
python "$PACK/scripts/smoke.py" --output "$PWD/artifacts/discovery_campaign/kernel_smoke"
```

`bootstrap_run.py` 只找入口、探测文件和记录当前 commit，不运行旧审核链。`smoke.py` 只验证受控构造及内核，不产出项目科学结论。六条路线的仓库 adapter、模型实验和新任务训练由 agent 依照文档实现；本包不是六个已经训练完成的项目。

## 阅读顺序
先读 `MASTER_AGENT_PROMPT.md`、`01_STRATEGY.md`、`02_ROUTE_MAP.md`；worker 读自己的路线卡和 `03_SHARED_ENGINE.md`。数据与模型入口在 `05_DATA_MODELS.md`。任务依赖在 `configs/tasks.json`，结果记录见 `examples/route_result.example.json`，对应轻量schema在 `configs/result_schema.json`；分工指令可用 `prompts/WORKER_PROMPT.md`。

## 六条路线
| ID | 新问题 | 优先角色 |
|---|---|---|
| R01 | 固定个体扰动分布后，联合依赖变化是否导致大幅失效？ | 主发现，先做 |
| R02 | 可以改变任务关系，却让表示近乎不变吗？ | 独立高风险主发现 |
| R03 | 少数固定协同模式是否跨场景、模型迁移？ | 主发现的机制和简洁性升级 |
| R04 | 只改变联合增广分布，能否修复 R01？ | 方法路线一 |
| R05 | 不更新骨干，用少量关系反例能否改变可访问性？ | 方法路线二 |
| R06 | 同边际协同变化是否造成长期预测/决策的非预期崩溃？ | 高风险重要性升级 |

默认先让 R01、R02、R05 出首轮数据，R06 完成最低成本试跑；R03 与 R04 使用初始共享模式库即可开工，不等待 R01 显著。总控根据发现把算力集中到一主故事和一方法；不设置统一 p 值或效果量门槛。

## 实验边界
本包中的新效应、胜率提升、迁移性都尚未测得。所有数字型标题均是填空模板，不是预测结果。历史数据与结果只读，探索输出另存。没有新增付费算力或 API 额度授权；使用既有明确授权资源，不以资源确认阻塞本地可行分支。

## 交付检查
已通过24项内核/接口测试。详见 `VALIDATION.md`。连续阅读可用 `CAMPAIGN_CONSOLIDATED.md`；实际执行仍以各独立路线文件为准。
