# 本目录 agent 规则
这份规则只服务新探索包，不覆盖仓库根安全/费用规则。科学优先级看 MASTER_AGENT_PROMPT。

总控负责目标、依赖、资源分配和论文取舍；worker 独占自己的 `experiments/discovery_campaign/Rxx/` 和 `artifacts/discovery_campaign/Rxx/`。共享内核的改动由总控合并，避免并发覆盖。可以使用 worktree，但不自动 push、删除分支或覆盖历史结果。

每个 worker 读自己的路线卡、03_SHARED_ENGINE、05_DATA_MODELS 即可开工，不要求通读全部历史报告。历史脚本可以复用计算函数，不复用与本轮无关的冻结/审批逻辑。数据明确可用后直接工作。

结果卡用 examples 模板；`status` 只用 `not_started/running/interesting/negative/implementation_blocked/pivoted/closed`。不把 `interesting` 写成 confirmed。所有初始例子和 smoke 均标 engineering_only。
