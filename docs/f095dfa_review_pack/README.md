# f095dfa综合评估与20条施工路线

固定基准：`FreddieWho/iclr2027 @ f095dfaa1341a057efee52c56fe2aea9c39ded6a`。

## 阅读顺序

1. [整体评估、三项顾虑与主观投稿判断](00_PROJECT_REVIEW.md)
2. [交给总控agent的执行指令](01_AGENT_MASTER.md)
3. [10条缺陷补足：完整合并文档](02_DEFICIT_10_ROUTES.md)
4. [10条优势拔高：完整合并文档](03_UPGRADE_10_ROUTES.md)
5. [来源与已有方法边界](reference/SOURCES.md)

单任务worker直接读取 `deficits/D01.md`至`D10.md` 或 `upgrades/U01.md`至`U10.md`。机器可读索引在 `ROUTE_INDEX.json` / `ROUTE_INDEX.csv`。

本包提供的是20份可选择执行的研究合同，不要求同时执行，更不要求全部阳性。新增代码路径是建议创建的入口；仓库现有代码需先核对实际接口。重模型计算/付费资源由现场测量和用户预算决定。

## 附带工具（已做设计级测试，未复算仓库实验）

`checks/threshold_certificate.py`：对二元多状态实例，精确计算局部可分性S、全局单阈值oracle上限J*、当前J及三段差距；支持非负实例权重。只依赖NumPy。

输入NPZ需含 `scores[n,k]`、`labels[n,k]`，可含`weights[n]`。每个实例要同时含两类标签。

```bash
python checks/threshold_certificate.py --input YOUR_SCORES.npz --threshold 0 --output NEW_SUMMARY.json
cd checks
python -m unittest -v test_design_invariants
```

9项设计测试已通过，日志 `checks/DESIGN_TEST_LOG.txt`。测试包括区间恒等式、半开端点、严格单调变换、暴力扫描对照、输入检查、8元素群闭合和oracle不变、全S4错误反例、群平均不变性。它们不是原项目212项测试，也不确认本项目的实证效应。

## 必须记住

较大的效应不等于较大的科学外推范围。优先建立一个能排除简单解释、能预测新结果的中心，不继续积攒20份同质小阳性。
