# Worker C 执行与复现

本轮完成的真实命令（原始默认产物保留）：

```bash
.venv/bin/python experiments/e1a933_review/fair_continuation.py
.venv/bin/python experiments/e1a933_review/fair_summarize.py
.venv/bin/python experiments/e1a933_review/football_repair.py --events-per-match 80
.venv/bin/python experiments/e1a933_review/football_summarize.py
.venv/bin/python experiments/e1a933_review/frontier_repair.py
.venv/bin/python experiments/e1a933_review/test_fair_football_frontier.py
.venv/bin/python experiments/f095_campaign/test_d10_noleak.py
.venv/bin/python experiments/f095_campaign/test_d06_oracle.py
```

重复训练/计算必须换新目录；三个 runner 默认遇到已有结果会 FileExistsError，防止覆盖本轮或不完整运行。以下目录也须从未使用：

```bash
.venv/bin/python experiments/e1a933_review/fair_continuation.py --out /tmp/e1a933_fair_reproduce_01
.venv/bin/python experiments/e1a933_review/football_repair.py --events-per-match 80 --out /tmp/e1a933_football_reproduce_01
.venv/bin/python experiments/e1a933_review/frontier_repair.py --out /tmp/e1a933_frontier_reproduce_01
```

两个 summarize 的 `--out` 指向读取的产物目录（默认本轮目录），会重建该目录的派生统计和仓库内相应专报，故只在需要重新出报告时执行。运行轨迹与具体预算在3份专报和对应JSON。旧 `artifacts/f095_campaign/` 与所有输入checkpoint没有写入。

完成收据：`artifacts/e1a933_review/WORKER_C_RECEIPT.json`。20项相关测试通过；旧max-only/PAVA/跨半场逻辑反例被回归显式拒绝。保护默认目录的3项真实入口检查均非零退出。初始状态hash断言验证6个encoder×seed组内4个续训臂完全相同。

未运行：U07全dose网格/独立sealed确认、U05新训练、自然轨迹E发生率、自然缺测学习模型和神经关系对比、全部256场景无oracle筛选部署策略、O06外部原生多模态。它们不能由本轮合同回归或几何探针替代。
