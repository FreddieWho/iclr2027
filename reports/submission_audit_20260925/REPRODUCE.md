# 本轮复算与交付检查

从仓库根运行。仅下面的索引命令默认是轻量文档/哈希操作：

```bash
python experiments/submission_audit_20260925/build_evidence_index.py
```

各组 AUDIT.md 记录其独立复算命令与真实运行结果，代码位于 `experiments/submission_audit_20260925/`。不要自动运行历史报告中的全训练命令或重新打开封存数据。

## 已回收的 route2 GPU 补算

任务锁：`artifacts/submission_audit_20260925/GPU_ROUTE2_JOB.json`。
结果：`artifacts/submission_audit_20260925/gpu_route2/out/static/`。
回收验收：`gpu_route2/COLLECTION_CHECK.json`，含预测、回执和 3 个检查点，共 16 个文件逐个校验。便于 Git 阅读的预测表另存为 `gpu_route2/PREDICTIONS.csv`。

已下载的压缩包可用以下命令复核（不连接服务器，不训练）：

```bash
python experiments/submission_audit_20260925/collect_gpu_artifacts.py   --directory artifacts/submission_audit_20260925/gpu_route2 --stem route2_run   --evidence-sha256 ca3925484177aa2265202f37aff270a624f1f6ce2b2eff5d8091f2d61cfb8f73   --checkpoints-sha256 5fbf3c8265b65e07f860dfd3c0a012b0ccb1e9c91c1f3fde98ce48b3e568ef77
```

## Figure 1

```bash
MPLCONFIGDIR=/tmp/iclr2027-figure-cache python experiments/e832_focus/route5_figure/select_figure1.py   --receipt reports/submission_audit_20260925/FIGURE1_SELECTION_RECEIPT.json
```

这会更新当前图；使用历史 dev512，不换 seed 或例子。显示值核验见 `artifacts/submission_audit_20260925/FIGURE1_AUDIT.json`。

## M2 后置交接

见 `M2_DEFERRED_HANDOFF.md`。任务已经启动；不要再次运行同一训练命令或覆盖目录。完成训练不等于科学验收。
