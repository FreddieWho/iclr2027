# M2 后置交接

用户最新指示：先完成其他工作并推送 GitHub，M2 补算后续再处理。本轮不等待、不收集或解释新 M2 指标。

- 已完成：旧运行脚本与 Git 基准 SHA 一致性确认；修复冻结/微调梯度、BN 状态、初始化时序、静态曝光和 CI 估计量；8 项针对性检查通过。
- 已启动：同一已有 2080 Ti 上的完整 16 格，任务锁 `artifacts/submission_audit_20260925/GPU_M2_JOB.json`；每个训练格 20 epochs，692 quartets / 351 parents，旧 bank/head/权重哈希不变。
- 当前接受状态：`DEFERRED_PENDING_ACCEPTANCE`。任何后台新数值均未计入本轮结论。
- 旧错误解释已经撤回。旧产物保留，不能用修复后的代码为旧训练背书。

## 远端恢复位置

主机：本次用户提供的 GPU 实例（连接信息不写入公开仓库）。

```text
/root/iclr_submission_audit_20260925/GPU_M2_JOB.json
/root/iclr_submission_audit_20260925/m2.launcher.log
/root/iclr_submission_audit_20260925/m2.pid
/root/iclr_submission_audit_20260925/m2_run/RUN_RECEIPT.json
/root/iclr_submission_audit_20260925/m2_run_evidence.tar.gz
/root/iclr_submission_audit_20260925/m2_run_checkpoints.tar.gz
```

下一轮先读取终态回执；若 FAILED，保留错误和已完成格，不改旧目录。若 SUCCEEDED，下载两类包及各自哈希，使用 `collect_gpu_artifacts.py` 验证；按原 bank parent 顺序独立重算 J3/J4/CCM、配对 parent-cluster CI，核查 frozen 权重/BN 不变和 tuned 非零梯度/参数变化，再决定科学措辞。不要跳过这一步直接引用 remote result.json。

runner、common metrics 的启动哈希锁在 GPU_M2_JOB.json；运行中不会读本机后来修改的源码。原始数据与旧运行都没有覆盖。
