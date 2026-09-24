# 视觉执行回执

> 状态更新：用户已提供远程实例并授权执行，完整GPU矩阵现已启动；本文件保留原规划/CPU历史，当前以 [REMOTE_GPU_EXECUTION.md](REMOTE_GPU_EXECUTION.md) 为准。
正式目录：`artifacts/e1a933_review/vision_canonical224_v2/`；原子状态`receipt.json`；日志`runner.log`。本机8 CPU线程，无CUDA、无下载/租机。持久进程从host启动，PID1886333。

> **已停止（2026-09-24 14:53Z，用户指示）**：远端 GPU 矩阵完成、回收、校验并进稿后，该兜底任务已无用途，且本项目规定 CPU/GPU 不得混表，故即使跑完也不进任何报告数字。
> 停止时完成 **19/21 臂**（`matched_s806` 仅有 `history.json`、无 `model.pt`；`shuffled_matched_s806` 未启动），**未删除任何产物**。
> `receipt.json` 现为 `STOPPED_BY_USER`；原始 RUNNING 回执按字节保留为 `receipt.RUNNING_asof_stop.json`（sha256 `8dcc9e7f…`）；停止记录见 `TERMINATION_RECEIPT.json`。
> 本文件其余内容为**启动时快照**，其中的“约9–12小时”等为当时估计，不作完成承诺。

实际命令：
```bash
python -u experiments/e1a933_review/vision_run.py --out artifacts/e1a933_review/vision_canonical224_v2 --epochs 20
```

初次prepare-only已完成；随后host subprocess.Popen(start_new_session=True, stdin=DEVNULL)持久启动上述命令。沙箱内nohup尝试未留下运行进程；真正启动为host PID1886333。首次基准8线程224px batch16前后向0.479秒；正式epoch1实测73.24秒，总21臂约9–12小时，以后续真实epoch耗时为准。status=MATRIX_COMPLETE仅代表矩阵及自动配对分析完成，科学解释仍另列PENDING_INDEPENDENT_INTERPRETATION。

恢复要求source/data/matrix哈希一致，否则拒绝混入旧结果；已经写出的result.json仅在该合同核对后跳过。没有完成的臂不覆盖，应检查FAILED回执后另开可审计恢复目录。源文件快照保存source_snapshot/；checkpoint按dev最低BCE选，测试J不可用于选择。工程回归命令：`python experiments/e1a933_review/vision_regression.py`。

O03/N01执行中；N03 NOT_RUN（匹配预测是否学会及是否无任务增益尚未知）。N02由独立sampling worker负责，见其专用报告。

归档factorial复算：`python experiments/e1a933_review/vision_factorial.py`。
