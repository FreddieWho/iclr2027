# P1 正式任务定时监测 Handoff

## 任务

- 项目：ICLR2027 中文书法探索性复现
- 正式配置：800 字符、25 epochs、seeds `11 23 47`
- 运行：RTX 3080、CUDA AMP、推理 batch 256、训练 batch 16、干预 chunk 128
- 当前交接快照：任务尚未完成；上次检查处于 GAT 响应计算阶段，未见 OOM

## 远端唯一标识

- SSH：`vipuser@223.109.239.11:13212`
- 认证：只从安全凭据存储读取；本文件不含密码
- 主机密钥：ED25519 fingerprint `SHA256:8bfoytm5mRuq131E5NqSZeCGlpx8nucGuSek+R4xbao`
- systemd unit：`iclr2027-calligraphy-p1-oomfix-20260830.service`
- 输出目录：`/home/vipuser/codex-jobs/calligraphy-3080-formal-oomfix-20260830`
- 项目报告：`/home/vipuser/iclr2027/reports/CALLIGRAPHY_PILOT_REPORT.md`

## 定时检查（只读）

每次检查记录时间，并读取：

1. `systemctl --user show` 的 `ActiveState/SubState/MainPID/Result/ExecMainStatus/MemoryCurrent/MemoryHigh/MemoryMax`
2. `free -h`、`nvidia-smi`
3. `journalctl --user -u iclr2027-calligraphy-p1-oomfix-20260830.service -n 30 --no-pager`
4. 输出目录的文件列表、大小和修改时间

不使用 `kill`、不重启、不修改脚本或输出，不扩大实验规模。`MemoryMax=28G` 是保护上限。

## 成功判据

只有同时满足以下条件，才标记 `SUCCEEDED`：

- unit 已正常结束：`inactive/dead`，`ExecMainStatus=0`；
- `intervention_validation.json.status == "pass"`；
- `pipeline_summary.json.status == "pilot_complete"`；
- `model_manifest.json` 含 6 个模型且全部 `status == "complete"`；
- 存在完整的 `spectrum_response.parquet`、`spectrum_summary.parquet`、`errr.csv`、`mode_accessibility.csv`、6 个模型 response parquet 和 6 个 baseline 文件；
- `CALLIGRAPHY_PILOT_REPORT.md` 已生成。

unit 仍为 `active/running` 时只能标记 `RUNNING`，不能因中间文件齐全而提前结算。

## 失败判据

任一条件成立即标记 `FAILED/INCOMPLETE`，不要生成科学结论：

- unit 非零退出；
- 日志出现 `oom-kill`、`Out of memory`、`Killed process` 或 CUDA OOM；
- 成功判据中的任一最终文件缺失、损坏或状态不完整。

失败时保留现场并报告 unit 状态、最后日志、内存证据和已有文件；不要用减少字符、seed、epoch 或干预行数的方式“修复”。

## 完成后的回收顺序

仅在 `SUCCEEDED` 后执行：

1. 将输出目录和项目报告回收到本地新的归档目录；先生成远端文件清单与 SHA-256，再传输并复核 checksum。优先使用 `lan_ssh_workflow` 的 collect 流程，禁止覆盖已有归档。
2. 本地校验通过后，停止该 systemd unit（若已结束则无需重复停止）。
3. 调用 AI Galaxy 的实例释放/结算流程；不要开启自动续租，不要删除与本任务无关的磁盘。
4. 释放后记录 provider 返回的 `costs`、实例最终状态和 `kept_disks`；若释放或结算失败，保留证据并报告，不宣称已完成。

## 隔离边界

本任务只涉及中文书法矢量数据和点集模型。禁止访问、更新或重建 `infra/bioinf-data-index/` 及任何生信索引。
