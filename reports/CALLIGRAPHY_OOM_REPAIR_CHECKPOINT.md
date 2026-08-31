# 书法受控 probe：OOM 修复与重启 checkpoint

日期：2026-08-30

## 当前状态

正式任务已重启，状态为 `RUNNING_NOT_COMPLETE`。在正式输出、完整 3 seeds 和 25 epochs 完成前，不做科学结论。

## 事故证据

- 旧正式任务在 16c/32GB 主机上被 Linux 内核杀死，原因是主机 RAM OOM，不是 CUDA 显存异常。
- 内核记录的被杀进程为 `pt_main_thread`，`anon-rss=31801764 kB`，约 30.3 GiB；旧任务没有形成可用正式结果。

## 修复

- 干预定义、字符顺序、随机种子、能量、频段、模型结构和正式规模均未改变。
- 干预 manifest 改为 128 行有界分块生成、校验和 Parquet 写盘，不再把全部 JSON 与 `_delta_array` 留在内存。
- 响应计算改为从 manifest 分块读取、分块推理和分块写盘；模型之间显式释放对象和 CUDA cache。
- 公开 manifest 与响应字段保留；新增的 chunk 参数只控制内存上界，不减少行数。

## 验收证据

- 本地：`py_compile` 通过；`tests/test_calligraphy_probe.py` 为 10 passed。
- 远端：RTX 3080 10GB、CUDA+AMP 的 24 字符 smoke 完整通过，用时 17.3 秒，生成 2,689 条干预和 5,318 条响应，无 OOM。
- 正式配置：800 字符、25 epochs、seeds `11 23 47`、CUDA+AMP、推理 batch 256、训练 batch 16、干预 chunk 128。
- 资源保护：systemd 用户服务 `iclr2027-calligraphy-p1-oomfix-20260830.service`，`MemoryHigh=24G`、`MemoryMax=28G`；启动约 14 秒时进程 RSS 约 619MB，主机可用内存约 29GB。
- 远端输出目录：`/home/vipuser/codex-jobs/calligraphy-3080-formal-oomfix-20260830`。

## 隔离边界

本次只处理中文书法矢量数据与点集模型路径；没有访问或更新 `infra/bioinf-data-index/`，也没有引入生信数据。
