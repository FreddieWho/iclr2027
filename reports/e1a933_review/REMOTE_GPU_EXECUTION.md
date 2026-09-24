# 远程视觉GPU执行（2026-09-24）

用户已提供部署完毕的实例并明确要求开始；本轮没有租机或新增环境安装。实际是RTX3080Ti12GB，非预设V100。SSH端点为connect.nmb2.seetacloud.com:30891；凭据不写入报告或任务文件。

## 实际环境与合同

- Python3.12.3；PyTorch2.5.1+cu124，torchvision0.20.1+cu124，NumPy2.1.3，CUDA12.4，driver595.71.05。
- 容器资源上限10CPU/30GB RAM；任务torch8线程。GPU总12288MiB。
- 原冻结数据SHA25663151a7670d2cadcab34d7f52b88f488e27ea0fb04ff4fcdd421d064efca0de6；离线包45,235,217bytes，传输校验通过。
- 3seed×7arm=21次；每次20epoch，batch32，FP32，AMP/TF32关闭；仅single-dev BCE选模型。新backend完整矩阵单独输出，不拼接CPU结果。

## 当前执行证据

远端根：`/root/autodl-tmp/e1a933/vision_20260924_ssh30891/`。
CUDA前后向和finite梯度检查已完成，预检peak allocated899,511,808bytes（约.84GiB）。这个峰值不是完整Adam/辅助训练峰值。
监督进程PID1971，训练进程PID1972，均脱离SSH终端；任务声明为`jobs/vision-21arms-gpu-20260924/vision-21arms-gpu-20260924.json`。
启动验收已完成pretrained_static/803与pretrained_flip/803两个完整20epoch训练臂，首个checkpoint SHA256实读吻合；已进入random_static/803。最近epoch5.4–5.8秒，GPU进程占用约1622MiB，stderr空。完整21臂暂估约1小时，辅助臂/最终评估尚需实测校准。

实时回执：`cuda_matrix/receipt.json`和任务目录`status.json`。训练日志为任务目录`task.stdout.log`/`task.stderr.log`。这些活文件优先于本报告启动快照；不能由RUNNING推断完成。

## 持久运行与完成产物

监督器不修改训练代码，等待训练结束后验证21个result/checkpoint哈希与预测文件，生成逐文件manifest、`cuda_matrix_results.tar.gz`和SHA256回收清单。只有训练回执MATRIX_COMPLETE且21臂产物完整，才写SUCCEEDED标记。否则明确FAILED并保留stderr，不把部分输出当完成。

该CUDA训练器不支持优化器中间态续训：同目录重启拒绝覆盖。若失败，保存失败目录，修复后另目录执行，不能把只加载best权重叫断点恢复。

本地自动回收器已启动（最新PID见COLLECTOR_LAUNCH_V3.json，状态以COLLECTION_STATUS.json为准）：基于Linux inotify完成文件事件的阻塞等待，不做高频轮询；待远端成功后按清单下载receipt、paired_analysis与完整归档并复核SHA256。计算/传输完成仍不等于科学判决；N01/O03效果及N03启动须再独立解读。

本地回执位置：`artifacts/e1a933_review/remote_vision_20260924_ssh30891/`。本地 CPU 训练曾作为兜底保留；**已于 2026-09-24 14:53Z 按用户指示停止**（完成 19/21 臂，产物未删；见该目录的 `TERMINATION_RECEIPT.json`），因为远端矩阵已完成进稿且 CPU/GPU 结果不得混表。其冻结源码与旧产物未改。

## 运行命令

```bash
/root/miniconda3/bin/python -u /root/autodl-tmp/e1a933/vision_20260924_ssh30891/vision_cuda_bundle/vision_cuda_run.py --bundle /root/autodl-tmp/e1a933/vision_20260924_ssh30891/vision_cuda_bundle --out /root/autodl-tmp/e1a933/vision_20260924_ssh30891/cuda_matrix --execute
```

此为已启动命令，禁止重复启动到同目录。完整任务与源代码hash见本地DEPLOYMENT.json、START_STATUS.json及tasks/；主机公钥首次沿用已有accept-new设置记录后，所有后续连接使用固定known_hosts与StrictHostKeyChecking=yes。

当前科学状态仍为RUNNING/UNRESOLVED，未把2个完成臂当完整21臂证据。启动快照见FIRST_ARM_STATUS.json，自动回收状态见COLLECTION_STATUS.json；全部结果回收后仍需科学分析与论文更新。

回收器兼容性修正：最初两次启动失败已保留（命令传递/远端缺os.pidfd_open），现改inotify+select，不设定时器，不做训练状态轮询；GPU训练未受影响。
