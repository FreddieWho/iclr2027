# 本轮 GPU 补完与关机条件

最新状态：**SAFE_TO_SHUTDOWN**，87个训练臂与272次VLM请求均已回收校验。N02独立解读/论文整合仍待本地完成，不需要GPU。释放检查脚本的文件名/重复键错误已修复并成功重跑，原失败回执保留。

用户于2026-09-24明确要求：剩余需要GPU的工作本次一起完成，再关闭实例。使用已提供的RTX3080Ti 12GB，不新增租机或付费API。

## 已安全回收

O03/N01完整21臂于16:54北京时间成功结束，用时40分45秒；16:56完整回收876,199,984字节压缩包并通过SHA256。校验回执：`artifacts/e1a933_review/remote_vision_20260924_ssh30891/COLLECTION_STATUS.json`。

## 追加范围

| 工作 | 冻结范围 | 状态 |
|---|---|---|
| N03决策瓶颈 | 几何监督/同容量通用瓶颈×3seed；20epoch、batch32、FP32；解析像素基线和oracle替换诊断 | 6/6完成、回收/内部manifest与checkpoint校验通过；专报完成 |
| N02采样机制 | 48臂：分辨率/stride/pretraining主对照，native224与低通观察对照，等曝光static-only对照；fresh parent数据 | 48/48完成、完整回收并逐文件校验；四配置检查通过，最高allocated2.07GiB |
| O05足球受控缺测 | raw/typed ×static/change ×3seed，按真实比赛隔离，保留因果基础模型 | 12/12完成、回收校验；不等于自然缺测已解决 |
| O06原生多模态 | Qwen2.5-VL-3B-Instruct固定revision；64quartet×4状态和16sanity请求，greedy解码 | 272/272完成、回收校验；权重通过官方SHA256对照 |

N03启动依据是辅助目标学习证据而非分类结果择优：matched开发集几何MSE三seed均低于ordered/shuffle，但分类J增益跨seed不一致。它不证明几何已精确恢复。

O01/O02扩展与R03无GPU必要依赖；历史回执显示小MLP适合CPU，血缘核查依赖记录与坐标匹配。它们不会阻止GPU关机。

## 执行与释放

远端根为`/root/autodl-tmp/e1a933/gpu_finish_20260924`。各实验独立目录；文件锁串行占用GPU；监督器等待进程退出后检查完成回执并打包。自动回收器通过inotify完成事件唤醒，不设定时轮询。模型下载可与训练并行，但推理与训练串行。

只有全部实际启动的GPU任务结束、必需checkpoint/逐样本输出回收且SHA256通过，才确认可关机。解释、论文、CPU分析可在本地继续；机器空闲本身不等于科学任务完成。遇到不可满足的合同或失败，保存失败证据并明确未跑范围，不静默替换成更弱任务。

N02于北京时间17:32开始计算；四配置实测训练step展开2781秒，含评估/渲染/打包回收暂估约1小时。该估时不是完成回执。

本地持久完成处理器PID2128789仅等待COLLECTION_STATUS文件事件，成功回收48臂后核验全部4项产物，再生成GPU_RELEASE_STATUS.json（SAFE_TO_SHUTDOWN）和GPU_RELEASE_RECEIPT.md；无自动机器关机命令。

运维修复保留失败事实：最初VLM官方直连不可达，改公开镜像并校验两个官方权重SHA；N03/O05首个本地collector使用旧Python缺tomllib，已改现有Python3.11，远端训练未受影响；旧O05仅排队未运行时worker补齐版本签名，取消旧队列后用独立v2重新提交。
