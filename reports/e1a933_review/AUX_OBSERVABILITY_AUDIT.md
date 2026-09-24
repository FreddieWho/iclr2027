# R05：辅助目标可观测性审计与生产修复

状态：生产修复和真实场景合同审计完成；新GPU N01科学结果见`VISION_GPU_RESULTS.md`，不能将本合同审计本身作为迁移有效证据。

原始训练集 `artifacts/discovery_campaign/scenes/train_101/scenes.npz` 的512个父场景全部检查。固定每场景noise seed，分别交换红线/蓝线内部端点（4元群），不交换可见颜色。标签恒定率100%；492/512个场景的整个轨道像素完全相同，所有2048个变换中98.046875%与原图完全相同。其余场景受旧renderer的方向相关栅格化舍入影响，最大像素差约0.8；因此旧renderer不能被无条件称为精确H不变。

rel10轨道方差平均为物理单位0.05394299747、orbit-pooled训练归一化单位0.47223474848。它描述隐藏命名冲突；只有完全相同图像、均匀命名的条件下才是相应平方损失下界，不是完整训练损失估计，也不能证明旧阴性的唯一原因。

生产 `u06_relation_distill.py` 新增整体四元匹配、轨道共享归一化、可匹配坐标控制和独立目标shuffle Generator。匹配使用一个完整结构的同一置换，禁止逐维独立匹配。批次Generator与模型/辅助层初始化分离；A保留10维不训练辅助头，评估正确提取tuple中的logit。旧有序MSE仍作为独立对照保留。历史固定lambda=.5不再称loss-scale matching。

新协议 `experiments/e1a933_review/vision_protocol.py` 在每种颜色内部按坐标排序端点再调用原renderer，不使用任务标签；全部512×4个变换严格像素相同。新O03/N01全部臂统一使用这一协议。排序只解决方向栅格化泄漏，不增加端点身份标记；不能把新旧模型作为相同观测协议的直接before/after。

生产回归：11个针对性检查全部通过，包括纯冻结BN参数/缓冲区哈希不变、重演旧net.train错误确实改变缓冲区、有序目标冲突被检测、整体匹配零误差、坐标各自匹配的错误反例、轨道归一化与独立shuffle。另用真实8张训练图跑过4个实际生产训练分支一epoch，属于工程回归，不是科学实验。

证据：`artifacts/e1a933_review/vision_regression/result.json`、`per_parent_observability.npz`；命令：`python experiments/e1a933_review/vision_regression.py`。最初生成但未训练的`vision_fresh224/`与`vision_canonical224/`为保留的工程准备产物；正式数据/训练仅`vision_canonical224_v2/`。
