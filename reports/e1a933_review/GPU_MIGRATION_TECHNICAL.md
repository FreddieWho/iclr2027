> **[历史文档 · 2026-09-24 补注]** 本文件写于 GPU 迁移之前，描述当时的规划与 NOT_RUN 状态。
> GPU 实际已执行：原21臂 + N03六臂 + O05十二臂 + N02四十八臂 = 87 训练臂与 O06 272 请求全部回收并逐文件校验。
> 现行状态见 `R_FIXES_REPORT.md`；释放回执见 `artifacts/e1a933_review/gpu_finish_20260924/GPU_RELEASE_STATUS.json`。

# CUDA迁移技术准备（仅规划）

> 状态更新：用户已提供远程实例并授权执行，完整GPU矩阵现已启动；本文件保留原规划/CPU历史，当前以 [REMOTE_GPU_EXECUTION.md](REMOTE_GPU_EXECUTION.md) 为准。
当前CPU完整矩阵继续运行，未停止、未改动冻结源码。GPU租用、传输和真实训练均未执行；此交付不构成付费或远端执行授权。

独立入口：`experiments/e1a933_review/vision_cuda_run.py`；独立生产移植：`vision_cuda_protocol.py`。默认仅preflight，必须显式`--execute`才启动完整矩阵。CPU生成的冻结bank直接按SHA256复用，不在GPU重造数据。3seed×7arm、20epoch、batch32、Adam3e−4、FP32、single-dev BCE选checkpoint保持不变；AMP/TF32关闭，不以OOM为理由偷偷改batch/precision。CPU输入插值/标准化保留相同实现，再传到GPU。

推荐完整21臂全在同一GPU backend重新运行，独立输出目录并记录backend/dtype；不能把CPU已完成模型与GPU剩余模型拼成主比较。GPU浮点执行和优化器内部实现仍可带来数值差别，因此不得承诺跨backend位级等价。没有迁移CPU训练中间态；仅迁移固定数据及ImageNet初始权重。

## 最小传输包

`artifacts/e1a933_review/vision_cuda_bundle.tar.gz`

包含两个独立Python入口/协议文件、冻结data.npz、data manifest、噪声无关查重回执、已缓存ResNet18 ImageNet权重和README；无旧模型或CPU新训练checkpoint，无数据下载需求。每个文件的字节数与SHA256见bundle/manifest.json，归档回执见vision_cuda_bundle_receipt.json。

## 实际检查与未跑项

CPU移植一致性检查已实跑：pretrained_static、pretrained_flip、matched、shuffled_matched四个生产分支，各用8张真实训练图跑1epoch，与原CPU生产函数的最终测试logit逐位相同，最大差均0。其结果只是代码移植的CPU回归，不是科学结果，更不是CUDA验证。

本机CUDA preflight实际调用返回`NOT_RUN_NO_CUDA`。尚未运行：CUDA设备分配、224/batch32/FP32前后向、显存峰值、CPU-CUDA数值对照、GPU吞吐/总时长测量、完整21次GPU训练，以及真实远端环境安装/驱动兼容/持久进程检验。无法凭CPU速度或GPU理论规格给可靠GPU总时长。

参考本地软件版本torch2.11.0+cu130、torchvision0.26.0+cpu、numpy1.26.4。torchvision CPU构建在当前CPU环境可用，但目标GPU机器上的组合兼容性尚未验证；应在实际获授权实例上先验证包导入和真实CUDA preflight，再决定运行。

复现CPU检查：`python experiments/e1a933_review/vision_cuda_cpu_check.py`（输出目录需新建，已有目录会拒绝覆盖）。本机preflight已保存`vision_cuda_preflight_local/receipt.json`。迁移后的命令在bundle/README.md；默认preflight和执行矩阵分别使用新输出目录。

传输精确大小：payload 49607074 bytes；tar.gz 45235217 bytes（约43.14 MiB）。归档SHA256：`339f9382fca70d97c0b99db4e0c01143e1c7dba905c62cd3b0a6d17891f8e0b6`。

交付时冻结CPU训练源文件哈希复核：True。该检查不触碰CPU进程或产物。
