# N02 GPU 补完：冻结合同与执行入口

状态：**48 臂已全部执行、回收并逐文件 SHA256 校验**（`MATRIX_COMPLETE`，48/48）；独立解读、专报重写与论文接线已完成，见 `N02_SAMPLING_ACCESS_REPORT.md`（本文件是 N02 的唯一现行版本，旧 12-checkpoint 零重训诊断降级为其第 9 节历史子节）与 `N02_INTERPRETATION.md`。逐臂指标、逐 parent 配对差、bootstrap CI 与预测-结果对照在 `artifacts/e1a933_review/N02_interpretation/`；独立重算与冻结结果的最大绝对差为 0.0（48 臂）、250/250 manifest 条目吻合。机制判定为 **UNRESOLVED**。本文件保留为冻结的臂设计与执行合同，不再作为进度来源。

## 冻结的48臂

- A=native64、B=bilinear(A,224)，各自 standard/lowstride × pretrained/random × 803/805/806，共24臂。
- C=直接native224、D=area(C,64)，standard × pretrained/random × 三seed，共12臂。
- A/B的standard pretrained/random补static-only（三seed），共12臂；其clean train索引重复到与flip臂相同每epoch曝光数。
- 所有臂20 epochs，Adam 3e-4，batch32，FP32，TF32/AMP关闭。只按singleton-dev BCE选择epoch；无AB train/dev，无目标J选型，不删除塌缩seed。
- lowstride仅将ResNet18 maxpool改为Identity，卷积权重可直接复用；参数量相同，但MAC、feature map、峰值显存和实际step成本单独测量。它也改变局部池化/感受野，不能把所有差异都叫aliasing。

## 原生图像合同

全部新臂共用新连续renderer：[-1,1]²视场，红/蓝连续线段，圆端帽，物理半径5/64（直径0.15625），每像素4×4 supersampling，红先蓝后遮挡，灰背景nuisance，颜色组内端点排序。A/C分别直接按64/224网格采样；没有用插值伪装C。B/D变换明确记录，不改变原始geometry/标签。

同一水平线的实测物理宽度：A=0.1562499702，C=0.1562499659；native C与bilinear(A)最大像素差0.19630185，颜色与视场检查通过、端点反转图像精确不变。

这是一个新的共同观测合同，区别于旧D04整数方形笔刷；旧checkpoint数字继续单列，不能与新renderer结果直接拼成一个因果差值。所有新几何state的完整stroke都在视场内；不裁掉前景。

## fresh parent与分母

- train：seed962001，512 parent，2048 static +758 single flips=2806例。
- dev：seed962002，128 parent，512 static +198 single flips=710例。
- test：seed962003，512 parent，2048 static +757 single flips=2805例。
- test E quartets：seed962004，178 quartet／85 parent；每parent最多4例，按固定miner次序；1个候选因stroke越界在固定几何资格阶段排除。训练/test分割先于编辑。
- 19个已有/暴露bank用颜色组内canonical坐标查重，容差1e-6匹配数均0；train/dev/test parent互斥，全部train state与test singleton/quartet state也无匹配。原bank路径/hash和逐split parent hash在data_manifest.json。
- 本地全部178 quartet的native A/C检查：不同label却整图完全相同=0；每个异label状态对至少有一个对比度变化>0.2的像素。此检查不等于完整人类可观测性或遮挡证明。远端在缓存后再次报告精确冲突，不从总分母删样本。

## 预先固定结构预测

预测移除maxpool会减少B−A的J优势；预计在两个组成编辑中较小者最大端点位移≤2 native64像素时更明显。阈值由像素尺度设定，没有按结果选择。该stratum只有5 quartet，必须照报其小分母，不能靠它宣称通用规律。完整85-parent验证及全样本交互照常进行；分层证据不足时保持机制未决。

输出包含逐样本logits、singleton/static/flip表现、A/B/AB/J/CCM分母、110→111、migration、111退化，以及parent聚类bootstrap；三个seed分开报告，不能作三个独立任务。

## 复现与远端运行

本地生成（当前环境的SciPy需要conda libstdc++）：

```bash
LD_LIBRARY_PATH=/opt/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python experiments/e1a933_review/sampling_gpu_prepare.py \
  --out artifacts/e1a933_review/N02_gpu_bundle
```

bundle仅包含三个sampling_gpu代码文件、冻结geometry、data/config/文件hash manifest、已有ResNet18权重，运行不下载。当前目录已存在，复现需另取新目录，入口拒绝覆盖。

GPU单次preflight：

```bash
python sampling_gpu_run.py --bundle /path/to/N02_gpu_bundle \
  --out /path/to/N02_gpu_preflight --preflight
```

执行完整48臂（会重新执行同样有界preflight，probe模型不会进入训练）：

```bash
python sampling_gpu_run.py --bundle /path/to/N02_gpu_bundle \
  --out /path/to/N02_gpu_full --execute
```

preflight测standard/lowstride × 64/224四种shape，每种3个一次性训练step（首step预热，后两step估时），包括Adam状态显存；记录feature maps/MACs与CPU/CUDA renderer差异。训练成本估计不含dev/test/cache/IO，不能当实测完成时间。

完成标志：`receipt.json: status=MATRIX_COMPLETE`，`completed`长度48；失败保存traceback。每个输出目录`exist_ok=False`。本地入口已验证manifest和NO_CUDA路径，GPU forward/backward仍待远端preflight。

native A/C缓存约4.6GB，默认放输出同级`<out>_image_cache`；不能把缓存当需要回收的科学结果。输出保留全部geometry、每图hash、缓存manifest和精确renderer。48个checkpoint约2.15GB，另有预测和统计；统一supervisor回收output后再关闭实例。磁盘缓存可由固定geometry/renderer重建。

不得因时间/关机省略四格、C/D、static-only、三seed或fresh-parent结果；若远端失败，应保留失败回执并修复，不能把preflight写成训练完成。
