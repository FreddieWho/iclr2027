# 项目问答记录（QA）

本文件记录本项目关于模型架构、技术路线和科学问题的问答。后续凡涉及这三类内容，均在本文件追加一条记录，保留日期、问题、回答、证据路径和当前不确定性。

## QA-2026-08-27-001：P0 干预、能量、频段与 embedding

### 问题

P0 是怎么实现干预的？能量误差和频段如何定义？使用什么模型生成 embedding？autoencoder 和 ResNet-18 有什么关系？

### 回答

#### 1. 输入样本和图

P0 使用 SkillCorner 比赛 `1886347`。从 tracking 文件中按覆盖率选择两队各 10 名外场球员，固定节点顺序；对每个节点的坐标按球场长宽归一化到约 `[-1, 1]`。从所有节点齐全的帧中等间隔抽取 1,000 帧。

每个样本单独构建一个 4-nearest-neighbor 图：每个球员连接空间上最近的 4 个球员，连接取并集并做对称化，边权是基于距离和样本内中位距离的 Gaussian 权重。随后计算归一化图 Laplacian：

\[
L=I-D^{-1/2}AD^{-1/2}=U\Lambda U^\top.
\]

#### 2. 干预如何生成

每个样本生成 10 条干预：

- `common`：所有节点朝同一方向移动；
- `single`：一个节点移动；
- `semantic_coalition`：固定的 5 节点结构支持样例施加随机二维位移；
- `random_coalition`：随机选择同样大小的 5 节点并施加随机二维位移；
- `band_0`–`band_5`：在指定 Laplacian 模态区间内随机组合模态，再乘一个二维方向。

干预首先在节点坐标上生成，之后统一归一化到目标能量。P0 生成的是干预记录，还没有计算模型对干预前后的响应；所以 P0 不能支持“模型更在意共同移动”等现象结论。

#### 3. 能量和能量误差

一个干预是 `N×2` 的位移矩阵 \(\delta\)，能量定义为 Frobenius 范数：

\[
E(\delta)=\|\delta\|_F
 =\sqrt{\sum_{i=1}^{N}\sum_{d=1}^{2}\delta_{i,d}^2}.
\]

P0 的目标能量是 `1.0`。归一化操作为：

\[
\delta' = \delta\frac{1.0}{\|\delta\|_F}.
\]

记录中的 `energy_error` 定义为：

```text
abs(norm(delta') - 1.0)
```

最大误差为 `4.44e-16`，属于浮点数误差。它表示坐标位移总量是否配平，不是像素差异、LPIPS 或模型响应误差；P0 尚未做像素感知能量配平。

#### 4. 频段如何定义

每个样本有 20 个 Laplacian 特征模态。P0 用模态序号等数量切成 6 段：

```text
20 个模态 -> 6 个 equal-count mode bins
```

每个 `band_b` 只使用该段中的特征向量组合，因而目标频段纯度接近 1。需要特别区分：P0 实际使用的是“等数量模态分桶”，不是“等特征值质量分桶”，也不是等能量分桶。后续 P1/P2 可以探索按归一化特征值区间或累计谱质量重新定义频段，并记录版本。

`mode_purity` 的定义是目标模态上的投影平方能量占全部投影平方能量的比例：

\[
\text{purity}(B)=
\frac{\sum_{k\in B}c_k^2}{\sum_k c_k^2},
\qquad c=U^\top\delta_x,
\]

其中 P0 实现用二维干预的第一个坐标分量计算投影；由于频段干预使用同一个标量模态场乘二维方向，这足以做 smoke 检查。

#### 5. 使用什么模型生成 embedding

P0 使用两个互相独立的 embedding 分支：

**坐标分支：tiny coordinate autoencoder**

- 输入：每个样本展平后的 `20×2=40` 个坐标值；
- 编码器：`40 -> 64 -> 32`，中间使用 `Tanh`；
- 解码器：`32 -> 64 -> 40`，中间使用 `Tanh`；
- 优化：Adam，学习率 `1e-3`，40 个 epoch；
- 输出：32 维坐标 embedding；
- 训练目标：重构原始坐标，P0 重构 MSE 约 `0.0263`。

它是一个小型坐标 baseline，用来确认结构坐标能够被模型编码，不是 AMR，也不代表科学性能提升。

**视觉分支：冻结 torchvision ResNet-18**

- 输入：由原始坐标渲染出的 1,000 张足球 minimap；
- 权重：ImageNet-1K `ResNet18_Weights.IMAGENET1K_V1`；
- 做法：去掉最后的分类层，保留 512 维特征；
- 状态：`eval()` 且参数冻结；
- 输出：每张 minimap 一个 512 维 embedding；
- P0 在 CPU 上完成，CUDA/MPS 不可用。

#### 6. Autoencoder 和 ResNet-18 的关系

两者不是串联关系，也不是一个训练另一个：

```text
同一个 canonical sample
       ├── 坐标 ──> tiny autoencoder ──> 32D coordinate embedding
       └── minimap ──> frozen ResNet-18 ──> 512D visual embedding
```

Autoencoder 直接看数值坐标，学习一个小型坐标重构表示；ResNet-18 看坐标渲染出的图像，作为固定的视觉观察器。P0 没有把两个向量拼接、对齐或联合训练，也没有用 ResNet embedding 反过来训练 autoencoder。

此外，P0 的 ResNet embedding 是对原始 minimap 的 embedding，不是对每一条干预后图像的响应差异。干预响应测量属于后续阶段。

### 证据路径

- 实现：[scripts/run_phase0_pipeline.py](scripts/run_phase0_pipeline.py)
- 阶段报告：[reports/PHASE0_REPORT.md](reports/PHASE0_REPORT.md)
- 汇总结果：[artifacts/phase0/pipeline_summary.json](artifacts/phase0/pipeline_summary.json)
- 干预表：[artifacts/phase0/intervention_manifest.parquet](artifacts/phase0/intervention_manifest.parquet)
- 频段检查：[artifacts/phase0/spectrum_sanity.csv](artifacts/phase0/spectrum_sanity.csv)

### 当前限制

- P0 只使用一场比赛和 smoke-scale 节点选择；
- `semantic_coalition` 还不是 P2 所需的严格同频语义对照；
- P0 尚未测量干预前后模型表示变化；
- 32D autoencoder 和 512D ResNet 特征只能证明接口可运行，不能证明 AMR 或任何科学假设成立。

## 维护规则

以后凡是关于以下主题的问答，都追加到本文件，而不是只在对话中回答：

1. 模型架构与输入输出；
2. 技术路线、实验实现和指标定义；
3. 科学问题、假设、解释、替代解释和研究边界。

每条记录至少包含：问题、通俗回答、实现或文档证据路径、当前限制或不确定性。

## QA-2026-08-28-002：点集数据为什么不需要视觉网络作为主线

### 问题

数据集已经把运动员表示成点，为什么 P1 还要使用 DINOv2、OpenCLIP 等视觉网络？

### 通俗回答

不需要把视觉网络作为主线。点集已经保留了球员位置、队伍、角色和图结构所需的信息，因此 DeepSets、GAT 和 Phase-GAT 可以直接在点坐标上工作。这条路线最贴合数据本身，也最容易解释干预和频段响应。

视觉网络只能接收由同一批点确定性渲染出的 minimap。它没有增加原始信息，只是在测试一个附加问题：把点集画成图像后，候选响应是否仍然存在。这个分支还会引入分辨率、颜色、点大小、抗锯齿和渲染方式等额外因素，因此不能用于和点集模型做无条件的架构优劣比较。

因此，P1/C1 的主线改为直接点集模型；DINOv2/OpenCLIP 结果保留为可追溯的辅助渲染稳健性分支，默认后置，不再是启动大规模计算的必要条件。

### 实现或文档证据

- 主线契约：[artifacts/phase1/point_mainline/execution_contract.json](artifacts/phase1/point_mainline/execution_contract.json)
- 主线报告：[reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)
- 流水线开关：[scripts/run_phase1_pipeline.py](scripts/run_phase1_pipeline.py)，参数 --point-only
- 点集模型响应：[artifacts/phase1/point_mainline/spectrum_response.parquet](artifacts/phase1/point_mainline/spectrum_response.parquet)

### 当前限制

- 视觉分支已经在本次停止请求前完成，原始结果保留在 artifacts/phase1/，但不纳入点集主线结论；
- 点集主线仍是 P1 的候选响应筛查，不足以证明 H1/H2 或 P2 的严格同频组织结论；
- 视觉分支若以后重新启用，必须单独说明它与主线的关系、额外计算成本和渲染混杂因素。

## QA-2026-08-28-003：P1 发现能否在中国书法上确认

### 问题

P1 当前发现是什么？能否在中国书法数据上确认同样的发现？

### 通俗回答

P1 的主线发现不是“模型已经理解了足球结构”，而是一个表示响应偏差：

1. 在总位移能量相同的情况下，让所有球员一起移动，通常比只移动某些角色球员引起更大的 embedding 变化。9 个点集模型上的共同移动胜出比例约为 91.5% 到 100%。
2. 低阶、整体性的图模式响应明显更强，细粒度的高阶模式响应通常较弱。
3. 目前没有确认一个稳定的中频 notch；现阶段只能把它当作候选形状。P1 还不能排除普通低通偏置、图定义或模型结构的影响。
4. 表示变化不能在 held-out 数据上可靠恢复具体模态编号，说明模型不是一个精确的频率识别器。

因此，P1 支持“模型可能更重视整体共同变化，而没有同等清楚地保留内部结构变化”这一探索性线索，但还不能把它写成已经证明的组织盲区。

书法上可以检验同一类线索，但现在不能直接宣称已经确认。最合适的对应关系是：

- 足球全队一起移动，对应书法整字或整页一起平移；
- 足球一个战术子群移动，对应一个偏旁或部件整体移动；
- 足球局部结构破坏，对应关键单笔或局部笔画关系变化。

第一轮应使用 Make Me a Hanzi 这类有 SVG 笔画、stroke medians、笔顺、部件分解和 stroke-component 对应关系的数据。它能在矢量层施加已知干预并重新渲染，避免把图像拼接伪影误当成结构效应。自然书法图像可作为第二轮外部检验，但需要额外解决笔画/部件提取、书法家和字符隔离以及许可边界。

书法端要确认同样发现，至少要满足：

- 整体、部件和单笔干预的坐标能量或像素感知能量严格匹配；
- 部件干预与随机笔画子集在支持大小、图频率和位移分布上匹配；
- 在多个字符、书体或书法家以及至少两类表示模型上方向一致；
- 同时检查客观结构任务，例如同字符跨页面检索、部件/笔画结构匹配和干预定位；
- 将全局姿态和内部构型分开评价，因为对字符识别而言整体平移通常应近似不改变字符身份，但对风格或页面布局任务可能是有意义的上下文。

如果书法端也出现“整体平移造成的表示变化大于同能量的部件/笔画结构变化”，并且在严格同频随机对照后仍然成立，才能称为跨域复现。若只有整体平移敏感而结构干预不敏感，也可能只是图像编码器的画布、裁剪或边界效应。

### 实现或文档证据

- P1 点集主线报告：[reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)
- P1 汇总：[artifacts/phase1/point_mainline/pipeline_summary.json](artifacts/phase1/point_mainline/pipeline_summary.json)
- 书法数据规划：[docs/04_DATA_AND_ENVIRONMENT_GUIDE.md](docs/04_DATA_AND_ENVIRONMENT_GUIDE.md)
- 跨域实验规划：[docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md](docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md)
- 当前本地受控书法数据：[data/raw/calligraphy/makemeahanzi](data/raw/calligraphy/makemeahanzi)

### 当前限制

- P1 只在足球点集上完成，书法端尚未执行同类 probe；
- 当前 common 与 semantic coalition 还不是 P2 要求的严格同频语义对照；
- 书法的整体平移可能是任务无关姿态，也可能是任务相关上下文，必须先固定任务定义；
- 视觉模型若用于书法，只能作为像素表征分支，不能与点/笔画图模型做无条件架构排名。

## QA-2026-08-28-004：P1 之后应先推进还是先做书法复现

### 问题

P1 之后，是直接推进足球端后续实验，还是先在汉字上复现现象？

### 通俗回答

建议先做一个小规模、低成本、严格受控的汉字复现，再决定是否进入大规模 P2 或方法训练。不是立即做完整书法数据，也不是把一个未完成的足球候选结果直接推广到书法。

原因是 P1 已经看到较强的共同移动信号，但当前 common 与 semantic coalition 还没有完成严格同频匹配，低阶整体模式也明显更强。因此目前仍有一个重要替代解释：结果可能只是普通的低通/图谱偏置，而不一定是模型真的忽略了组织结构。

汉字小试验可以快速提供跨域方向性证据。优先使用 Make Me a Hanzi 的 SVG 笔画和部件结构，选少量字符和两个代表模型，比较：

- 整字整体平移；
- 一个偏旁或部件整体移动；
- 一个局部笔画移动；
- 支持大小、图频率、位移能量都匹配的随机笔画对照。

如果在多个字符、多个模型和严格对照下仍出现“整体移动比结构变化更能改变表示”，就值得继续做足球 P2 严格控制和书法外部验证。如果只有原始图像模型出现，优先怀疑画布、裁剪或渲染效应；如果书法端不复现，则把规律收紧为足球/当前图结构特有的候选偏置。

因此推荐顺序是：汉字小规模受控复现作为 go/no-go 检查，随后回到足球 P2 排除普通频率偏置，最后才做全量书法和 AMR 跨域验证。这个小试验不需要大型视觉网络或高成本训练。

### 实现或文档证据

- P1 主线结果：[reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)
- 书法受控数据：[data/raw/calligraphy/makemeahanzi](data/raw/calligraphy/makemeahanzi)
- 书法数据与许可说明：[docs/04_DATA_AND_ENVIRONMENT_GUIDE.md](docs/04_DATA_AND_ENVIRONMENT_GUIDE.md)
- 足球 P2 同频对照设计：[docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md](docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md)

### 当前限制

- 汉字端目前尚未运行 probe，以上是下一步决策而非实验结果；
- 小规模复现只能判断方向性和明显混杂，不能替代完整跨域验证；
- 足球端的严格同频控制仍然是解释 P1 候选信号所必需的实验。

## QA-2026-08-28-005：笔画应作为节点还是端点应作为节点

### 问题

足球数据中的球员是没有朝向的点，而书法笔画是有方向、具有长度和曲率的对象。如果把整条笔画压成一个点，会严重丢失局部变化。是否应该改用笔画端点作为节点？

### 通俗回答

你的纠正是对的：整条笔画不能直接等同于足球中的一个球员点。笔画应当保留为有序的延展对象，端点必须成为节点，但只用两个端点仍然不够，因为中间的弯曲和局部形变会消失。

本项目采用折中但可解释的表示：每条笔画的 `median` 按等弧长采样为 5 个有序点，即起点、终点和 3 个内部点。这样：

- 起点和终点保留书写方向的边界；
- 内部点保留横、竖、撇、捺及弯曲过程中的局部变化；
- 切向、弧位置、首尾标记、局部曲率和笔画顺序作为节点特征；
- 一条笔画是一个五点子结构，不再被压成一个无方向节点。

图仍使用对称边和对称归一化 Laplacian，以便与足球 P1 的实谱进行可比的作用模式分析；方向不通过有向 Laplacian 强行编码，而是通过有序点、切向和首尾特征编码。所有干预复用基线图和谱基，不在干预后重新构图。

### 实现或文档证据

- 实现：[scripts/run_calligraphy_probe.py](scripts/run_calligraphy_probe.py)
- 受控数据说明：[data/raw/calligraphy/makemeahanzi/README.md](data/raw/calligraphy/makemeahanzi/README.md)
- 受控 smoke 契约：[artifacts/calligraphy_pilot/smoke/execution_contract.json](artifacts/calligraphy_pilot/smoke/execution_contract.json)
- 干预校验：[artifacts/calligraphy_pilot/smoke/intervention_validation.json](artifacts/calligraphy_pilot/smoke/intervention_validation.json)

### 当前限制

- 五点采样仍是离散化，不等同于完整笔画轮廓；短笔和长笔目前按相同节点数表示。
- 当前环境没有 `torchvision`，因此 ResNet-18 只能记录为阻塞的辅助分支，不能用随机权重替代。
- Make Me a Hanzi 的 `matches` 提供的是结构分解路径标注；当前 `semantic_component` 名称应理解为结构标注部件，不是字源学或自然语言意义上的语义部件。
- 当前 smoke 只用于接口和方向性检查，不能确认跨域规律；正式 800 字符计算尚未启动。

## QA-2026-08-29-006：如何在不改变科学定义的前提下优化并适配 Tesla V100

### 问题

正式 800 字符计算可能较慢。哪些部分适合本地优化，哪些部分值得针对 Tesla V100-32GB 做 CUDA/FP16 优化？如何避免为了提速改变模型条件或污染生信项目？

### 通俗回答

把“数据和科学定义”与“模型张量计算”分开：字符划分、五点采样、固定图和谱、干预生成、能量/边界校验仍在 CPU 的高精度路径中完成；只有点集模型的训练和 embedding forward 使用设备加速。这样加速的是重复计算，不是改写实验问题。

当前实现按节点数把字符和干预组成 dense batch，推理时使用 `inference_mode`，并跳过不参与响应的 decoder。DeepSets 不创建邻接矩阵，GAT 仍使用每个字符的固定基线邻接。默认训练 batch 固定为 16，CUDA 默认只增大推理 batch；显式请求 CUDA 但环境不可用时会直接失败，不会伪装成 GPU 结果。

针对 V100，提供 `--device cuda --gpu-profile v100 --amp on`。FP16 只用于模型张量；GAT 的 attention logits、mask、`log(adjacency)`、softmax 和 loss 归约保持 FP32，以防下溢或 NaN。正式计算前必须先用小 smoke 做 CPU FP32、CUDA FP32 和 CUDA AMP 的数值对照。

### 结果

本地优化版 smoke 从旧参考的 148.76 秒降到 88.12 秒，约 1.69 倍加速；干预数量、有效率和能量误差不变；对齐后的两类模型响应最大绝对差约 `1e-7`。这只是执行等价和性能证据，不是科学发现。

### 证据

- 优化 checkpoint：[reports/CALLIGRAPHY_OPTIMIZATION_CHECKPOINT.md](reports/CALLIGRAPHY_OPTIMIZATION_CHECKPOINT.md)
- 优化版产物：[artifacts/calligraphy_pilot/optimized_smoke](artifacts/calligraphy_pilot/optimized_smoke)
- 实现：[scripts/run_calligraphy_probe.py](scripts/run_calligraphy_probe.py)

### 限制

- 本地没有可用 CUDA，V100 吞吐、显存和 AMP 等价尚未实测；拿到算力地址后必须先跑 V100 smoke。
- 当前没有 `torchvision` 本地环境和预训练权重，ResNet-18 仍不是科学证据。
- 本次优化没有启动正式 800 字符任务，也没有触碰 `infra/bioinf-data-index/`。

## QA-2026-08-29-007：正式计算需要租用多久、什么 CPU 和内存

### 问题

正式 800 字符任务还要考虑上传和环境准备时间。在 `8c16g`、`8c24g`、`16c32g`、`16c48g` 中如何选择？

### 通俗回答

如果机器带 Tesla V100-32GB，推荐 `8c24g + V100`。`8c16g` 理论上也能跑，但正式任务的 pandas/parquet 中间对象和系统余量较小；16 核对当前主线帮助有限，因为代码中的 CPU 线程数已限制在最多 4 个。`16c32g` 和 `16c48g` 对带 V100 的本项目属于过配，除非平台价差很小。

当前本地优化版 CPU smoke 是 88.12 秒；按正式参数的简单线性外推约 30.6 小时，但这不是 V100 实测。V100 的实际收益受小型 dense batch 和 Python 调度影响，不能直接假设固定倍数。因此建议：

- 计算预算按 6–12 小时预留；
- 加上开机、上传、依赖检查、V100 smoke 和结果校验，按小时租用至少 12 小时；
- 如果平台不保证续租，或当前没有中断恢复机制，直接租 24 小时更稳妥。

需要上传的主线 payload 约 32MB：`dictionary.txt`、`graphics.txt`、probe 脚本和依赖说明。不要上传 `svgs/`、外部 CCSE 目录、整个项目或任何生信目录。网络很慢时，上传和校验按 10–30 分钟预留即可；如果误传完整 330MB 数据目录，时间会明显增加。

### 选择结论

| 场景 | 推荐配置 | 建议租期 |
|---|---|---:|
| V100 正式主线 | `8c24g + V100-32GB` | 12 小时起，24 小时稳妥 |
| V100 成本优先 | `8c16g + V100-32GB` | 可用，但内存余量较小 |
| 没有 GPU、必须 CPU | `16c32g` | 约 36–48 小时 |
| `16c48g` | 当前主线不需要 | 不推荐为本任务加价 |

### 限制

- 上述 6–12 小时是 V100 未实测前的预算区间，不是承诺；拿到地址后先跑小 smoke 再决定是否延长。
- 当前正式脚本没有完整的中断恢复机制，租期不足可能需要从头重跑。
- ResNet-18 是未启用的辅助分支，不应为了它额外扩大 CPU、内存或上传内容。

## QA-2026-08-29-008：AI Galaxy 当前性价比最高的租用方案

### 问题

结合本项目的实际负载，AI Galaxy 当前哪些 GPU、CPU 和内存组合最划算？

### 实时报价与结论

AI Galaxy 报价目录刷新时间为 `2026-08-29 20:29:41`。平台按近似 dense FP16 TFLOPS/元小时排序时，1 张 RTX 3080 排第一；它有 10GB 显存，但本项目只有小型点集 AE/GAT，不运行自然图像 ResNet 主线，预计显存够用，仍需先跑 smoke 实测。

| 方案（8c24g） | 平台小时价 | 12 小时 | 24 小时 | 平台近似性价比 |
|---|---:|---:|---:|---:|
| RTX 3080 Docker | 约 ¥0.7225 | ¥8.67 | ¥17.34 | 约 164.7 FP16 TFLOPS/元小时（按 24GB 主机配置） |
| RTX 3080 KVM | 约 ¥0.8325 | ¥9.99 | ¥19.98 | 约 142.94 |
| RTX 3090 | 约 ¥1.1625 | ¥13.95 | ¥27.90 | 约 122.15 |
| Tesla V100-32GB | 约 ¥1.3825 | ¥16.59 | ¥33.18 | 约 90.42 |

因此，性价比优先时推荐 `1× RTX 3080 + 8c24g`，优先 Docker；若需要更传统的 SSH/KVM 环境，则选同规格 KVM。若必须保持 V100-32GB 作为目标设备，推荐 `1× V100-32GB + 8c24g`，但它是兼容/复现实验选择，不是价格性能最优选择。

AI Galaxy 当前只读评估中的 CPU-only 选项是 `8c16g`、约 ¥0.325/小时；没有给出满足 24GB 内存的 CPU-only 报价。结合本项目 CPU 线约 30.6 小时的简单外推，CPU-only 不值得替代 GPU。

### 租期建议

计算本身按 6–12 小时预留，另加 20–40 分钟用于实例启动、上传约 32MB 主线数据、环境检查、GPU smoke 和结果校验。建议先租 12 小时；如果平台不能无缝续租，或任务中断后需要从头开始，租 24 小时更稳。

### 限制

- 上述 AI Galaxy 性价比是平台公开规格的近似排序，不是本项目实测吞吐；RTX 3080、RTX 3090 和 V100 的真实速度要以同一 smoke 对照为准。
- 本次只调用了 AI Galaxy 的目录/报价读取接口，没有执行租用、扣费或释放操作；针对具体约束的预租用接口当前返回没有满足条件的 enabled instance specification，因此下单前还要在平台界面确认库存。
- 本项目不需要上传 SVG、ResNet 权重、整个项目或任何生信目录。

## QA-2026-08-29-009：AI Galaxy 平台是否支持原生自动续租

### 问题

这里的“自动续租”是指 AI Galaxy 服务器/平台自身的自动续费设置，不是由助手或外部程序定时重新下单。

### 通俗回答

是。AI Galaxy 官方 v2 租用流程在创建实例时明确包含“是否自动续租”，实例管理中也有“管理自动续费”入口。因此，平台原生自动续租应在 AI Galaxy 控制台创建实例时开启，不需要我编写 watchdog 或定时调用续租接口。

此外，AI Galaxy OpenAPI v2 提供了独立的 `POST /openapi/v2/instance/set_autorenew` 接口，要求传入 `instance_name`、`autorenew_on`、`autorenew_unit` 和 `pay_type_first`。也就是说，平台端开关可以由受控 API 工具开启；不是平台能力缺失。

但“自动续租”和“任务完成后自动关闭订单”是两件事：前者有官方文档和 OpenAPI 接口依据；后者需要 AI Galaxy 提供“任务完成/作业退出后自动释放实例”的原生工作流。当前已核对的 AI Galaxy Compute MCP 只提供 `plan_rental`、`rent`、`plan_release`、`release` 等显式操作，没有暴露 `set_autorenew` 或任务完成回调，所以不能把后者当成已经具备的能力。

### 当前结论

- 平台原生自动续租：AI Galaxy 控制台支持，创建实例时选择“自动续租”；
- 通过 API 解开开关：技术上可行，但需要给 MCP 增加一个带实例归属和显式确认的 `set_autorenew` 工具，或直接在控制台操作；
- 助手外部定时续租：本项目不实现，也不是用户要求；
- 任务完成后平台原生自动关闭：当前证据不足，不能承诺；
- 本次没有执行下单、续租、释放或任何计费操作。

### 风险边界

开启原生自动续租后，如果任务异常退出而没有平台原生的完成触发器，实例可能继续计费。因此正式下单前仍需在 AI Galaxy 控制台确认是否有“任务完成自动释放”、最大租期/预算上限以及磁盘保留设置；不能把固定到期释放误称为任务完成释放。

参考： [AI Galaxy v2 租用流程](https://gpu.ai-galaxy.cn/docs_v2/)；[设置实例自动续费 API](https://s.apifox.cn/b0fc397f-c455-4c9a-9d82-875fc48ae106/api-242142486)；[当前 MCP 的工具与自动续费策略说明](https://glama.ai/mcp/servers/FreddieWho/mcp_ai_galaxy)。

## QA-2026-08-31-010：当前按计划还需要做什么？

### 通俗回答

项目共 9 个阶段（P0–P8）。P0 已完成；P1 的足球点集主线已完成。P1 之后的正式科学工作仍未完成：当前书法 800 字符任务是辅助性的探索 pilot，不能替代 P2 的严格同频控制，也不能直接算作 P5 的跨域确认。

剩余顺序为：

1. P2：用语义联盟、连通随机子图、任意随机子集和频率匹配的相位随机扰动做同频对照，判断组织效应是否超出普通谱偏置。
2. P3：做增强耦合、unary/relational pooling、invariance/equivariance 的受控因果矩阵，并做 layer-wise response 与 mode accessibility，定位候选凹陷或错序的机制。
3. P4：依次实现 CAP、AMR-Fixed、AMR-Learned，比较全局鲁棒性、内部结构可访问性、自然任务、参数量和运行成本。
4. P5：根据体育端实际得到的弱频段和支持域比例，建立多个体育到书法的谱 rank 映射；在受控汉字和自然书法结构任务上检验预测、失败模式及 AMR 修复。
5. P6：统一 match/character-level bootstrap、效应量、图构造敏感性、seed 稳定性、数据许可、泄漏检查、运行成本和 claim ledger，冻结证据包。
6. P7：从实测结果中选择 5–6 张主图，完成摘要和主文结果叙事；阴性或未确认结果也按探索性结果保留。
7. P8：完成 9 页主文、补充材料、代码/产物清单、复现 smoke、最终 checksum 和提交包。

### 当前状态判断

- P1 足球主线：`completed`，250 个样本、10 场比赛、3 类点集模型、3 seeds；候选 ERRR 和中频凹陷已观察到，但尚未通过 P2 控制。
- 书法 pilot：800 个字符、6 个点集模型、180495 条有效干预；可作为跨域线索，不能作为最终复现结论。当前使用每笔 5 个有序中线点，保留端点和内部局部变化，不是把整笔压成一个点。
- P2、P3 尚无正式检查点报告；AMR 尚无方法结果；因此当前最有信息增益的下一步是先做 P2，而不是立即训练 AMR 或再次租 GPU。
- 本项目是探索性研究，允许根据 P2/P3 结果继续、扩展或 pivot，不设置预注册或类似的事前限制。

### 证据路径

- 阶段计划：[configs/project.yaml](configs/project.yaml)
- 实验与检查点规划：[docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md](docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md)
- P1 主线报告：[reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)
- 书法 pilot 报告：[artifacts/remote/retrieval-20260831/CALLIGRAPHY_PILOT_REPORT.md](artifacts/remote/retrieval-20260831/CALLIGRAPHY_PILOT_REPORT.md)

### 当前限制

- P1 的 common-versus-semantic 比较不是严格同频匹配，不能单独证明“组织效应超过频率”。
- 书法 pilot 是受控矢量数据，不等同于自然书法图像，也没有完成 P5 的客观结构任务和跨域预测闭环。
- P2/P3 的实际 band、图构造和匹配方法仍可随探索结果迭代；必须保留每次版本和选择理由。

## QA-2026-08-31-011：P2 应该如何实现？

### 问题

如何实现 P2，才能判断足球 P1 的候选现象究竟是组织效应，还是普通的图频率偏置？

### 通俗回答

P2 不重新训练模型。P1 的角色联盟干预是角色组内各节点独立随机位移，更接近结构破裂；P2 主分析会为所有可用角色组重新生成四个固定方向的 rigid subgroup translation，并把 P1 fracture operator 作为次要连续性分析。P1 当前随机对照只匹配节点数；P2 要进一步匹配同一帧、同一球队、支持大小、总能量、operator/方向、Rayleigh quotient、六频段能量和拓扑量。

正式控制分为三类：同队同大小任意随机支持、拓扑与频率共同匹配的随机支持，以及保持每个 eigenmode 功率但随机翻转非 DC 谱系数符号的 exact-spectrum control。三类分别排除基础 size/team confounding、拓扑与粗频率解释、完整频谱功率解释。

P2 复用 P1 的 250 个样本、9 个点集模型 checkpoint 和 baseline embedding，不重训模型。rigid semantic anchors 与新 controls 需要重新推理；只有 P1 fracture 连续性 lane 可以复用旧 semantic responses。主指标是 matched Semantic Coalition Gap：

\[
\Delta_{SCG}=d_z(semantic)-d_z(control).
\]

统计先形成每个 matched set 的 paired difference，再汇总到比赛级；bootstrap 单位是 match，不把帧、干预行或 seed 当成独立比赛。P2 的 probe graph 比较 P1 kNN-4 与 weighted Delaunay；GAT encoder 始终使用 P1 训练时的固定 kNN-4 adjacency，避免混入模型输入图变化。

P2 的科学结果可以是组织效应存活、退化为普通频率偏置、仅在特定图/模型存在，或匹配质量不足。它是探索性检查点，不设置单一机械显著性阈值，也不包含预注册约束。

### 实施顺序

1. synthetic graph 单元测试和匹配器；
2. matching-only 本地 smoke，不加载模型；
3. 单个 DeepSets checkpoint 响应 smoke；
4. 250 samples、10 matches、4 energies、2 probe graph definitions、9 models 的正式点集运行；
5. 输出 C2 报告并决定进入 P3 或 pivot。

### 证据与完整方案

- P2 实施方案：[docs/07_PHASE2_IMPLEMENTATION_PLAN.md](docs/07_PHASE2_IMPLEMENTATION_PLAN.md)
- P1 点集契约：[artifacts/phase1/point_mainline/execution_contract.json](artifacts/phase1/point_mainline/execution_contract.json)
- P1 干预表：[artifacts/phase1/intervention_manifest.parquet](artifacts/phase1/intervention_manifest.parquet)
- P1 主线报告：[reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md](reports/EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md)

### 当前限制

- P1 只有 10 场比赛，P2 可作为机制分流检查点，但最终论文证据仍需要在 P6 评估独立比赛数量是否足够。
- 高能量 `epsilon=2.0` 的边界失效率较高，必须单独报告；不能把失败行静默删掉。
- Phase-GAT 的 P1 任务性能较弱，只能作为辅助模型族，不能独自支撑结论。
- P2-A/B matching-only 已执行；模型响应、SCG 统计和 C2 科学解释仍未执行，不应写成 P2 已完成。

## QA-2026-08-31-012：P2-A/B 首轮实现做了什么？

### 问题

正式模型推理之前，如何验证 P2 的干预和匹配本身没有把能量、球队、支持大小或频谱定义混在一起？

### 回答

先在本地 CPU 上做 matching-only。主干干预是角色组内节点的刚性同向平移，支持大小、方向和总 L2 能量明确记录；控制包括同队同大小任意支持、通过拓扑/频谱 caliper 的同队支持，以及保持每个 eigenmode 功率不变的谱符号随机化控制。P1 的 250 个点集样本和两种 probe graph（P1 weighted kNN4、weighted Delaunay）作为输入，暂不加载任何模型。

首轮 3 个真实样本产生 576 个 matched-set、2304 个干预臂；1378 行有效，926 行因坐标边界或匹配条件不满足而显式标记为无效，未静默删除。有效行最大能量误差约为 `7.1e-15`，exact-spectrum 控制的逐模态功率最大误差为 `0`；两种图没有出现 `GRAPH_INVALID`。这证明的是构造和审计链可运行，不是组织效应成立，也不是模型结果。

### 当前边界

matching-only 的能量和谱保持校验通过，但只有 98/576 个 matched-set 在首轮 smoke 中四臂完整，因此产物状态是 `INCOMPLETE_MATCHING_ONLY`，不能写成匹配阶段 PASS。高能量边界和严格拓扑/频谱 caliper 是主要缺口，正式运行前需评估匹配覆盖率和是否需要探索性调整 caliper。P2-C 单模型响应 smoke、P2-D 正式推理和 C2 统计仍为 `NOT_RUN`。

### 证据

- 匹配器：[scripts/p2_matched_controls.py](scripts/p2_matched_controls.py)
- matching-only 管线：[scripts/run_phase2_pipeline.py](scripts/run_phase2_pipeline.py)
- 首轮产物：[artifacts/phase2/smoke_matching/pipeline_summary.json](artifacts/phase2/smoke_matching/pipeline_summary.json)
- 独立远程经验目录：[docs/ai_galaxy_remote_pitfalls/README.md](docs/ai_galaxy_remote_pitfalls/README.md)

## QA-2026-08-31-013：P2 当前阻塞是什么，应该怎样优化？

### 问题

P2-B 为什么只有 98/576 个完整 matched-set？这是代码、边界还是科学设计问题，下一步应该做什么？

### 通俗结论

当前没有外部资源阻塞，也不缺 GPU。主阻塞是“同一帧、同一球队、同样节点数，并且拓扑和频率都足够接近”的非语义支持集不容易找到；次阻塞是固定方向平移会把靠近球场边缘的节点推出 `[-1,1]`。两者彼此独立，不能只靠放宽一个阈值解决。

576 个 set 中，topology-frequency 控制有 433 个没有合格候选，是最大的损失来源。边界问题随能量迅速恶化：四臂完整数从 `epsilon=0.25` 的 38/144 降到 `epsilon=2.0` 的 2/144。因此 `2.0` 目前只能作为压力测试能量，不能和低能量一起支撑完整剂量结论；但它仍要保留并报告，不能静默删除。

首轮 3 个样本还都来自同一场比赛，所以不能把 576 个干预组合误当成跨比赛重复。额外做的 response-blind 检查每场比赛只取 1 帧：当前 band-power L1 上限 `0.20` 时 topology 控制为 39/114；保持语义重叠上限不变，仅把 band 上限试探到 `0.30` 时为 87/114。后者的 RQ 差异仍不超过 `0.0459`，但 band 差异允许达到 `0.2998`。这证明 band 条件是主要瓶颈，也说明不能只追求覆盖率：放得越宽，“同频率”的解释越弱。

### 最小优化

1. 保留当前 `p2_v1` 结果，不覆盖；增加候选级诊断表，记录每个候选的 RQ 差异、band-power L1、语义重叠、边界余量、失败主因和次因。
2. 在不读取任何模型响应的条件下画覆盖率—频率平衡曲线，再选择一个可解释的匹配版本。它不是预注册，也不禁止后续继续探索；作用只是避免把不同 matcher 版本和同一次模型结果混在一起。
3. 主匹配继续保持同队、同大小和语义重叠不超过 0.5。把重叠放宽到 1.0 虽能增加候选，但会让“随机控制”包含大部分原语义角色，暂不作为主方案。
4. exact-spectrum 控制允许对谱符号做有界、确定性的重复抽样，直到找到不越界的符号组合；每次仍精确保留逐模态功率，并记录尝试次数。32 次 response-blind 试算使 `epsilon=0.25/0.5/1.0` 中所有 anchor-valid set 都找到可行 exact-spectrum 控制；`2.0` 仍有 4/28 个 anchor-valid set 无法解决。
5. 下一轮 matching-only smoke 应按 10 场比赛分层取样，而不是读取文件头部连续 3 帧；继续保留四个方向和全部失败行。

### 下一步顺序

先实现上述 matching revision，并在跨 10 场比赛的小样本上查看覆盖—平衡曲线。随后使用确定版本的低能量完整集，在本地 CPU 上运行一个 DeepSets checkpoint 的 P2-C wiring smoke，只检查加载、配对、有限数值和确定性，不解释语义效应。只有匹配覆盖不集中于少数比赛/方向、残余频率差异可接受后，才进入 250 样本、9 模型的 P2-D。

短期内不需要 GPU；P2-C 的单模型、20 节点推理本地 CPU 足够。任何模型响应都不能反过来决定 matcher 参数。

## QA-2026-08-31-014：P2 优化和下一步怎样落地实现？

### 问题

如何把当前匹配阻塞的优化、P2-C 单模型 smoke 和后续正式运行组织成可执行任务？

### 回答

实现分为两个连续包。第一个包只处理 response-blind matching：保留 v1 结果，新增候选级几何/频率诊断、边界余量、失败分类、exact-spectrum 有界符号重采样和覆盖—平衡曲线；随后按 10 场比赛每场 2 帧运行分层 smoke。matcher 的选择只看覆盖和残余匹配差异，不看任何模型响应，也不设置预注册式机械阈值。

第二个包才是 P2-C：使用 P1 唯一的 `build_torch_models()` 恢复 `deepsets_ae_seed11`，校验 P1 脚本、checkpoint、model manifest 和 baseline `.npy` 的 hash。P1 baseline embedding 只有在 sample/row/checkpoint/preprocess 全部一致并通过少量重算 parity 后才复用。P2-C 只消费低能量、四臂完整的 matched-set，输出状态固定为 `WIRING_ONLY_NOT_SCIENTIFIC`，不训练模型、不形成 SCG 或 C2 结论。

为降低回归风险，本轮不把模型类迁出 P1，也不修改历史 P1 脚本；P2 通过 adapter 调用 P1 模型工厂并记录 source hash。未来若需要共享模型模块，应另做等价性重构。GAT 在后续正式推理中继续使用 P1 baseline kNN-4 adjacency，不能按干预后坐标重建图。

执行顺序是：matching 测试与实现 → 20 样本分层 matching smoke → 记录 matcher 版本或保持共同支持不足 → 本地 DeepSets response smoke → 根据实测吞吐决定 P2-D 使用 CPU 还是 GPU。前三项均在本地完成，短期不租 GPU。

完整文件、schema、测试、checkpoint 和资源计划见：[docs/08_PHASE2_OPTIMIZATION_AND_NEXT_TASKS.md](docs/08_PHASE2_OPTIMIZATION_AND_NEXT_TASKS.md)。

## QA-2026-08-31-015：为什么 matching v2 把频带差从 0.20 调到 0.30？

### 问题

这个调整是为了让结果更容易出现吗？是否看过模型 response 后才改？

### 回答

不是。matcher 诊断明确拒绝 model、embedding、response、SCG 等字段，只使用节点支持、图拓扑、频率差和边界可行性。20 个样本覆盖 10 场比赛的诊断显示：保持 Rayleigh 差 `0.05`、语义节点重叠上限 `0.5` 不变时，band L1 从 `0.20` 调到 `0.30`，低能量共同支持由 617/1824 增至 1260/1824。

代价也被完整保留。按 frequency ratio 优先、topology 距离次级的正式排序，selected control 的平均 band L1 由约 `0.152` 增到 `0.201`，平均 Rayleigh 差由约 `0.0121` 变为 `0.0124`，平均 topology match score 由约 `1.956` 变为 `2.036`。因此这是用小幅残差变宽换取共同支持明显增加的探索折中，不是为了迎合模型结果，也不是预注册阈值。

证据：`artifacts/phase2/matching_v2_stratified_smoke/coverage_balance_curve.csv`、`artifacts/phase2/matching_v2_selected_smoke/selected_control_balance.parquet`、`configs/phase2_matching_v2_selected.yaml`。

限制：`0.30` 不是永恒的“正确值”。正式解释必须同时报告实际 band 残差，并可把更窄版本作为敏感性参照。

## QA-2026-08-31-016：P2-C 的 DeepSets 是怎样接入的，baseline parity 又是什么意思？

### 问题

P2 是否重新实现或重新训练了 DeepSets？为什么还要检查 baseline parity？

### 回答

P2 没有重新训练，也没有复制一份模型架构。`scripts/p2_point_model_adapter.py` 直接调用 P1 的 `build_torch_models()`，实例化 `deepsets_ae_seed11`，核对 P1 脚本、model manifest 和 checkpoint SHA256，然后用 `torch.load(..., weights_only=True, map_location="cpu")` 恢复权重，并固定在 `eval()` 和 inference mode。

baseline parity 是检查“同一个原始阵型经过当前接线后，是否仍得到 P1 保存的同一 embedding”。最终运行的 20 个 smoke 样本逐元素完全相等，最大绝对差为 `0`。测试也保留显式 float32 数值等价容差，以覆盖不同 batch 形状可能产生的矩阵乘法末位差异；容差、exact-equal 状态和误差都写入 `baseline_parity.json`，没有静默忽略。

证据：`artifacts/phase2/p2c_deepsets_seed11_smoke/model_receipt.json`、`baseline_parity.json`、`source_provenance.json`。

限制：P2-C 只证明工程连接正确。它没有比较模型、没有检验角色组织效应，也不能产生 SCG/C2 科学结论。

## QA-2026-08-31-017：GAT 在干预后为什么仍使用原始 kNN-4 图？

### 问题

球员坐标改变后，为什么不根据新坐标重建 GAT 邻接图？

### 回答

因为 P1 的 GAT 学到的是“原始观测下固定 weighted kNN-4 图上的位置变化”。如果干预后重建图，节点位移和图边变化会同时发生，response 就混合了两种机制，无法判断变化来自坐标还是来自换图。

所以 P2 adapter 对 GAT 强制要求 adjacency，并只接受 P1 canonical artifact 中保存的 baseline weighted kNN-4 adjacency。P2-D 的 GAT-AE 和 Phase-GAT benchmark 也遵守这个约束；matching 使用的 Delaunay/kNN probe graph 只定义干预与匹配特征，不替代模型的 P1 baseline adjacency。

证据：`scripts/p2_point_model_adapter.py`、`artifacts/phase2/p2d_resource_gate/benchmark_receipt.json`。

限制：固定图回答的是“固定关系结构下坐标扰动的表示响应”。如果以后要研究关系边本身变化，应建立独立干预族，不能与当前结果混写。

## QA-2026-08-31-018：下一步需要 GPU 吗，为什么正式 250×9 还没有直接运行？

### 问题

既然 P2-C 已跑通，是否应该马上租 GPU 并运行 250 个样本、9 个模型？

### 回答

不需要 GPU。P2-C 的 5040 条 DeepSets response 在本地 CPU 上端到端约 13.8 秒，峰值 RSS 约 979 MiB。P2-D 又对 DeepSets-AE、GAT-small-AE、Phase-GAT 各做了一个 512 条有界 benchmark；结合共享机器负载后的当前保守外推，9 个点模型约为数分钟量级，输出约 12 MiB。模型顺序执行时内存也不应乘 9。因此当前结论是 `LOCAL_CPU_SUFFICIENT_GPU_NOT_JUSTIFIED`。

正式 250×9 保持 `NOT_RUN`，原因不是算力，而是还需选择首个探索分析范围：是否先聚焦 `epsilon=0.25/0.5`、如何按 match 聚合避免伪重复、怎样并列报告完整集和边界缺失。它们是开放式研究选择，不是预注册限制。选择后可以直接本地推进。

证据：`artifacts/phase2/p2c_deepsets_seed11_smoke/runtime.json`、`artifacts/phase2/p2d_resource_gate/resource_estimate.json`、`docs/09_PHASE2_IMPLEMENTATION_RESULTS.md`。

限制：资源外推没有包含全 250 样本 matching 的 CPU 成本，且完整集密度来自 20 样本；正式运行前仍应保留运行收据和缺失状态。

## QA-2026-08-31-019：P2 rigid 正式运行完成后，究竟发现了什么？

### 问题

正式 250 样本、9 模型的 rigid 干预怎样实现？结果是否证明模型忽略了足球组织？

### 通俗回答

这次把一个角色联盟中的球员作为整体移动：联盟内每个球员得到完全相同方向和大小的位移，联盟内部形状不被打散；总位移能量固定。每个 semantic coalition 同时配三个对照：同队同人数随机支持、拓扑与频段近似匹配支持，以及逐模态功率完全相同但谱符号重新随机化的 exact-spectrum control。

全部 250 帧先在完全不读取模型响应的情况下完成 matching，之后才让 9 个冻结模型计算干预前后的 embedding 余弦距离。GAT 和 Phase-GAT 始终使用训练时的 canonical kNN-4 图；Delaunay/kNN-4 probe graph 只定义 matching 的频率，不替换模型图。统计先在同一 matched set 内做 semantic-control 差，再依次平均方向、coalition、team、role、sample，最后把 10 场比赛作为等权独立单位；三个模型 seed 不当作额外比赛。

结果不是简单的“有”或“没有”：

1. exact-spectrum control 已经和 semantic 干预具有完全相同的逐模态功率，但三类架构、两种图和两档低能量下，semantic 响应仍全部更大。这说明模型不仅看各频率有多少能量，还对这些频率怎样在空间中排列、也就是谱相位或空间组织敏感。
2. 但与 topology-frequency control 相比，DeepSets 接近零或为负，GAT 两族更常为正，kNN-4 与 Delaunay 也不同。把 band-power L1 从选定 matcher 的 0.30 收紧到 0.20 后，低能量结果都不能稳定离开零。
3. defender、midfielder 和 forward 的方向不同：defender 相对 topology control 整体偏负，midfielder 偏正，forward 混合。总体均值会掩盖这一点。

因此不能说“模型已经理解或忽略了足球组织”，也不能说“角色联盟在所有模型中都特殊”。当前最准确的标签是 `mixed_or_graph_specific`：存在空间组织敏感性，但角色联盟效应依赖模型、图、角色和匹配严格度。下一步应单独做 fracture continuity，用完全相同的非零位移向量 multiset 重新分配端点支持；它不能与本次 rigid 结果混合。

### 实现或文档证据

- 正式配置：`configs/phase2_rigid_formal_v2.yaml`
- 正式编排：`scripts/run_phase2_rigid_formal.py`
- match-level 统计：`scripts/p2_statistics.py`
- C2 报告：`reports/EXPLORATION_CHECKPOINT_2.md`
- claim ledger：`reports/P2_CLAIM_LEDGER.md`
- 完整产物：`artifacts/phase2/p2_rigid_formal_v2/`

### 当前限制

- 低能量四臂完整率约 63%–73%，高能量 2.0 只有约 6.6%–6.9%；
- exact-spectrum control 严格控制频谱功率，但通常是稠密位移，不控制局部支持形状；
- raw cosine-distance 差值很小，尚无下游任务证据说明其实际性能意义；
- heldout 只有两场，静态 role 也不等于动态战术真值；
- fracture continuity 和中国书法跨域验证尚未运行。

## QA-2026-08-31-020：建议的 fracture continuity 属于哪个阶段？

### 问题

下一步建议实施的 fracture continuity 属于总方案中的 P 几？今后怎样标注建议所属阶段？

### 回答

该任务属于 **P2（Novelty Discriminator）的补充诊断分支**，不是 P3。它保持非零位移向量 multiset 不变，只重新分配端点支持，用来继续区分“局部组织或连接方式的影响”和“普通频率功率偏置”。这仍在回答 P2 的核心判别问题。

当前 `P2_RIGID_C2_COMPLETE` 表示 rigid intervention 正式分支已经闭合；fracture continuity 是独立 operator，状态仍为 `NOT_RUN_SEPARATE_OPERATOR`。由于 rigid 结果是 `mixed_or_graph_specific`，应先完成这一项最小诊断，再决定是否进入 **P3（Causal Mechanism）** 的增强耦合、unary/relational pooling、invariance/equivariance 和 layer-wise mechanism 实验。

今后的任务建议统一附带以下信息：

- `所属阶段：P编号 + 阶段名称`；
- `任务性质：主线 / 补充诊断 / 阶段门 / 后续分支`；
- 若不是顺序进入下一阶段，说明为什么仍停留在当前阶段。

例如：`建议任务：fracture continuity；所属阶段：P2_NOVELTY_DISCRIMINATOR；任务性质：rigid 分支完成后的补充诊断。`

## QA-2026-08-31-021：P2 fracture continuity 下一步怎样实现？

### 问题

在 rigid P2 得到 `mixed_or_graph_specific` 后，下一步 fracture continuity 应怎样实现？

### 回答

建议采用最小两臂正式实验：新生成角色组内 node-independent 二维 fracture anchor，再把完全相同的非零位移向量 multiset 一一重排到同队、同大小且完全不重叠的端点支持。控制从全部合法支持与向量—端点双射中响应盲地选择 topology/frequency 最近者。

该任务属于 `P2_NOVELTY_DISCRIMINATOR` 的补充诊断，不是 P3。正式范围为 250 个样本、10 场比赛、1,425 个 defender/midfielder/forward 角色组、3 个 fracture draws、`0.25/0.5` 两档能量、`knn4/delaunay` 两个 probe graph 和九个冻结点集模型；不训练模型，不运行视觉网络，也不进入书法或 AMR。

不直接复用旧 P1 anchor 数值：P1 只覆盖 defender，且不同能量使用不同随机位移，旧谱字段也未使用完整二维功率。新 anchor 用全精度基础向量场按能量缩放，从而形成可解释的低能连续路径。普通随机支持只作为候选池描述，within-support permutation 仅在主结果显示 assignment fragility 时追加，exact-spectrum 不重复运行。

工程完成 checkpoint 为 `P2_FRACTURE_C2_COMPLETE`。只有两臂匹配、九模型响应、match-level 统计、独立报告和 provenance/hash 链全部闭合才算完成。结果若跨比赛、图、能量和模型仍保留组织残差，才进入 P3 最小因果矩阵；若趋近频率解释、结果混合或匹配不足，则按对应分流停留在 P2 或停止该机制主张，不使用机械显著性阈值。

### 完整方案

- `docs/10_P2_FRACTURE_CONTINUITY_PLAN.md`
