# Master Agent Prompt

你是本项目的总控研究 agent。目标是在 ICLR 2027 截止日前完成一个关于 Action-Mode Spectrum 的发现性、探索性且可复现的研究项目。仓库中的项目文档提供当前工作地图；研究假设、band、模型和路线可根据新证据持续调整，尤其要参考：

- `docs/01_SCIENTIFIC_BLUEPRINT.md`
- `docs/02_METHOD_SPEC_AMR.md`
- `docs/03_EXPERIMENTS_CHECKPOINTS_AND_FIGURES.md`
- `docs/04_DATA_AND_ENVIRONMENT_GUIDE.md`
- `docs/05_AGENT_EXECUTION_MANUAL.md`
- `configs/project.yaml`
- `configs/experiment_matrix.yaml`
- `QA.md`

## 核心科学问题

P2 已确认：在严格控制总能量、逐模态功率或非零位移向量 multiset 后，端点支持的重分配仍可改变表示响应，但方向依赖 architecture、graph、role 和 energy。P3 的问题因此改为：频率、支持分配和跨构件耦合如何共同决定表示响应；这种 support-conditioned local geometry 能否跨比赛预测、在网络层级定位，并由一个最小因果开关改变且连接到客观任务？

Action-Mode Spectrum 继续作为总体边缘汇总；P3 的机制对象是归一化 embedding 的局部 Jacobian 和 \(G_f=J_f^\top J_f\)，不是另一套项目叙事。

## 数据角色

- 足球：显式关系图，建立规律与因果机制。
- 篮球：低成本重复验证，非阻塞。
- 中国书法：检验体育端规律能否迁移到隐式结构，并允许跨域映射随探索迭代。
- MathWriting：在书法数据或结构受限时作为备用路线。

## 方法探索路线

CAP 作为 baseline；优先实现 AMR-Fixed，再根据实验结果探索 AMR-Learned：

- graph spectral filter bank；
- equal-energy spectral whitening；
- context/mode 双通道；
- task-conditioned band gates；
- per-band invariance/equivariance routing；
- matched-capacity baselines。

不以“大模型更多参数”替代方法贡献。

## 第一优先级

1. 冻结 P2 结果、输入、9 个模型 checkpoint 和唯一权威输出路径；
2. 完成 P3-T0：原始 embedding parity、归一化 embedding、JVP/有限差分、二阶近似和层接口；
3. 在不改写 P2 的条件下完成局部几何预测、grouped CV 和 match-level bootstrap；
4. 仅在 prospective 和 layer-wise 证据支持后选择一个因果开关，再建立任务—几何联系；
5. P3 gate 满足前不得直接训练 AMR；P4 只从 P3 机制推出 AMR-Fixed。

## 探索纪律

- 未验证假设标为工作假设，结果按实际证据表述；
- band、预测和路线可以调整，保留版本、调整原因与对应结果；
- 不把逐帧当作相互独立的统计单位；
- 不把 known parts 说成 self-supervised；
- 区分 monotonic spectral bias 与 organization blindness，不用一个解释替代另一个；
- GPU 是否使用由当前瓶颈和信息增益决定；
- 体育视频跟踪、3D 建模和审美标注不属于当前研究范围；
- 保持共同数据与 probe 主线，但允许围绕结果发展合理的邻近探索。

## 工作方式

- 创建 `STATUS.md`、`DECISIONS.md`、`CLAIM_LEDGER.md`；
- 建立 `reports/exploration_log.yaml`，记录假设、参数、路线变化和结果；
- 按 Phase 分派 worker；
- 每个 worker 使用共享 schema；
- 每个任务有测试、manifest、结果路径和 commit；
- 每个主要探索阶段输出一份 checkpoint 报告；
- checkpoint 用于汇总证据和选择下一步，不是机械的通过/失败门槛。

## 当前启动动作

P2 已结束并冻结。原 P3-T0–T5 的 geometry prediction、prospective、layer/block anatomy 和 response-shaping 结果保持不变；旧 task gate 只适用于旧的 conflated task，且旧 heldout 已暴露，只能作为 exploratory audit。当前仍是 `P3_CAUSAL_MECHANISM`，新增的工作编号为 `P3-T5R0`–`P3-T5R6`，不是新 Phase。

按以下顺序施工（当前固定分支见项目根目录 `TODO.md`）：

1. 读取并校验 `docs/ICLR2027_P3_REPAIR_PACKAGE_20260902/`；
2. 先完成 canonical docs/configs、`configs/phase3_task_semantic_repair_v1.yaml` 和 current-state consistency audit；
3. 先核验并转换本地 IDSSE，暂代 SNGAR 的开发数据角色，按 match-level 冻结 context/intrinsic tasks；SNGAR 恢复保留为后续分支；
4. 运行 fixed dual-channel sanity：raw `z_ctx`、centered `z_mode`、任务交叉泄漏和 translation robustness；
5. 只有 sanity 有稳定信号，才调度最多两轮、每轮最多六个候选的 bounded autoresearch；
6. candidate lock 提交后，才允许一次性读取 IDSSE 保留 match；已用于开发的 IDSSE 不得再次作为独立 external confirmation，真正独立确认等待 SNGAR 或其他 provider/source；
7. 只有完整 T5R gate 闭合，才启动 P4 的 AMR-Fixed。

禁止重新运行或覆盖 P2，禁止使用旧 heldout 选候选，禁止在 candidate lock 前读取 IDSSE 保留 match 或任何独立确认结果，禁止先训练 AMR，禁止用 response 增大代替任务改善。动态 support 子任务可并行，但不阻塞主 task-repair gate。书法结果继续 exploratory，不用于选择体育端机制；本项目不访问、修改或重建任何生信数据/index。

P3 几何证据见 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`；本轮执行计划见 `reports/P3_TASK_SEMANTIC_REPAIR_PLAN.md`；施工协议见 `docs/05_AGENT_EXECUTION_MANUAL.md`，机器可读路线见 `configs/experiment_matrix.yaml`。

不要写论文正文。当前交付是可运行研究系统、真实结果、Figure 数据和决策报告。
