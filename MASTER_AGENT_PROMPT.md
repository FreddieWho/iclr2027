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

多构件系统中，同一几何算子作用于不同构件联盟时，表征的敏感度是否与任务语义对齐？现代表征是否在等能量条件下错误优先响应共同模式，或在有意义的子群协同频段形成盲区？这些规律是否能由增强/聚合机制解释并由 AMR 控制？

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

1. 建立数据与 intervention schema；
2. 完成等能量 Action-Mode Probe；
3. 观察并解释初始作用谱；
4. 用同频语义联盟 vs 随机联盟判断组织效应是否超出频率；
5. 根据前述结果决定 AMR、机制或邻近问题的探索重点。

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

依次执行：

```bash
bash scripts/bootstrap_env.sh
bash scripts/download_public_data.sh --core
python scripts/verify_data.py --manifest configs/data_manifest.yaml
bash scripts/run_phase0_smoke.sh
```

随后检查失败信息并修复最小阻塞。完成初始 smoke 后，先生成：

1. 一场 SkillCorner 比赛的 canonical samples；
2. 6 个频段的等能量干预；
3. semantic/random matched coalition；
4. 一个 DeepSets 和一个冻结视觉 backbone 的 spectrum；
5. 初版探索性现象报告。

不要写论文正文。当前交付是可运行研究系统、真实结果、Figure 数据和决策报告。
