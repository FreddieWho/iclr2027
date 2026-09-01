# ICLR 2027 Action-Mode Spectrum：P3 任务语义修复与数据扩展包

生成日期：2026-09-02  
审计对象：`FreddieWho/iclr2027`，以当前公开主分支及最近 P3 提交为基础  
用途：交给总控研究 agent 原位更新现有方案、获取数据、修复实验协议，并继续执行一个有界的 P3 机制闭环。

## 一句话结论

当前已经较强地证明：**support-conditioned normalized-embedding local geometry 能预测并解释模型的表示响应，而且 pooling 能塑造这种几何。**尚未证明的是：**这种塑形能够改善语义正确的下游任务。**旧 T5 的主要缺陷不是单纯“算法没调好”，而是把需要保留绝对位置的 context task 与需要忽略绝对位置的 intrinsic task 放进了同一个互相冲突的评价门。

本包因此不重启 P2、不直接启动 AMR，也不新增 Phase。它在既有 `P3_CAUSAL_MECHANISM` 内授权一个有边界的 `P3-T5R` task-semantic repair lane：

1. 修复 heldout 暴露和候选冻结协议；
2. 扩充独立比赛，而不是扩充同场相邻帧；
3. 分开 context 与 intrinsic structure 任务；
4. 先做固定双通道最小检验；
5. 再进行最多两轮、每轮不超过六个候选的单轴 autoresearch；
6. 候选锁定后只运行一次未暴露测试和外部数据复现；
7. 成功才进入 P4 AMR-Fixed，失败则冻结为诊断/机制论文。

## 文件索引

1. `01_DATASET_ACQUISITION_AND_SPLIT_PLAN.md`  
   数据源、下载地址、用途、许可证、冻结分割、动态协同支持构造方法和降级路线。

2. `02_EXISTING_PLAN_UPDATE_INSTRUCTIONS.md`  
   给方案迁移/更新 agent 的原位修改规范。逐文件说明应更新什么、保留什么、禁止改什么。

3. `03_MAINLINE_RESEARCH_AGENT_PROMPT.md`  
   给总控研究 agent 的完整施工 Prompt。涵盖文档同步、数据转换、任务重构、最小双通道、候选锁定和外部验证。

4. `04_AUTORESEARCH_TASK_REPAIR_PROMPT.md`  
   给 task-repair autoresearch agent 的有限搜索 Prompt。严格禁止读取最终测试集，限制搜索自由度。

5. `05_AUTORESEARCH_DYNAMIC_SUPPORT_PROMPT.md`  
   给动态“语义小组”构造子 agent 的 Prompt。用于检验静态 defender/midfielder/forward 是否只是粗糙 proxy；它不是 P4 的阻塞条件。

6. `06_CURRENT_STATUS_AND_REVISED_ROUTE_REVIEW_REPORT.md`  
   可直接提交第三方评审的完整报告，包括科学问题、方法、数据、所有关键结果、证据等级、代码/协议风险、更新方案和审稿问题。

7. `configs/dataset_acquisition_manifest_v2.yaml`  
   机器可读数据清单和 split-lock 规则。

8. `scripts/download_priority_datasets.sh`  
   下载命令模板。默认只下载开发数据；最终测试数据必须存在 `candidate_lock.json` 才允许下载。

9. `templates/candidate_lock.schema.json`  
   候选冻结文件的 JSON Schema。

## 推荐使用顺序

```text
02_EXISTING_PLAN_UPDATE_INSTRUCTIONS.md
    ↓ 原位更新仓库方案与状态
03_MAINLINE_RESEARCH_AGENT_PROMPT.md
    ↓ 总控施工
01_DATASET_ACQUISITION_AND_SPLIT_PLAN.md
    ↓ 下载与转换
04_AUTORESEARCH_TASK_REPAIR_PROMPT.md
    ↓ 有限候选搜索
05_AUTORESEARCH_DYNAMIC_SUPPORT_PROMPT.md
    ↓ 可并行但非阻塞的动态 support 检验
06_CURRENT_STATUS_AND_REVISED_ROUTE_REVIEW_REPORT.md
    ↓ 第三方评审与最终 go/no-go
```

## 最高优先级数据

- 开发主数据：`OpenSportsLab/SNGAR-Action-Spotting-Tracking`，64 场完整 tracking，官方 45/9/10 场 train/valid/test；只先下载 train+valid。
- 当前域动态 support：SkillCorner Open Data 的 `dynamic_events.csv` 与 `phases_of_play.csv`，特别利用 pressing chain、simultaneous engagement、off-ball run、passing option 等字段。
- 外部来源确认：IDSSE，7 场德甲一/二级联赛官方 25 Hz tracking + synchronized events，CC BY 4.0。
- 采集域迁移：SoccerTrack v2，10 场全场景 panoramic tracking + 12 类 BAS 标签，CC BY 4.0；仅在主候选锁定后使用。
- 书法后续确认：Make Me a Hanzi 负责矢量受控干预；HCSU 或 MCCD 负责自然作者/风格/结构任务。书法不得反向选择体育端机制。

## 关键纪律

- P2 artifacts、report、manifest、hash 继续只读。
- 旧候选搜索所见 heldout 必须标记为 `EXPOSED_DURING_CANDIDATE_SEARCH`，仅作 exploratory audit。
- SNGAR test、IDSSE external-confirmatory 和 SoccerTrack final matches 在 `candidate_lock.json` 之前不得读取标签或结果。
- 不将更大的 embedding response 当作更好；必须依任务语义定义应保留或应不变的信息。
- 不把静态角色组称为真实动态战术协同单元。
- 不通过换 band、换 graph、换阈值或筛选 seed 来“救回”统一 notch。
- 不以同一比赛更多帧冒充更大独立样本量。
