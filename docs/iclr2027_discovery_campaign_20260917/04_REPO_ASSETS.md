# 04｜资产接线与旧流程替换
基准 commit 已在本次通过 GitHub 读取确认；本机未克隆/重跑完整项目。

## 已知入口
| 入口 | 用途 | 本轮怎么用 |
|---|---|---|
| `scripts/run_t5r3_sanity.py` | TaskModel/GraphEncoder、图、raw/centered views、task heads | 引入计算类，绕过与本轮无关的旧lock函数 |
| `scripts/p3_support_geometry.py` | 历史模型的JVP/几何计算 | 作为冻结模型的候选提案器 |
| `artifacts/phase1/models/` | 历史多架构checkpoints | 低成本试验；存在性由inventory验证 |
| `artifacts/phase3/task_semantic_repair_v1/t5r3_sanity_v3/` | raw/centered/dual等模型与清单 | 复用训练资产，不重跑sanity |
| `artifacts/phase3/task_semantic_repair_v1/` | IDSSE视图、各轮训练、T5R5/T5R6 | 只取必要输入与权重；不假定所有大文件都推到git |
| `artifacts/data_v2/idsse/` 与 `artifacts/data_v2/soccertrack/` | 已转换比赛输入 | 有本地授权副本就直接读，无需重新下载 |
| `artifacts/phase4_amr/` | AMR/CAP/teacher分支 | 可作为被测试对象，不作为待救援项目 |
| `paper/` | 旧稿 | 先另写new-story目录，选定后按07替换 |

报告明确T5R3有12模型，但manifest不是checkpoint字节。总控应按训练脚本和model_results实际找权重，不猜文件后缀/键名；缺checkpoint仅该支阻塞。所有下载/输入保持原许可。

## 代码语义提醒
当前T5R3 encoder名含Phase-GAT，但实现是固定加权图message passing，不应在新表格写成标准可学习attention GAT。raw/centered双视图、mode-head前后输出应明确标记；真实前向始终走被评价模型完整路径。

对于动态图：真实前向按部署方式重建图；局部梯度可以固定当前图做近似，但不得省去真实前向中的重建。`graph_policy`写进结果行，不用另一种图偷换模型。

## 新目录
```text
experiments/discovery_campaign/      # agent写模型adapter和六路runner
artifacts/discovery_campaign/        # 所有新结果，分route/run
reports/discovery_campaign/          # 结果卡和阶段取舍
paper/discovery_story/               # 胜出新主线稿件
```

本包脚本默认只读查找和在指定新输出目录写入，不改旧文件。若旧root instruction继续要求先做旧gate，总控在CAMPAIGN_STATE注明用户本轮授权与优先级覆盖后开展新目录实验，不需要再问用户。
