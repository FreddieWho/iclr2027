# Paper patch audit (e1a933)

源基准：e1a933e6b32dd07c6895c7925cbccd9004da7f54。此次写作先纠正解释，再等待修复结果。

| 文件 | 更新与来源 | 分母/边界 |
|---|---|---|
| paper/sections/00_abstract.tex | 保留源域主要结果；加U10跨任务优势、T2负I与D04视觉基线 | T1/T2为2任务，不是18次独立复制 |
| paper/sections/04_mechanism.tex | 限定结构归因；加U10三seed数值与I | 源dev 237 quartet/106 parent；T1 2724/322；T2 3443/369 |
| paper/sections/04b_boundary.tex | 区分旧小CNN端点实验与ResNet18 J；保留D04全部3seed | 224为同一64像素数组上采样；无static-only则不称repair因果 |
| paper/sections/discussion.tex、conclusion.tex | 更新模型/任务范围、S(head∘encoder)、归因边界 | 不把J*当部署准确率；不把子集量当查询成本 |
| experiments/f095_campaign/gen_app_tables.py → app_f095.tex | 自动从原JSON保留数值，统一R01–R10限定；转义% | all8_raw不是平均模型all8；D03/D09 freshness逐模型 |
| experiments/f095_campaign/gen_fourarm_table.py | 保留CSV→summary断言，输出全部4臂及parent/quartet分母 | 3训练seed不算3独立bank |

来源：artifacts/f095_campaign/U10/U10_T1_EVAL.json、U10_T2_EVAL.json，D04各result.json，D06角色/前事件probe，D08/D08SIX、D10、U04/U07及artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv。新结果未到达前上述附录是明确标识的历史再分析。

构建：使用/home/huyudi/.TinyTeX/bin/x86_64-linux/pdflatex，在paper目录、独立/tmp/e1a933-paper-build输出。修订PDF：`reports/e1a933_review/paper_revised.pdf`（旧paper/main.pdf未替换）。结果逐步合并后重新编译并解析交叉引用；最终20页，结论第9页，参考文献/附录另计；零LaTeX错误、零未定义引用、零overfull。详情与SHA见PAPER_BUILD_RECEIPT.json；编译日志paper_build.log。首次从仓库根编译缺style的问题通过正确cwd解决。

## 已同步的新结果

- `artifacts/e1a933_review/data_u10/U10_CORRECTED.json`：六个原尺度交互及2000次parent bootstrap；18 typed重训。正文使用未舍入逐样本复算的I，历史附录保留旧四位小数算术并注明差别。
- `data_u10/SYMMETRY_PIPELINE_AUDIT.json`：旧/新完整pipeline logit差与预测差；新typed表现不等于架构公平性证明。
- `vision_regression/result.json`：旧renderer492/512完整轨道像素相同，canonical512/512。严格限定不可辨识反例，不假称旧renderer全局H精确不变。
- `readout_refit_20260924/summary.json`：12 encoder、6 selected heads、237 quartet/106 parents；S(head∘encoder)实证反例。旧bank再分析；新任务前瞻预测已完成（见 `O04_FORWARD_INTERVENTION.md`，主判定 MISS）。
- `fair/CONTINUATION_RESULTS.json`：30 matched-forward 300ep arms，237/106；不稳定的保护J差异、未见原子回归。
- `football/FOOTBALL_RESULTS.json`：240 sampled events、1449合法候选、670 searched E；selected48/155。非自然事件率/传球结果counterfactual；部分观测只做past-only解析基线。
- `frontier/FRONTIER_REPAIR.json`：43 dev/96 eval条件可行场景，3encoder×3seed。正文不升级旧操作点排除；附录完整列出统一dev .5目标的九组实际eval coverage/quality/CI，避免挑选有利seed。

所有新增表格均由`gen_app_tables.py`读取结果生成。没有手改数值单元格。原21臂视觉矩阵现已完成并入稿；后续GPU实验在获得各自独立结果前不升级。

- `data_lineage/MODEL_BANK_OVERLAP.csv`：96 checkpoint×3bank；D09 namespace历史错误撤回，N236全未见、4N178未见、16N0未见；dev237全未见。**augmentation 逐状态审计已完成**（392412 状态、quartet 银行 0 exact/0 near，检测器已标定），**96/96 训练来源已由 checkpoint 内嵌 mu/sd 指纹唯一定位**，无剩余 historical inference；唯一阳性披露见下。
- `data_o01/O01_FACTORIAL.csv`：108 candidates按devBCE选、1229quartet/250parent新池；正文增加优化后数据/表示比较，全六size-seed表自动生成。
- `R04_CONFIDENCE_REPORT.md`：72行起点置信复算，保持原P1B范围，D02完成groupavg数值闭合检查。

## 最后R03/N02更新

- `data_lineage/D03_D09_CROSSBANK.json`：旧D03/D09共享8个真实E父，分别承载74/14个quartet。现稿明确禁止把二者当完全独立确认；未来builder修复不回填旧bank。
- `N02_sampling_seed{803,805,806}_n64/result.json`：12checkpoint同一64quartet/58parent输入切换完整表，作为**历史探针**保留（旧整数方形笔刷渲染合同 + 已暴露旧bank，数字不与新矩阵相减）。24,576图像前向、零重训；4000次parent等权bootstrap。所有6个64训练模型直接224输入的J为0，保留803塌缩。低stride/native224/完整重训与新parent前瞻验证**已完成**（见 `N02_SAMPLING_ACCESS_REPORT.md` 与 `N02_INTERPRETATION.md`）；机制仍 UNRESOLVED。

## O03/N01 GPU21臂已完成

来源：`vision_gpu_interpretation/independent_summary.json`与`cuda_matrix/paired_analysis.json`。新同canonical协议159quartets/74test parents，dev707图/128parents；same1760steps总预算static→flip差分别+14.47pp/+20.34pp，但parent曝光分布仍不同。N01三seed matched auxiliary dev误差均更低，J并无一致额外改善。新附录全部21臂表及6组parent-CI/110 repair-migration/111退化分母。经典图像基线由`vision_n03_analytic/result.json`生成（J114/159）。摘要、像素正文及模型局限同步；旧D04结果仍留作不同观察协议的历史证据。

## GPU追加结果同步（2026-09-24）
N03六臂、O05十二臂、O06 272请求已回收验hash并进入app_f095附录和claim ledger。N03几何误差与N01不相当，oracle替换失败；O05学习不超过CV解析基线；O06基础能力不足，不作组合机制外推。N02四十八臂已回收校验、独立重算（48臂 max diff 0.0、250/250 manifest 吻合）并写入附录。

## 本轮（8 lane 收尾）新接线进附录的内容

每次修改后均重新编译验证；主文分页自始未变（结论仍起于第 9 页），页数增长全部来自附录 `app_f095.tex`。

| 来源 lane | 附录新增内容 | 关键分母 |
|---|---|---|
| N02 | 48 臂矩阵（B−A、C−B、D−A、MACs 12.25×、static 对照、CCM、冻结预测逐格对照、机制 UNRESOLVED）；旧 12-checkpoint 探针降为明确标注的历史段 | 178 quartet / 85 fresh parent |
| N01-P1 | 起点置信不降低编辑风险（keep 0.100–0.171、flip 0.049–0.259），终点自置信口径把风险压到 ≈0 的对照；matched 在该轴无一致差异 | 159 quartet / 74 parent |
| O04 | 前瞻干预主判定 MISS（P1 7/8、P2 8/8）；head 级与主干级变化监督是替代关系；白化下分段归属 7/8→4/8；D10 回声的定义域边界 | T1 2724/322；T2 3443/369 |
| R09 | 全 256 场景部署估计（不可行 117/256=45.7%；dev 覆盖 0.4884→实测 0.1455–0.3239）；不可行场景成本占比；阈值多重性；oracle 诊断上界 | 256 场景 × 49 候选；eval_full 213 |
| O02 | 匹配容量（6681 vs 6529–6849）+ 匹配优化预算后 typed 劣势仍存（T1 +0.486…+0.529，T2 +0.701…+0.755）；机制是拟合失败（训练目标差 ~10³） | T1 2724/322；T2 3443/369 |
| O01 | 距离优势在一个被充分搜索的 raw 基线之上仍成立（等配方 +0.434…+0.561；扩优化器族 +0.475…+0.574；两分布不重叠）；warm-start 不帮忙；保护类损失买“不忘”不买“更对” | 1229 quartet / 250 parent；648 训练 |
| R03 | 逐状态血缘口径与 0 exact/0 near 结果；检测器标定；`d09fresh665` 单编辑重叠的**强制披露**；96/96 来源指纹定位 | 96 checkpoint × 3 bank；392412 状态 |
| R08 | 自然 E 发生率（0.55% 情境级、4.94% 事件级）与搜索可构率（46.2%）的 84× 差异；条件率；窗口依赖；真实帧对照；旧统计更名 | 1449 情境 / 7117 帧对 |

### 本轮发现并修复的两处自身/上游缺陷

1. **R05/R06 合同检查**：`normalization_commutes_<p>` 四项不可能失败，已替换为可失败检查并附 mutation 证据（`test_vision_contract.py`、`vision_contract_mutation_check.py`）。
2. **线程数口径混杂**：附录 U10 表的 typed 列是 **2 线程**评估（该设置下逐字节可复现），而 I 列来自 raw/six（**4 线程**可复现）；同一 typed flip 臂最多相差 9.0pp。已在表格 caption 与正文披露，禁止跨口径比较。

另：claim ledger 原表两行缺 `denominator` 单元格（导致其后各列左移）已修复；本轮新增行中的 2 处未转义 `|` 已改用 `\lvert`/`\rvert`。现全文按未转义管道数校验，所有行均 11 列。
