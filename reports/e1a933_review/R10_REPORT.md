# R10 论文口径修订

任务类型：Operational/non-code，外加表生成器的格式/说明修订。未启动模型训练或把样例测试算成科学复现。

已完成：摘要压缩并加入跨任务输入优势与T2负交互边界；正文加入T1/T2的quartet/parent分母；视觉段加入ResNet18强基线、上采样与初始化边界；discussion纠正过时的“只测小CNN/像素无J”；附录生成器同步R01–R10的历史结果与归因限制。所有旧模型数字均保留，不修改旧artifacts/reports。

`AFFECTED_CLAIMS.csv`和`claim_diff.csv`包含原结果、发现问题、作用范围、修正结果、最终措辞，并增列竞争解释。状态表示claim本身的文稿处理，不代表对应实验已重跑。R04生产字段已按起点置信修正并复算72行；稿件说明分母和范围，不将它冒称新的P1B证据。

最小检查：从原始逐quartet CSV复算four-arm值，与原U1 summary一致。附录由修改后的生成器生成。修复了生成器未转义百分号导致LaTeX把句尾当注释的问题。现有TinyTeX可编译；最后构建信息见PAPER_PATCH_AUDIT.md。

已合并结果：U10未舍入逐样本I与parent CI、18 typed重训及完整pipeline数值对称性；视觉旧renderer492/512轨道相同与canonical512/512；O04的12 encoder×6 heads（仅exposed-bank）；R07同起点30臂；R08几何E可构670/1449候选、48/155已选传球；R09九组dev预算外推。以上均从新artifacts读取，未修改旧产物。

原21臂视觉GPU矩阵已完成并独立复核：O03支持同总预算static→flip联合修复；N01改善辅助几何可学性但未稳定增加J。追加N02/N03/O05/O06待各自结果与独立解读，当前稿件不提前升级其科学结论。R03发现旧D09 n218剔除源于namespace错误，已撤销“去重”措辞并整合96 checkpoint×3 bank坐标矩阵；O01新鲜1229quartet/250parent优化比较已进入主文与生成表格。未执行的新实验不以文稿修订代替。

最终构建：paper_revised.pdf，20页/主文9页，零未定义与overfull；生成器校验/日志/哈希见PAPER_BUILD_RECEIPT.json。视觉O03/N01已完成并入稿；后续GPU结果到达仍须另行更新，不宣称整包科学完成。

追加：D03/D09共享8个真实E父(74/14quartet)已写入附录和ledger。N02全部12checkpoint输入切换表已写入附录，保持有界旧bank诊断、机制UNRESOLVED，不升为因果或新方法结论。

## GPU21臂结果迁移

已读取VISION_GPU_RESULTS.md、N03_GATE_AND_PROTOCOL.md、independent_summary.json及paired_analysis。摘要/视觉正文/局限更新：同1760optimizer steps的static→flip平均J差14.47/20.34pp，parent曝光分配并不完全匹配；所有6个seed/初始化配对区间为正。N01 matched aux误差更低但对ordered/BCE的J区间均跨零。附录含全部21臂、6组迁移/111回归分母及经典像素J114/159，不声称神经优越。
