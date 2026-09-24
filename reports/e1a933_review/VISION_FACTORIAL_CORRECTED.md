# R06：视觉factorial纠正

生产修复完成；新O03 GPU矩阵已完成，独立结果见`VISION_GPU_RESULTS.md`。下列旧factorial为归档重算，不与新renderer协议混合。

`d04_vision_train.py` 将`headonly`实现为backbone.eval + fc.train，并在完整训练前后比较所有backbone参数/缓冲区哈希；`bnadapt`明确允许BN统计更新。历史headonly结果必须称BN-adaptation probe，不回填为纯冻结。新字段A_acc、B_acc、atomic_joint_acc澄清历史atomic_acc实际为两原子同时正确；新保存每parent logits/labels。

64→224仍然只是相同64图像的双线性上采样，不是新增观测信息。三个seed的旧归档J重算（四位小数，非重新训练）保存`artifacts/e1a933_review/vision_regression/legacy_factorial.json`：224 pretrained-random分别+72.03、+7.49、+9.87pp，平均+29.80pp；collapsed random seed803保留。init×input-scale交互平均+11.88pp，三个seed范围−11.89至+54.48pp。三seed的t区间非常宽，不能宣称稳定正交互或“非预训练”；未经截断的区间超过概率范围仅反映样本极少的近似t不确定性，不是概率预测。

新实验统一canonical同色端点渲染，native64上采样224；3个seed×pretrained/random×static/flip完整12臂。static与flip每epoch图像曝光次数一致，static重复同一批父场景clean图，flip加入训练域单编辑终点标签；N01各臂共享全部图像/顺序/初始化。不能将此新协议的结果直接减去旧renderer成绩声称同条件repair。独立新测试父场景和完整baseline110配对分析预置在生产runner。

R06专用回归及旧错误重演见`vision_regression/result.json`；旧checkpoint完全未改。
