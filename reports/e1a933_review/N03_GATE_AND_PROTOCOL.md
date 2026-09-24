# N03门槛判定与追加实验合同

状态：门槛按真实辅助学习结果触发；6次GPU训练已由总控启动，当前科学结论待结果。

N01三个seed的matched dev轨道MSE均显著低于ordered/shuffle和BCE未训练辅助头，且第20epoch更低；与此同时J相对ordered/BCE无一致改善。没有事后发明“必须85%”等阈值。这里的“学会”仅指真实可预测关系误差明显改善，不指精确几何重建，尤其805主任务所选epoch2仍有较大误差。它满足“已有信息可能未充分进入决策”的可检验情形，不证明这一机制已成立。

6臂：geometry_bottleneck与generic_bottleneck各seed803/805/806。全部从相同ImageNet ResNet18初始化，20epoch、batch32、FP32；采用同冻结bank，同样single-dev BCE选epoch。两臂完全同结构和参数量：图像主干→10维向量→按合法端点群平均的MLP(10,64,1)→分类，没有任何绕过10维向量的RGB通路。geometry臂增加整体匹配的rel10监督；generic臂无几何监督。因此generic的10维没有被赋予真实几何意义，但控制了相同容量与同样群平均结构。

使用相同train-only初始梯度标定规则确定geometry辅助权重；训练/开发几何误差与主任务曲线同时记录。不得使用test几何/标签选超参。

每个训练好的关系head额外保存oracle_replacement_DIAGNOSTIC：把预测10维替换为真实几何的同尺度rel10，再经冻结的同一head。它仅用于分辨感知误差与关系计算误差，不计入图像部署性能。对generic臂同样保存该替换，但其特征没有几何语义，不能将generic的替换差解释成感知误差。

单独CPU图像解析基线：已知红/蓝颜色分割→PCA拟合各线方向→按renderer固定2px方形stroke校正端点→解析proper crossing。不读取真坐标，不按标签调参；失败拟合固定输出负类并保留分母。实现`vision_n03_analytic.py`，结果`artifacts/e1a933_review/vision_n03_analytic/result.json`。它达到测试单图准确率.98145，E-quartet J=.71698（114/159），原子同时正确128/159，CCM=114/128；636四态图中2个拟合失败均保留。该朴素像素算法已是强对照，后续不得通过忽略它来声称神经方法优越。

GPU入口`vision_n03_run.py`，独立bundle`artifacts/e1a933_review/vision_n03_bundle.tar.gz`；复用已验证bank/缓存，不覆盖旧21臂。完成判据receipt.status=MATRIX_COMPLETE且completed包含6臂，paired_analysis比较geometry对generic、旧BCE、旧matched。CPU两臂短训练只是工程烟测，不能作为GPU或科学完成。
