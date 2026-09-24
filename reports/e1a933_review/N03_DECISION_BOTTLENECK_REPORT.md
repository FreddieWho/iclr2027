# N03 决策瓶颈结果

六臂GPU训练及完整产物已回收校验。几何监督相对同容量通用瓶颈三seed的J均提升，但对强视觉基线没有稳定收益，不能确认为几何决策机制或主方法。

| 瓶颈 | seed | J | dev matched MSE | 真几何替换同head J（诊断） |
|---|---:|---:|---:|---:|
| geometry_bottleneck | 803 | 0.6981 | 0.9326 | 0.0000 |
| geometry_bottleneck | 805 | 0.7044 | 2.6150 | 0.0000 |
| geometry_bottleneck | 806 | 0.7044 | 0.5998 | 0.0000 |
| generic_bottleneck | 803 | 0.6792 | 79.7163 | 0.0000 |
| generic_bottleneck | 805 | 0.6226 | 51.9862 | 0.0000 |
| generic_bottleneck | 806 | 0.6855 | 45.7357 | 0.0189 |

## 配对解释

- geometry_bottleneck minus generic_bottleneck: 均值 +3.98pp；803: +1.89pp, parent CI [-2.33, +6.83]; 805: +8.18pp, parent CI [+0.69, +16.05]; 806: +1.89pp, parent CI [-4.67, +8.51]。
- geometry_bottleneck minus pretrained_flip: 均值 +1.89pp；803: +5.66pp, parent CI [+0.00, +11.93]; 805: +0.63pp, parent CI [-3.27, +4.97]; 806: -0.63pp, parent CI [-4.29, +2.86]。
- geometry_bottleneck minus matched: 均值 +1.05pp；803: +0.00pp, parent CI [-4.30, +4.35]; 805: +3.14pp, parent CI [-1.20, +7.74]; 806: +0.00pp, parent CI [-6.40, +6.94]。

几何瓶颈的dev MSE为0.60–2.62，明显高于N01 matched的0.118–0.297，因此B/C不是“几何误差相当”的干预；不能将差异归因于信息被迫进入决策。三个几何瓶颈将真几何代入冻结head时J均为0，这表明head没有在真几何输入下形成正确关系计算；替换也改变了输入分布，不能据此将错误完全分解成感知误差。

经典颜色分割/PCA像素基线J=114/159=0.7170，高于三个几何瓶颈的描述性点估计；该比较不作独立显著性宣告。所有实验使用159 quartet/74 parent的已暴露bank，不是新确认。

## 噪声诊断

Post-training descriptive diagnostic on 707 saved dev latents; isotropic Gaussian noise in standardized 10d relation space, 16 fixed draws. No image perturbation or threshold tuning; no physical-validity claim.

| 瓶颈 | seed | sigma=.1决策变化率 | sigma=1决策变化率 |
|---|---:|---:|---:|
| geometry_bottleneck | 803 | 0.0000 | 0.0157 |
| geometry_bottleneck | 805 | 0.0000 | 0.0019 |
| geometry_bottleneck | 806 | 0.0000 | 0.0458 |
| generic_bottleneck | 803 | 0.0000 | 0.0004 |
| generic_bottleneck | 805 | 0.0000 | 0.0011 |
| generic_bottleneck | 806 | 0.0003 | 0.0029 |

不再据此重启架构搜索；本瓶颈未取得可靠任务增益，不触发N03向足球迁移。O05独立的部分tracking学习比较不依赖此路线成功。

逐臂原始结果/完整修复、迁移及111退化分母：`artifacts/e1a933_review/gpu_finish_20260924/n03/extracted/n03_results/paired_analysis.json`。本地解释与噪声结果：`artifacts/e1a933_review/n03_interpretation/summary.json`。
