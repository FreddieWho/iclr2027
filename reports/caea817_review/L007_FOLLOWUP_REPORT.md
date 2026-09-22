# L007 后续诊断 — 掠射交互与 kink 密度（双阴性）

动机：独立文献线提出 5 个零训练候选（research.md 存档于子代理产物），
排序为分歧分解 → 掠射角×带宽 → kink 密度。分歧分解已由训练线覆盖
（8 新种子复训＋ensemble，运行中）；本报告执行第 2、3 项。
全部冻结模型、同一批 200 路径、同一转折定义；判据全部预注册。

## G：掠射角 × 过渡带宽（死）

脚本 `experiments/last15h/l007_grazing.py`；输出
`artifacts/next_novelty/l007_grazing/`。N02 只测了无条件梯度方向，
本交互项（w/a，其中 a=|g·ê|/‖g‖ 为路径切向与边界法向夹角余弦，
w 为 |logit| −k→+k 的 Δt）确与其正交——但结果阴性：

| seed | Spearman(w/a, \|terr\|) | w 单独 | a 单独 |
|---|---|---|---|
| 11 | +0.12 | +0.18 | +0.15 |
| 23 | −0.06 | −0.04 | +0.06 |
| 47 | −0.10 | −0.10 | +0.04 |

符号不一致、量级全 <0.2，判据（≥0.3 同向 3/3）未触发。
结论：同等法向扰动不按 1/sinθ 放大为位置误差；边界朝向无判别
（与 N02 一致，且关闭了 N02 没测的交互项版本）。

## K：kink 密度（死，谱偏置解释关闭）

脚本 `experiments/last15h/l007_kink.py`（forward hook 取双层 ReLU 激活模式，
129 点密网格）；输出 `artifacts/next_novelty/l007_kink/`。

| seed | kink-dist (+) | kink-count (−) | t_theta 落 kink 率 |
|---|---|---|---|
| 11 | +0.15 | −0.22 | 0.20 |
| 23 | +0.06 | −0.29 | 0.15 |
| 47 | +0.17 | −0.28 | 0.16 |

符号全在预测方向，但量级全在 0.3 线以下（s23 的 −0.29 也不跨线，
不追——多重比较下追线即 fishing）。miss 路径 kink 略少（3.1–4.1 vs
4.3–4.4），方向一致但弱。结论：分段线性断点 budget 不决定翻转位置
分辨率；谱偏置解释关闭。连带推论：S3 的 Fourier 特征重训失去动机
（它以 kink 阳性为前提），保持 park。

## P5：mode connectivity（inconclusive，1 个弱阳性实例）

脚本 `experiments/last15h/l007_connectivity.py`（11 个现有 flipmine 权重，
5 对 × 9 个 α，naive＋贪心激活匹配双跑）；输出
`artifacts/next_novelty/l007_conn/`。判据：对齐后相对 barrier <5% 且 |terr|
极差 >1σ_seed（0.0149），≥3/5 对=通过。

- 仅 r04b11-r04b23 对齐成功（barrier 0.9%），|terr| 极差 0.053（≈3.5σ，弱阳性实例）。
- 其余 4 对贪心对齐后 barrier 仍巨大（0.56–1626），检验作废——
  这不证明"精度跟随损失"，只证明该检验在此无判决力（matcher 弱 vs 真不可连，无法区分）。
- 附带发现：跨 era（r04b vs 2001+种子）几乎不可连，配方逐字相同——
  不同 init 落入的 basin 差异比预期大；与种子方差"纯 init 噪声"结论一致。
- 判决：inconclusive。exact-Hungarian 跟进性价比低（最好情况也只是第二个弱实例），park。

## 累计排除清单（转折位置精度）

监督侧：斜率下限、平坦性、mixup、稠密标签、SDF、tangent、coupled-margin、
global-grad、BAN（E7 v3，12 臂 60 训练）。
几何侧：输入几何、隐空间轨迹几何、输出斜率/曲率（除同构邻接量）、
掠射角×带宽交互、kink 密度（本报告＋L007 诊断）。
待定：跨种子分歧/ensemble（训练线运行中）；离路褶皱（park，成本高优先级低）；
max-margin 隐式偏置（park，需重训）。
