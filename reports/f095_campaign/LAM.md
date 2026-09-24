# flip-BCE lam 伸缩（N×w64，3 lam×3 种子，判决 NEGATIVE-for-cause）

- 问题：6/12 flip 格塌缩（恒 loss ~1.3，J=S=0）是否 lam=1.0 不当？
- 运行合同：raw w64f32×N，lam {0.25, 0.5, 2.0} × seeds × flipmine（9 训，全收敛）；dev512 同 quartet 评价。输出 `LAM/LAM_EVAL.json`。
- 结果表（J；锚点 lam1.0：.068/.059/.076）：

| lam | s11 | s23 | s47 |
|---|---|---|---|
| 0.25 | .042 | .080 | .089 |
| 0.5 | .068 | .084 | .059 |
| 2.0 | .080 | .106 | .063 |

S 全在 .63–.73（锚 flip .67–.73 同级），J 全 .04–.11——lam 响应平坦，种子噪声主导。

- 主 claim：lam 不是塌缩原因（N 尺度 9/9 收敛；塌缩集于宽 ≥256/数据 ≥4N 的随机子集，属 scale×basin 随机性）。可行动发现是 D02-aug 的：g8 增广治愈 6/6（含 s47）——稳定性解在输入多样性，不在 lam。
- 处置：大尺度 flip 训练默认加 g8 增广（已验证）；lam 固定 1.0 不调。
- 判决：NEGATIVE（lam 致因）/ SUPPORTED（塌缩画像：scale 相关、seed 特异、可治愈）。
