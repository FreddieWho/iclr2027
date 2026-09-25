# 路线3：source 强基线

## 命令与合同

```bash
python experiments/e832_focus/route3_baselines/strong_baselines.py --out artifacts/e832_focus/route3
```

代码直接复用路线1的新 source parent/E 生成器；没有读旧 source E bank、dev512 或任何封存池。source 测试 parent seed 832503、编辑 seed 832504；E=230，parent=142（见 `artifacts/e832_focus/route3/results.json`）。三 seed 11/23/47，single-flip，CPU 4 threads，100 epochs。

## 结果

解析 segment-intersection baseline 的 A/B/AB 与 J3 都为 1.0（E=230）。这是廉价解析器解决该 source 任务的结果，不是学习模型。

三个学习 seed 的 J3：capacity-matched raw 为 0.026/0.022/0.013；合法 segment DeepSets/relational 为 0.022/0.017/0.052；G8 群平均 raw 为 0.017/0.009/0.000。参数、训练状态前向和推理 MAC 在 JSON 逐行记录。

没有运行错误对称性的 S4 all-point sum，并明确标为不是有效 relational baseline。source 是坐标任务，不含图像，因此“经典图像 parser”对本路线为 NOT_APPLICABLE；Route 2 才拥有可见颜色 parser/renderer 接口。

## 结论边界

本路线最可靠的结果是廉价解析器已经解决 source E 银行。因此 source 的学习式排序/交互差异必须缩窄为“解析器之外的学习诊断”，不能继续写成任务需要复杂关系模型，也不能拿解析器 J3=1 与 Route 2 未运行的视觉数字比较。

允许：解析 baseline=E 上 1.0；学习基线的带分母描述。禁止：把 group average 说成完整不变表示、把 raw all-point sum 冒充关系模型、把 source 解析器结果外推为视觉机制。
