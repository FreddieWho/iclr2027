# 路线3：source 强基线

## 命令与合同

```bash
python experiments/e832_focus/route3_baselines/strong_baselines.py --out artifacts/e832_focus/route3
```

代码直接复用路线1的新 source parent/E 生成器；没有读旧 source E bank、dev512 或任何封存池。source 测试 parent seed 832503、编辑 seed 832504；E=230，eligible parent=65（2026-09-25 审查修正原误写的 142）（见 `artifacts/e832_focus/route3/results.json`）。三 seed 11/23/47，single-flip，CPU 4 threads，100 epochs。

## 结果

解析 segment-intersection baseline 的 A/B/AB 与 J3 都为 1.0（E=230）。这是廉价解析器解决该 source 任务的结果，不是学习模型。

三个学习 seed 的 J3：capacity-matched raw 为 0.026/0.022/0.013；合法 segment DeepSets/relational 为 0.022/0.017/0.052；G8 群平均 raw 为 0.017/0.009/0.000。参数、训练状态前向和推理 MAC 在 JSON 逐行记录。

没有运行错误对称性的 S4 all-point sum，并明确标为不是有效 relational baseline。source 是坐标任务，不含图像，因此“经典图像 parser”对本路线为 NOT_APPLICABLE；Route 2 才拥有可见颜色 parser/renderer 接口。

## 结论边界

本路线最可靠的结果是廉价解析器已经解决 source E 银行。因此 source 的学习式排序/交互差异必须缩窄为“解析器之外的学习诊断”，不能继续写成任务需要复杂关系模型，也不能拿解析器 J3=1 与 Route 2 未运行的视觉数字比较。

允许：解析 baseline=E 上 1.0；学习基线的带分母描述。禁止：把 group average 说成完整不变表示、把 raw all-point sum 冒充关系模型、把 source 解析器结果外推为视觉机制。

## 三轮证据驱动尝试（2026-09-25，不制造阳性）

脚本：`experiments/e832_focus/route3_baselines/route3_rounds.py`（新增文件，不碰 `results.json`）。输出：`artifacts/e832_focus/route3/round1_optim.json`、`round2_repr.json`、`round3_freshbank.json`。选型只看 dev BCE，不看测试 J；封存池未读。

R1 优化诊断（同银行 E=230/65、同臂 capacity/relational、epochs {100,300} × lr {1e-3,3e-4}）：300 轮把训练 BCE 从约 0.40 压到约 0.18–0.29、训练准确率提到约 0.85–0.90，但 J3 仍在 0.004–0.065；dev BCE 反而恶化（过拟合训练分布）。结论：近零 J3 不是简单欠拟合，是结构/表示问题。

R2 表示诊断（同银行、300 轮、同一 MLP 族、只换输入）：raw J3≈0.01–0.03，six≈0.14–0.17，rich_unsorted≈0.14–0.15，rich_sorted≈0.13–0.23，orbit_canonical（G8 轨道字典序代表）≈0.30–0.37。结论：capacity/relational 臂的近零部分是输入表示问题——规范表示把 J3 从 ~0.02 抬到 ~0.34，但仍远低于解析器的 1.0。

R3 新银行确认（预先指定：解析器 + rich_sorted + orbit_canonical；新 parent seeds 833501–833504，E=171/56）：解析器 J3=1.0 依旧；rich_sorted≈0.21–0.24，orbit_canonical≈0.24–0.29。结论：parser-bounded 间隔在独立 parent 上成立；orbit ≥ sorted 的方向复现但差距收窄（s47 基本打平），不写成稳定超越。

三轮总结论：解析器解决 source E（两次独立银行均为 1.0）；学习臂的差异是表示诊断，最高只到 ~0.35，任务仍需要解析器级别的几何。论文口径不变（parser-bounded diagnostic），不新增主文主张。
