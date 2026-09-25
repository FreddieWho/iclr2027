# 路线4：独立稳健性（新 parent + 坐标观测变化）

## 命令与分母

```bash
python experiments/e832_focus/route4_robustness/robustness.py --out artifacts/e832_focus/route4
```

结果：`artifacts/e832_focus/route4/results.json`。新 source train/dev/test/edit seeds 为 832701/832702/832705/832706，与路线1不同；E=207，parent 分母固定，不按结果筛选。训练 seed 11/23/47，single-flip，60 epochs，CPU 4 threads。

条件在模型和 E 分母锁定后统一施加：canonical、平移 `(+0.071,-0.043)`、绕原点旋转 7°。平移/旋转是 source oracle 下的严格坐标变换，标签和 parent 分母保持不变。每行有 J3、J4(missing)、A/B/AB、atomic_joint、参数、训练前向、推理 MAC。

## 观察

`rich_sorted` 三 seed 的 J3 为 0.014/0.068/0.092；`rich_unsorted` 为 0.005/0.014/0.010；这三个刚体条件内每 seed 数值相同，符合 oracle 几何不变性。raw/additive/repaired 在本次低预算下接近 0。排序差在三个条件都同号，但这只是新 parent 上的坐标稳健性描述，60 epoch 与小 E 不足以升级为强主张。

非刚体 `noise_sigma=0.003` 没有混入上述分母，状态明确为 NOT_RUN：需先冻结并审计新 E 的 label-preserve/sum-flip 合同，不能看结果后换标签。视觉 renderer 条件也明确分开：Route 2 为 BLOCKED_GPU，Route 4 没有视觉性能数字。

## 结论边界

允许：新 parent、三个刚体条件下 sorted/unsorted 的描述性比较。禁止：把刚体不变性说成 renderer 稳健性、把未运行的 noise/visual 说成阴性确认、把 J3 接近零解释成源任务已解决（路线3解析器才是该 E 上的解决基线）。
