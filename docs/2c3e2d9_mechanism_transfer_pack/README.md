# 2c3e2d9 机制与迁移方案包

阅读顺序：00评估 → 01纠错 → 02总控 → 03/04两条主线 → 05条件备选 → 06稿件/GPU。

本包新增了一个可直接验证的T1排序表示碰撞；其价值是提供可证伪的机制入口，不是宣布旧随机测试集的所有错误已经被解释。

`checks/`中的12项微型测试已实际运行，见回执。它们测试本包转录的公式及参照实现，不是原仓库训练/实证结果复现。`repo_contract_audit.py`可在实际checkout上抽取并运行当前函数，且不导入仓库顶层训练入口；其JSON会报告当前合同失败，非修改原文件。

运行最小测试：
```bash
python -m unittest discover -s checks -p 'test_contracts.py' -v
```
在用户实际仓库核查：
```bash
python checks/repo_contract_audit.py --repo /path/to/iclr2027 --out live_contract_audit.json
```

文件不含数据、模型权重、SSH凭据或字体。没有在用户2080Ti实例上执行任何操作。新GPU工作由agent使用已授权实例完成。
