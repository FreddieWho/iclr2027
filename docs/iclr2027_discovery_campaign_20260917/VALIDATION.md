# 交付前工程验证

日期：2026-09-17。审阅仓库基准：`f86dbdf6107bad3d156b250a3c23bd5dcd8bb9d2`。

## 实际执行并通过
- `python -m unittest discover -s .../tests -v`：24/24通过。包括成对扰动的边际/能量/平衡、完整对过滤、局部二次恒等式、固定基等功率、分布权重、源端贪心覆盖、PSD度量、精确碰撞不可恢复、关系oracle、场景生成、任务DAG、示例空值及只读inventory。
- 全部Python文件编译/语法解析；全部JSON解析；结果JSON Schema本身及空结果模板验证。
- `scripts/smoke.py`独立运行：随机矩阵/构造身份检查通过。副本在`examples/engineering_smoke.json`，标记`engineering_only`，不是实测科学效果。
- `scripts/generate_relation_scenes.py --n 64 --seed 314 ...`：生成64个带解析关系标签的坐标场景和metadata。没有渲染、没有神经模型评价。
- `scripts/bootstrap_run.py`在临时本地目录执行：正确记录没有git仓库/没有项目模型的状态，未改输入。此项只验证CLI，不是对用户真实运行环境资产的完整盘点。
- 打包后执行ZIP完整性测试；包中逐文件SHA-256见`SHA256SUMS.txt`。

## 没有执行的部分
没有克隆并重跑整个项目，没有下载DINOv2权重，没有训练六条新路线，没有验证NRI模拟器的现代兼容性，也没有获得新的项目效应量或发表胜率。

包内已实现的是可复用数学/数据构造内核和工作说明；真实模型adapter、渲染器、模拟器移植、训练及评价runner由施工agent依照对应路线卡完成。

以上检查只保证这些起点能够运行，不保证假设成立。本轮所有候选headline均待新实验填入；旧数据不被重算或冒充新发现。
