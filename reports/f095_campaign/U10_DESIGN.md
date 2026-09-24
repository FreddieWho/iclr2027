# U10｜设计（三角含＋线段圆盘，2026-09-23）

## 目标选择（已检索：无既有 circle/flat/triangle 任务，全仓库字面命中仅 flatten 类噪音）
- **T1 点-三角形包含**：4 点（V1,V2,V3 顶点＋Q 查询点），raw 输入 8 维（与线段题同格式，架构代码复用）。oracle：重心坐标符号测试。**对称群：S3(顶点)×{Q固定}，6 元；Q 不得与顶点交换**（不是 S4，不照搬）。
- **T2 线段-圆盘相交**：A,B 端点＋C 盘心，半径 R0 为任务常数（记录值）；raw 输入 6 维。对称群：{id, A↔B}，2 元。structured：3 成对距离。typedT：phi(A)+phi(B) 汇合、盘心独立分支。
- structured（sixdist）在两任务上都是"全部点对距离"（T1 六个、T2 三个），与线段题同构造原则；typedT 为任务适配的新结构（顶点汇合/Q独立；端点汇合/盘心独立）。

## E 挖掘（同 E 逻辑，新 oracle）
- 单点小位移编辑对：各自保持标签、联合翻转（oracle 搜索）。先跑可行性探针：E-quartets 数量不足则判该任务 INCONCLUSIVE，不硬凑。
- 训练只用单编辑，绝不训练联合态；static-dev 选参；target-dev 只做实现/收敛调整。

## 矩阵（3 架构 × 3 监督 × 3 种子 × 2 任务 = 54 训，CPU 分钟级）
- 架构：raw / sixdist / typedT。**砍 concat**：源任务两次 verdict 稳定（concat≈raw），迁移问题只剩几何-vs-对称，省 1/4 算力（记录理由）。
- 监督：clean / flip全量 / f25（嵌套固定，同 U07 口径）。
- 配方：全批量 300ep / Adam 1e-2 / BCE / lam=1.0，与源任务逐项相同；新任务 margin/幅度参数单列。

## 评价
- dev J 主判据＋ S 分解（U03 证书复用）；R_full/M（B110 口径沿用）；最简单强 baseline（clean）＋同数据量对照必报。
- 报告源预测命中/失败/未决（对账 U10_SOURCE_PREDICTION.md），不要求绝对阈值。
