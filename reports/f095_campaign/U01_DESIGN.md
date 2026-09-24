# U01｜typed-pair 精确不变架构设计（待执行）

- 候选强 claim："任务兼容的结构表示改变单编辑监督能否转化为完整组合正确性"（手工距离→可训练结构干预）。
- 正确架构（保留线段归属，复用 D02 群库测试）：
  `h_AB=phi(A)+phi(B); h_CD=phi(C)+phi(D); z=psi(h_AB)+psi(h_CD); logit=rho(z)`，
  phi/psi 为非线性网络（纯线性 phi 相加只剩中点，丢方向/长度——显式禁）。对端点内交换＋整段交换精确不变；旋转/平移不变另列（相对坐标/EGNN 为独立干预，不一次全改）。
- 对照矩阵（同训练父场景、同 flip 样本、相近可训参数量 ~7k）：typed-pair×{clean,flip} 主矩阵；raw MLP、sixdist MLP（6 成对距离，无半径，刚体不变但列有角色）、D02 群平均 raw；容量匹配"独立编码拼接"模型（区分"多一层 MLP"与"共享/汇聚"）。无类型四点 DeepSets 仅作信息丢失反例禁作主基线。
- 检查：8 重标号输出差≤容差；改 AB/CD 配对改变 oracle 与模型表示（配对信息保留测试）；static-dev 停止规则沿 D01；3 seed 起步。
- 评价：identity/orbit 均值/worst-orbit/all-orbit J、U03 S/J* 分解、R_full/M、推理成本单列。
- 分支：不变成立但 J 无提升→"命名不变性不足"；typed+flip 超群平均/六距离→结构×监督互补；六距离仍最优→承认手工表示优势。
- 输出：`u01_typed_invariant.py`＋群不变量测试＋四臂逐 parent 结果；主文只给最小结构公式＋完整对照。
