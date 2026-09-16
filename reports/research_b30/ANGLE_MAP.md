# B30 角度地图（无重叠、无黑洞；已排除旧简报覆盖区）

旧简报已覆盖（禁重复开题）：A谱近似 / B塌缩 / C不确定性 / D软共享 / E平衡两阶段 /
F成功先例 / G BGRL / H Slepian-μ / I小模型 / 评审期4份（landscape/taste/logistics/positioning）

## W1 机制与科学性（10题：M1–M10）
- M1 层级探针：encoder各层表征的线性可分性剖面（机制深度证据）
- M2 因果形式化：干预-响应的因果estimand是什么，能否写成do-calculus
- M3 pair/干预分裂现象学：第3例了（v5-s23/v6-s47/JGCL），何时ranking与响应分离
- M4 margin变薄机制论：aux压力下间隔收缩的理论解释
- M5 种子方差结构：哪些指标seed-稳、哪些靠运气（方差分解设计）
- M6 时间维度：单帧局限的代价，运动连贯性作为缺失信号
- M7 centroid -34% 为何独强：有无几何定理支撑（headline的理论根）
- M8 耦合增长函数形：support scaling的标度律拟合与外推
- M9 证伪设计：什么实验结果会杀死geometry→response claim（Popperian对照）
- M10 跨provider机制复现：T5R6机制数作为机制复制证据的强度评估

## W2 架构算法与新颖性（10题：A1–A10）
- A1 EGNN/SetTransformer泛化性：文献上结果该不该迁移（Limitations核心句）
- A2 读出头设计：cosine之外有无更好读出（可学习vs解析）
- A3 模式嵌入蒸馏：能否把mode embedding蒸成闭式（可解释性+）
- A4 生成式对应物：formation流形上的diffusion/flow（时髦缝合候选）
- A5 内建结构的等变架构：显式centroid/mode分离的E(2)-GNN设计
- A6 超越team-mean的分层池化：多尺度池化的文献与收益预估
- A7 测试时自适应：用routing头做TTA的可行性
- A8 margin理论：triplet vs InfoNCE在此任务上的理论分野（接GOAL-R2）
- A9 容量标度律：encoder放大是否改变机制（pilot设计，不执行）
- A10 跨运动迁移：篮球数据的文献＋数据可行性（parked项的解冻条件）

## W3 评估故事趣味wild（10题：S1–S10）
- S1 大数字狩猎：哪个指标还有headroom（pair/margin/centroid/干预）
- S2 图表设计：评审冲击力最大的Figure/Table组合
- S3 标题摘要framing：新颖性定位的三种写法
- S4 related-work柔道：把Gruver/Geiger/TacticAI比较写成优势
- S5 rebuttal预演：十大最可能质疑＋现成答案
- S6 wild：书法同构复活（outlook种子，用户曾问）
- S7 wild：实时应用故事（教练工具？broader impact）
- S8 wild：对抗鲁棒性framing（reallocation即攻击）
- S9 wild：神经科学连接（grid/place细胞的位置-上下文分离）
- S10 wild：benchmark提案（多智能体表征探针基准，社区贡献角度）

## 黑洞检查
- 训练动力学追踪（epoch内机制形成过程）：未覆盖→并入M1子问题
- 数据本身（IDSSE偏差/标注噪声）：未覆盖→并入M10子问题
- 计算成本叙事（小算力可复现）：未覆盖→并入S2子问题
