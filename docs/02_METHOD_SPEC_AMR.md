# 方法规格：AMR — Action-Mode Routing

> 一句话摘要：AMR 不把某种变换整体宣布为“该忽略”或“该保留”，而是在关系图的作用模式频段上，按任务学习不变、等变与上下文保留的分配。

## 1. 方法定位

本项目希望同时发展：

1. 一个新的测量对象：Action-Mode Spectrum；
2. 一个机制发现：谱响应为何出现错配；
3. 一个可独立成立的方法贡献。

因此，**Centered Action Prediction（CAP）只能作为最小 baseline**，不能成为最终方法。主方法是 AMR，其新意不在“预测一个相对位移”，而在：

- 将不变/等变约束放入构件关系图的频段坐标；
- 让不同任务学习不同的模式路由；
- 保留独立的上下文通道，不把全局信息永久删除；
- 通过谱白化采样，避免训练信号只覆盖容易的模式；
- 在显式图和隐式 token 图上使用同一原则。

### 1.1 P2 后的方法边界

P2 已冻结为 \`mixed_or_graph_specific\` 的 representation-response 结果。AMR 不从 P2 的方向或响应大小后验调门，也不把 role 当作跨域方法的必要输入。P3 先判断 support-conditioned local geometry 是否能预测、定位并干预该异质性；只有 P3 gate 满足后，才允许实现 M1。因果开关未成立时，AMR 保持 \`NOT_STARTED\`，不以方法训练替代机制证据。

## 2. 方法输入与接口

### 2.1 显式结构输入

体育样本：

\[
X\in\mathbb R^{n\times d_x},\quad A\in\mathbb R^{n\times n},\quad m\in\{0,1\}^n.
\]

- \(X\)：坐标、速度、角色或外观特征；
- \(A\)：关系图；
- \(m\)：有效节点掩码。

### 2.2 隐式像素输入

书法样本：

\[
I\in\mathbb R^{H\times W\times C}.
\]

图像编码器输出 token：

\[
T=[t_1,\ldots,t_p]\in\mathbb R^{p\times d}.
\]

图来源分三级：

1. **oracle/矢量图**：Make Me a Hanzi 的 stroke/component graph，用于诊断和受控训练；
2. **pseudo graph**：CCSE 或几何骨架产生的近似笔画图；
3. **latent token graph**：由 token affinity 和二维位置先验构造，用于 AMR-I 扩展。

论文需要明确区分这三种信息条件，但具体采用哪一种可随数据和结果调整。

## 3. 关系图与谱坐标

给定对称归一化图拉普拉斯：

\[
L=I-D^{-1/2}AD^{-1/2}.
\]

小图诊断可以精确求：

\[
L=U\Lambda U^\top.
\]

但方法不应依赖逐样本 eigenvector 的稳定对齐。AMR 使用 \(B\) 个谱频段滤波器：

\[
P_b(L)=\sum_{r=0}^{R} a_{b,r}T_r(\tilde L),
\]

其中 \(T_r\) 是 Chebyshev 多项式，\(P_b\) 近似低通、中频和高频带通滤波器。

建议默认：

- \(B=6\) 或 \(8\)；
- 多项式阶数 \(R=3\) 或 \(5\)；
- 频段按归一化谱区间均匀或等特征值质量切分。

优势：

- 置换等变；
- 不受 eigenvector 符号翻转影响；
- 可用于不同节点数；
- 计算量低。

## 4. 表征结构

AMR 输出两个互补通道：

\[
z=(z_{ctx},z_{mode}).
\]

### 4.1 Context channel

\[
z_{ctx}=g_{ctx}(\operatorname{Pool}(H)).
\]

保存：

- 球场绝对位置；
- 页面姿态；
- 场景上下文；
- 其他通用 backbone 不应被迫删除的信息。

### 4.2 Mode channel

先得到节点/token 表征：

\[
H=E_\theta(X,A) \quad\text{或}\quad H=E_\theta(I).
\]

每个谱频段得到：

\[
r_b=\operatorname{Pool}_b\left(P_b(L)H\right).
\]

然后：

\[
z_{mode}=[r_1;\ldots;r_B].
\]

可选关系增强：

\[
r_b^{rel}=\operatorname{Pool}_{(i,j)\in E}
\psi_b(h_i,h_j,h_i-h_j),
\]

用来检验 unary aggregation 是否是根因。

## 4.3 Normalized-embedding local geometry

设任一可比较层的输出为 \(z_f(X)\)，先定义

\[
\hat z_f(X)=\frac{z_f(X)}{\|z_f(X)\|_2},
\qquad
J_f(X)=\frac{\partial\hat z_f(X)}{\partial\operatorname{vec}(X)},
\qquad G_f(X)=J_f(X)^\top J_f(X).
\]

P3 的主预测量不是直接拟合 P2 response，而是

\[
q_f(X,\delta)=\frac12\|J_f(X)\delta\|_2^2
 =\frac12\delta^\top G_f(X)\delta.
\]

在小扰动、无 dropout 的局部近似下，\(q_f\) 预测归一化表示余弦距离。若输入有 \(N\) 个二维节点，\(G_f\) 由 \(2\times2\) block 组成：

\[
\delta^\top G_f\delta
=\sum_i\delta_i^\top G_{ii}\delta_i
+2\sum_{i<j}\delta_i^\top G_{ij}\delta_j.
\]

其中 \(G_{ii}\) 是节点自身敏感度，\(G_{ij}\) 是跨节点耦合。P3 必须同时报告 full、diagonal-only 和 off-diagonal contribution，并按 team 内外、support 内外、graph edge/non-edge 做描述性分解。若 off-diagonal 没有超出 diagonal 的解释力，方法叙述收缩为节点敏感度各向异性。

工程上优先使用 JVP 计算 \(J_f\delta\)，只在小规模诊断或内存允许时形成 full \(G_f\)；不得为了存储完整 Jacobian 改变模型或 checkpoint。必须验证 autograd JVP 与中心有限差分，并检查二阶近似在小 \(\varepsilon\) 下按预期收敛。

## 5. 干预与谱白化采样

### 5.1 等能量要求

比较干预时优先采用等能量条件，并报告实际偏差：

\[
\|\delta\|_F=\varepsilon.
\]

若是像素域，额外报告：

- 像素 \(L_2\)；
- LPIPS 或冻结感知距离；
- 目标模式激发纯度。

### 5.2 谱白化

自然增强分布常在某些模式能量很高、另一些模式几乎没有训练信号。AMR 训练时先采样频段 \(b\)，再采样该频段内的扰动并归一化：

\[
\delta_b
=
\varepsilon
\frac{P_b(L)\xi}{\|P_b(L)\xi\|_F},
\quad \xi\sim\mathcal N(0,I).
\]

使各频段获得近似均衡的干预次数和能量。

对于显式图，直接在坐标上施加；对于矢量笔画，作用于 stroke median/path 后重新渲染。

## 6. 任务条件的模式路由

### 6.1 核心思想

对于任务 \(\tau\)，每个频段有门：

\[
\alpha_{\tau,b}\in[0,1].
\]

- \(\alpha\to1\)：该模式在此任务下应被 invariant head 忽略；
- \(\alpha\to0\)：该模式应保持等变/可恢复；
- 中间值：软路由或不确定任务边界。

门由：

\[
\alpha_{\tau}=\sigma(g_\omega(e_\tau,s_x))
\]

产生，其中 \(e_\tau\) 是任务 embedding，\(s_x\) 是样本级统计。第一版可只使用任务 embedding；不要一开始引入复杂超网络。

P3 将上式升级为固定、可审计的 factorized route：频段 × 支持尺度/关系描述共同决定路由条件。支持/关系描述应来自 response-blind 的图和干预元数据，例如 support fraction、induced density、cut/edge summary 或 P3 证明的机制量；不能从 P2 response 学门：

\[
\alpha_{\tau,b,S}=\sigma(g_\omega(e_\tau,s_S,b)),
\]

其中 \(s_S\) 是支持/关系统计。M1 不学习该门，而固定使用 P3 证据支持的频段 × 支持/关系路由；M2 才允许在冻结机制和任务定义后从任务 embedding 与样本关系统计学习路由。

静态足球 role 只能作为诊断分层或 intervention-supervised 信息，不能成为推理时通用 AMR 的硬编码真值输入。若某任务确实有已知 role 标签，必须单独报告 role-oracle 与无 role 版本，不能把前者写成跨域方法必要条件。

### 6.2 为什么采用 task-conditioned

同一共同模式：

- 对 formation identity 可能是 nuisance；
- 对 high/low block 或场区部署可能是 signal；
- 对书法结构匹配，页面平移通常是 nuisance；
- 对版面章法任务，整体位置可能是 signal。

因此，固定删除 DC 分量并不适合通用表征。

## 7. 损失函数

设干预前后模式表征差为：

\[
\Delta r_b=r_b(X\oplus\delta_b)-r_b(X),
\]

真实谱系数或频段目标为：

\[
c_b=P_b(L)\delta.
\]

### 7.1 Invariance branch

\[
\mathcal L_{inv}^{(b)}=\|\Delta r_b\|_2^2.
\]

### 7.2 Equivariance / recoverability branch

使用严格限制的线性或浅层头：

\[
\hat c_b=W_b\Delta r_b,
\qquad
\mathcal L_{eqv}^{(b)}=\|\hat c_b-c_b\|_2^2.
\]

头不能无限复杂，否则只能证明信息仍以不可访问方式存在。

### 7.3 Mode routing objective

\[
\mathcal L_{route}
=
\sum_{b=1}^{B}
\left[
\alpha_{\tau,b}\mathcal L_{inv}^{(b)}
+
(1-\alpha_{\tau,b})\mathcal L_{eqv}^{(b)}
\right].
\]

### 7.4 Context preservation

对全局作用参数或绝对上下文，保留可访问性：

\[
\mathcal L_{ctx}=\ell(q_{ctx}(z_{ctx}),y_{ctx}).
\]

注意：这不是要求每个任务都使用上下文，而是避免 backbone 永久丢失它。

### 7.5 结构任务损失

\[
\mathcal L_{task}=\ell(q_\tau(z_{ctx},z_{mode}),y_\tau).
\]

任务包括 formation retrieval、phase/deployment、结构匹配和扰动定位。

### 7.6 防塌缩与解耦

建议最低限度使用：

\[
\mathcal L_{var}=\sum_b\max(0,\gamma-\operatorname{Std}(r_b)),
\]

\[
\mathcal L_{orth}=\left\|\operatorname{Cov}(z_{ctx},z_{mode})\right\|_F^2.
\]

总损失：

\[
\mathcal L
=
\mathcal L_{task}
+\lambda_r\mathcal L_{route}
+\lambda_c\mathcal L_{ctx}
+\lambda_v\mathcal L_{var}
+\lambda_o\mathcal L_{orth}
+\lambda_g\mathcal R(\alpha).
\]

门正则 \(\mathcal R(\alpha)\) 可选：

- 平滑：相邻频段门不应剧烈抖动；
- 稀疏/低熵：避免全为 0.5；
- 任务差异：不同任务的门不得被强制相同。

## 8. 方法层级

### M0：CAP baseline

预测去均值或图差分干预：

\[
\mathcal L_{CAP}=\|q(f(x'),f(x))-B\delta\|^2.
\]

用途：验证“记住差分、共同作用自然进入零空间”的最低原则。

### M1：AMR-Fixed — 优先实现

- 只有在 P3-T1–T5 的机制 gate 至少条件支持后实现；
- 保留谱频段、谱白化采样和显式 context/mode 双通道；
- 使用由 P3 证据支持的固定“频段 × 支持/关系”路由，不把所有中频预设为应修复对象；
- 固定的 invariant/equivariant/recoverability 分配必须在查看 AMR 结果前写入配置；
- 不使用静态 role 作为通用推理输入，并与 CAP、centering、canonicalization、relational pooling 做 matched-capacity 比较。

M1 的最低目标是证明固定机制路由改善 robustness–structure Pareto；它不是把所有 SCG 推成正值。

### M2：AMR-Learned — 目标主方法

- 只有 M1 和 P3 任务联系成立后才启动；
- 在固定频段与支持/关系特征接口上，从任务 embedding 和样本关系统计学习 \(\alpha_{\tau,b,S}\)；
- 不用 P2 response 作为目标或阈值选择器；
- 以任务语义敏感度、结构可访问性和 context robustness 的 Pareto 改善评价，而不是以 response 变大评价。

### M3：AMR-Implicit — 条件性扩展

- 在图像 token 上学习稀疏 affinity graph；
- 使用 Chebyshev filter bank 路由；
- 图一致性和防退化正则；
- 不以无监督偏旁涌现作为主承诺。

只有 M1/M2 成功后才投入 M3。书法主线可先用矢量 stroke graph 或 pseudo graph。

## 9. 可选高风险扩展：学习关系图

某评审建议将“使敏感度算子对角化的图”作为正确关系图：

\[
\min_{L_\eta}
\operatorname{OffDiag}
\left(U_\eta^\top J_f^\top J_f U_\eta\right).
\]

该想法精彩，但退化风险高：模型可能学出平凡图。若探索，建议加入：

- 固定节点度或稀疏度预算；
- 谱熵下界；
- 连通性约束；
- 与体育真值战术单元的对齐验证。

它不进入最低施工路径，也不应阻塞 AMR-Fixed/Learned 的探索。

## 10. 与近邻方法的实质差异

### 10.1 相比普通 invariant/equivariant split

普通方法按层或输出头分 invariant/equivariant；AMR 按**关系图作用模式频段与任务**分配约束。

### 10.2 相比 PooDLe / SER

PooDLe 和 SER 保留 dense/token equivariance；AMR 研究的是构件联盟频谱，并让最终任务表示按模式路由，而不是仅在中间层保留空间响应。

### 10.3 相比 canonicalization

Canonicalization 选择一个规范姿态；AMR 不删除全局姿态，而是把它保存在 context channel，并使任务决定是否使用。

### 10.4 相比 CAP / transformation prediction

CAP 预测单个相对干预；AMR：

- 覆盖连续频谱；
- 平衡各频段训练信号；
- 学习 task-conditioned invariance/equivariance boundary；
- 保留上下文通道；
- 输出可解释的模式指纹。

### 10.5 相比 GNN oversmoothing 修复

AMR 不以提升节点区分度为最终目标，而以输入作用模式到表示响应的传递函数为目标；同频语义联盟对照用于证明其不只是频率低通。

### 10.6 相比普通 Jacobian regularization、relational pooling 与 graph spectral routing

- 普通 Jacobian regularization 通常约束输入敏感度的总量、平滑性或不变性；AMR 把归一化 embedding 的局部 metric 作为诊断和路由依据，并按任务、频段和 support/relationship 条件分配 invariant、equivariant 或 recoverability 目标。
- relational pooling 只说明关系项可被聚合；AMR 先用 \(G_{ii}/G_{ij}\) 和 layer-wise 证据判断关系信息在哪里形成、丢失或被放大，再选择是否使用关系池化，不能预先把关系池化写成机制结论。
- graph spectral routing 可能只按图频率分配通道；AMR 的必要条件是 support-conditioned/factorized routing，并保留独立 context channel，同时接受“只有 diagonal 或特定 architecture 有效”的收缩结果。

## 11. 机制型理论目标

优先证明或在线性设定说明：

### Proposition A：可分 unary pooling 的关系相位缺失

对旋转/周期变量，单构件 Fourier feature 为 \(e^{ik\theta_i}\)。共同旋转引入公共相位。若对每个构件独立消除相位再做加法聚合，相对相位也被丢失；而关系项：

\[
e^{ik\theta_i}\overline{e^{ik\theta_j}}
=e^{ik(\theta_i-\theta_j)}
\]

天然消除共同相位并保留相对关系。

这为跨构件交互的必要性提供机制解释。

### Proposition B：增强谱塑造敏感度谱

在小扰动和二阶近似下，alignment/invariance 目标包含：

\[
\sum_k \sigma_k^2 H_f(k)^2,
\]

其中 \(\sigma_k^2\) 是增强在模式 \(k\) 上的能量。由此推导可验证预测：提高某频段 invariance 增强会进一步压低该频段响应。

这只是局部线性机制，不需要包装成一般定理。

## 12. 实现建议

### 体育显式图

- Encoder：DeepSets、2 层 GAT/Graph Transformer 或小型 Set Transformer；
- hidden：128–256；
- mode bands：6；
- 节点：单队 10 名外场球员优先；
- 训练输入：单帧或 1 秒平均窗；
- 不需要视频模型。

### 书法图像

- 初期：冻结 DINOv2/CLIP/MAE 提取 token；只训练 AMR head；
- 方法训练：小型 ViT/ResNet 或冻结 backbone + LoRA/最后两层；
- 矢量干预使用 Make Me a Hanzi；
- 自然书法在受控结构流水线稳定后进入，并可与主线探索并行推进。

## 13. 方法成功标准

AMR-Learned 至少满足：

1. 在同等 global robustness 下，语义频段 \(H_f\) 或线性可访问性显著提高；
2. 优于 CAP、普通 relational pooling、centering/Procrustes 和 learned canonicalization；
3. 学到的门随任务变化，并符合可独立验证的任务语义；
4. 改善至少一个不是由干预标签直接构造的自然任务；
5. 体育和书法共享同一实现原则，而不是两个专用模型拼接。

若 M2 未超过 M1，仍可保留 AMR-Fixed 作为方法，并相应收缩“自动发现边界”的表述。

## 14. P3 实际 gate 状态（2026-09-01）

P3 的 local geometry prediction 和 layer/block anatomy 获得条件性支持，但唯一的 pooling-accessibility 开关只改变了表示 geometry/response：3 个 seed 的 heldout macro-F1 没有稳定改善，且 normalized context response 均上升。因此旧 task gate 的结论仍是 response shaping only，不实现或宣称 AMR-Fixed，不启动 M2，不扩大 causal matrix。2026-09-02 起，补丁包在同一 P3 内授权 T5R，以分离 context/intrinsic task 并建立正确的 candidate-lock 证据；T5R 没有结果前 AMR 仍保持 `NOT_STARTED`。

## 15. P3-T5R 前置模型：固定双通道（2026-09-02）

在 AMR-Fixed 之前，新增一个最小 protocol-corrective 检验，不把它称为 AMR：

```text
shared encoder E
raw positions:       X           -> Pool -> z_ctx  -> context head
globally centered:   X - mean(X) -> Pool -> z_mode -> intrinsic head
```

`z_ctx` 允许保留 absolute deployment information；`z_mode` 要求对 global translation 这一 nuisance 稳健，同时保留内部构型与 support/relationship 信息。context head 只读 `z_ctx`，intrinsic head 只读 `z_mode`，cross-readout 仅用于检查泄漏。

P3-T5R 的任务是修正评价语义，不是把所有 context response 压到零。当前暂以 IDSSE 作为 SNGAR 的开发替代数据，固定双通道必须先与 raw single-channel、centered single-channel、relational pooling 和 raw-coordinate/Procrustes baseline 比较；只有 context noninferiority、intrinsic task signal、mode robustness 和 geometry relation 同时出现稳定信号，才允许 bounded autoresearch。candidate lock、IDSSE 保留 match 检查和真正独立 provider confirmation 全部完成前，AMR-Fixed 保持 `NOT_STARTED`。
