# FINAL_NOVELTY_DECISION (2026-09-22, 通俗中文)

## P1: confidence 到底能不能衡量变化后的可靠性？
1. 统一 population 上成立。同一批 parent、同一个 confidence 分数：静态和
   preserve 错误随 confidence 下降（0.43→0.05），flip 更新错误随 confidence
   上升（0.58→1.00）。6/6 模型，确认池复现。
2. 是的，含义分离：confidence 对"现在对不对"是经典校准方向，对"变了之后
   还对不对"是反方向。两者在同一总体、同一分数上同时成立。
3. 不主要由阈值距离解释：|f|/|Δf| 的 AUC（0.99）高于单用 confidence（0.93），
   但联合模型里 confidence 系数仍显著为正（+0.75），confound 模型的 AUC 增益
   CI 不含 0。所以是"大部分走阈值距离＋残留 confidence 效应"，P1 收窄但成立。
4. Retention top25%：静态错误~2%、条件 flip 错误 100%、preserve 错误~5%。
   信高置信的部署者换来的是"现在基本全对、真变了基本全错"。总体部署风险取
   决于环境里变化的比例 rho（rho=0.25 时综合风险 0.28，rho=1 时 1.0）。
5. 通过独立 confirm（confirm_1007，冻结阈值/设定）：曲线形状、confound 增益、
   配对残留三项全复现。等级：P1-STRONG（坐标域）。

## P3：repair 是恢复能力还是搬错误？
6. AB gain 里真正 full repair 很少：基线 110 子集上 R_full 只有 0.03–0.10。
7. 大部分是 error migration：M=0.36–0.40，每 10 个端点"修复"里约 8–9 个是
   AB 对了但原子错了。preserve-balanced 试点（3 种子）降不下 migration，
   J 也无一致提升。
8. Joint consistency 几乎不动：0.05–0.06 → 0.06–0.08，所有臂（含新试点）。
9. 不值得提 atomic-preserving repair 方法：keepbal 试点阴性。只有测量层面
   的 novelty（transition table + denominator discipline），方法分支关闭。

## P2：repair 收益是能力还是 operating point？
10. Affine/threshold shift 解释一部分（single、comp-endpoint 方向），但解释
    不了 static 增益、joint 和 action-lexico 成功——affine 在这些上面持平或
    更差。等覆盖下 repair 质量更高但 cost 也更高。所以是两者兼有：component
    能力真涨了（static/single/recall/ranking），action 覆盖是 operating-point
    移动，而 compositional capability（joint）没动。
11. P2 进 Discussion，带一句正文钩子：测量都在 dev 场景/银行上，无 fresh
    确认，撑不起独立正文结果，但足以把"coverage"口号换成数字。

## P3 定级与收官
12. P3 = phenomenon（P3-PHENOMENON）：migration 主导且稳健（preserve 平衡也
    打不掉）；方法阴性；"首次条件化"本来就不属于我们（Press/双门控在先）。
13. 最强的 2–3 个 novelty claim：
    (a) P1 置信反转＋筛选后果（坐标，确认池，6/6 模型）；
    (b) P3 error migration 主导＋分母解剖（配对、有 s23 反转）；
    (c) P2 能力/operating-point 分离＋规则依赖（affine 对照）。
14. 必须删除或降级：①"首次条件化原子正确"（删，Press 在先）；②"compositional
    C1 miss 52.4→34.7"（改为 single-flip C1，headline 组合修复改报 151/176 与
    109/141 非配对）；③"incidence/切向决定失败率"（WP1 已杀，改成分母/采样语言）；
    ④"所有修复都不提高精度"全称（删，静态 0.18→0.11 白纸黑字）；⑤足球 P1 旧数
    （作废，已重写为地板 null）；⑥"修复治愈更新失败"（改为：残留 0.53–0.92，
    迁移主导）。
15. 值得继续的强 lead：L1 relfeat-J（单种子，需多种子验证）；其余 L2–L4 为
   观察级。确认池 P1 部分已用；新方法若出现，用 dev，不动 holdout。

## 故事线（数据支持才用）
Static confidence、compositional repair、downstream success 确实各自藏着
不同的正确概念：高置信标出"现在对但最抗拒真变化"的状态；修复把端点修对
的同时把原子搞错（迁移主导）；下游成功靠多尝试而不是把已尝试的做好。
三者统一在"正确性不是一维的"这句话下——各自证据独立成立，不强拼因果链。
