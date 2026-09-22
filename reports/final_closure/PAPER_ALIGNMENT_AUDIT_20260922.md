# PAPER_ALIGNMENT_AUDIT — 2026-09-22

任务：① 全项目扫描可并入最终叙事的遗留内容；② paper/ 全部 9 节与实验结果逐数对齐。
方法：数字逐条对 `MASTER_CLAIM_LEDGER.md`，再抽查 ledger 行对应 artifact 原值
（E1/P1B±confirm/P3_FINAL/RELFEAT/RELFLIP_CONFIRM/P2_OPERATING/P2_AFFINE_BEHAV/
U02(+u02_fixed)/U03/U03B(+s805/s806)/X03B）。

## 任务一结论：内容完整性

主线内容**无缺口**——ledger 全部 claim 行在正文均有落点。以下为"已完成但未进正文"
的候选项，逐条核对历史裁决后确认均为**有意排除**，仅 3 项列出供用户复议：

| 候选 | 状态 | 历史裁决 | 复议建议 |
|---|---|---|---|
| WP2 boundary coverage +7.3pp（CI 不含 0） | 未进正文 | "footnote 级，约束叙事"（WP2 决定） | 可选：§4 加半句支持 coverage-not-precision |
| WP4 static 0.94 vs 0.54（Level D 被拒） | 未进正文 | "transition 语言只描述不因果"（WP4 决定） | 可选：§3(i) 加一个从句 |
| T5R6 SoccerTrack-v2 独立确认（CONFIRMED，8 场同向） | 未进正文 | 属旧 P3 几何叙事，路线调整后主线外 | 不进正文；留作 **rebuttal 储备**（独立 provider 证据） |
| foundation bridge（DINOv2/v3/CLIP/SigLIP2 全阴性） | §7 limitation 一句 | 多次"正文不动"记录 | 维持 |
| event_updater / radius_loss / l002_ball / novelty_round3 | 未进正文 | 阴性/混杂，"正文零修改"记录 | 维持 |
| 书法 / AMR / 篮球 | §7 背景与未来工作 | 立项以来多次裁决 | 维持 |

修正一处文档失真（本次审计附带）：README/PROJECT_MAP 原称 bridge 线"进论文 §6"，
实际为 §7 limitation 一句；§6 实为 E7/X03/U02/阈值温度失败移除集。已更正。

## 任务二结论：逐数对齐结果

核查约 60 个数字/主张。**55+ 项与 artifact 原值完全一致**（含 headline 151/176=85.8%、
turn 28/64=43.8%→20/64、matched-single 61.0%(n=251)、C1 52.4→34.7%、C0 26.2→23.8%、
action 75.0→62.5%(n=144)、static 18.0→10.7%(n=512)、P1 分位反转 0.36–0.61→0.98–1.00
（12 组合 11×1.00+0.983）、grouped-CV AUC、top25 保留（101/271）、P3 全族
（R_full 0.03–0.10 / M 0.36–0.40 / R_endpoint 0.43–0.48 / J 0.05→0.06–0.08 / s23 反转
0.920→0.947）、relfeat/relflip 全族、足球九数、像素五风格+s805/s806 增益、
X03 全族、U02 3.6×/0.57→0.88/0.23 vs 0.22、T*=8.41/7.88、E7 60 训练）。

**发现并已修复 6 处**（全部为措辞/笔误级，无任何结论变化）：

1. §5：`static accuracy unchanged at 0.090` → **static error**（u03 `static_err`=0.0898
   两臂相同；atomic accuracy 实为 ~0.53）。已改。
2. §4.3：`improved static accuracy (0.15–0.17 → 0.06)` → **reduced static error**
   （RELFEAT.json static 字段是错误率：raw 0.149–0.171 → relfeat 0.059–0.064）。已改。
3. §4：段落标题残留 `30.1/30.2/30.3` 编号（会印入 PDF）→ 已删编号。
4. §3：top25 `~5% static error` → `~4%`（P1B：dev 4.21%、confirm 4.08%）。已改。
5. §6：`move 0–51% of action selections` **无源**（U02 报告与 artifact 均为
   T0.5/T2.0 的 10–50%）→ 改为有源表述 "at T=0.5/2.0 move 10–50%"。已改。
6. §4.1 硬编码 `(Sec.~7)` → `(Sec.~\ref{sec:external})`。已改。

**ledger 同步**：relfeat 行补 static error 字段；pixel 行补 static_err 0.0898 与
"0.090 是 error 非 accuracy"禁令；新增 U02 temperature move 行（含禁用 "0–51%"）。

**遗留待用户决定（未动）**：

- §4.3 relflip 句 "reaches J 0.38–0.47 … on confirmation"：J 区间跨 dev(0.38–0.42) 与
  confirm(0.44–0.47)，而 "on confirmation" 修饰整句。边界合规但建议拆分两个区间。
- P2 affine 的 s11 网格贴边（(1.0,−8.0)）：ledger 要求引用时声明局限，§7 未声明。
  建议 §7 加半句或确认不引用该拟合。
- §7 引用零接线：`\cite` 全文未使用，bib 11 条中无 Press/Guo/Geifman/Composition
  Collapse 对应条目（即 SUBMISSION_CHECKLIST 第 4 项；related work 当前为纯文本点名）。
