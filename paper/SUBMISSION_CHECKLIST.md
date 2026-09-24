# SUBMISSION CHECKLIST — ICLR 2027 投稿就绪状态

> 生成：2026-09-22（结构整理收口）。截止：全文 2026-09-25 AOE，主文 9 页。
> 下方2026-09-22/23检查为历史记录，不能代替2026-09-24修复结果。当前审计入口：`reports/e1a933_review/PAPER_PATCH_AUDIT.md`。

## ✅ 结构侧（2026-09-22 已就绪）

- [x] 论文骨架：`paper/main.tex` + `sections/00–08`（最终叙事 spine）
- [x] 原始结果来源：`reports/final_closure/MASTER_CLAIM_LEDGER.md`；当前修正口径：`reports/e1a933_review/AFFECTED_CLAIMS.csv`
- [x] 一键复算：`reports/final_closure/REPRODUCE_FINAL.md`（路径已对齐 scripts/ 分区）
- [x] 测试：166/166 通过（含冻结证据完整性测试——config sha 对 manifest）
- [x] 仓库导航：README.md（前门）→ PROJECT_MAP.md（叙事→目录）→ 各目录 INDEX.md
- [x] Fig2（P1 双曲线）/Fig3（P3 迁移）已接线（`paper/figures/`，`scripts/figures/fig_p1_p3.py` 可重生成）
- [x] 证据完整性：大文件 sha 清单 `artifacts/SHA256SUMS_UNTRACKED.txt`；封存资源清单见 PROJECT_MAP §6

## ⬜ 剩余动作（投稿前；按阻塞顺序）

- [x] 1. OpenReview 行政：机构邮箱 profile、作者名单、互审资格、摘要注册——用户已完成 ✅
- [x] 2. 官方 ICLR 2027 样式包已装（`paper/iclr2027_conference.sty/.bst`，官方站下载；`main.tex` 已接线）。本机TinyTeX可用
- [x] 9. **重标号稳健性（2026-09-22 新增）**：§4 写入交互对称群稳健性（dev 24/24 ＋ confirm 24/24，预注册 `e7f9850`）；§7 新增限制 (xi) 披露所有绝对率均为单一标号约定下的值；摘要补一句。oracle 不变性经重算（5688 dev ＋ 12216 confirm，零 mismatch）。编译后正文仍 8 页。
- [ ] 3. **AI-use 披露为草稿**（`main.tex` 注释块，覆盖探索/代码/分析/审计/写作，需作者核准；不计页）
- [x] 4. `references.bib` 39 条入库（全部官方出处核验，见 `paper/refs_research/VERIFIED_REFS.md`），30 条正文接线渲染（40 为软目标）
- [ ] 5. Figure 1 概念图（§1）未生成（槽位空留，无虚假完成声明）；fig2/fig3 已重生成验证，fig4 四臂图已切 U1/U2 数据源并加 J/H panel（c）
- [x] 6. 页数核验：主文 9 页 ≤9 页 ✓（2026-09-23 本机 TinyTeX 实测，f095 机制句 +4 行后 8→9；参考文献+附录另 4 页不计入；零错误零未定义引用零 overfull）
- [ ] 7. 双盲扫描：正文/补充/figures 元数据无作者身份痕迹（当前 Anonymous ✓，换样式后复查）
- [x] 8. ~~sha256 锁重新部署~~ **已取消**（用户指示 09-24）：分区后时代锁现场可复验性不可恢复，重锁边际价值低；审计路径 = scripts/INDEX.md（bc3b169 检出）＋ SHA256SUMS_UNTRACKED.txt

## 投稿打包注意

- 补充材料若含代码：用 git 快照（匿名化），不要直接打包工作树（`data/` 48G、
  `artifacts/` 36G 本地大文件不入包；sha 清单随包）。
- `references.bib` 在仓库根，`main.tex` 以 `../references` 引用——打包时保持
  相对结构或内联 `.bbl`。
- 封存资源（holdout_909/confirm_1007/holdout_895/J03WQQ/SoccerTrack-v2）的任何
  二次读取是新协议事件，需用户显式授权。

## 2026-09-24审计修订

- [x] R10即时措辞与附录生成器修正，保留未受影响数值；T2负交互进入摘要/正文。
- [x] 新增任务与模型证据同步：T1/T2输入优势、ResNet18 J及其上采样/初始化边界。
- [x] 已完成结果同步与本轮PDF构建：主文9页、全稿20页；`reports/e1a933_review/paper_revised.pdf`。视觉O03/N01已完成并入稿；追加GPU实验的科学结果待独立解读，状态以各R/O/N报告为准。
