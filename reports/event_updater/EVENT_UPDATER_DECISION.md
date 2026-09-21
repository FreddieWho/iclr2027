# Event Updater Decision: SIDE_DIAGNOSTIC_ONLY (2026-09-22)

Main line: **`EVENT_UPDATER_NEGATIVE`** — explicit update architecture on
frozen state features fails the G2 gate on all 3 seeds (0.67/0.68/0.72 <
0.80). Training objective was never the whole story: even with direct
stay/flip supervision, frozen z does not yield a generalizable updater.
STOP: no composition test, no multistep, no counterexample loop, no confirm,
no arch expansion, paper untouched.

Overall round verdict: **`SIDE_DIAGNOSTIC_ONLY`** — main line negative, but
the radius diagnostic (side D) is a stable, decomposed, 3-seed-consistent
independent signal worth retaining (miss cases need ~2x model displacement
at matched oracle difficulty). Side C permanently killed.

## 九问（人话）
1. Static和Updater谁在A+B上更好？没比——Updater连单步门都没过，没资格上组合考场。
2. 优势是否来自真update？无优势，无此问。
3. crossing多了谁更稳？没测（G4未开）。
4. Counterexample-guided更省样本吗？没测（G5未开，主线已停）。
5. missing/false update能闭环吗？不能——连开环的单步都只有0.7。
6. 二阶交互解释composition miss吗？不能。方向随种子翻转，永久关闭。
7. 半径失配解释transition miss吗？能，这是本轮唯一活口：miss的模型翻转半径约2倍于hit，oracle难度相同。保留为诊断信号，不自动发展成loss。
8. 哪条值得升级为新主线？没有。Updater和二阶都不值得；半径只值得在讨论段留一行。
9. 哪些旁路永久关闭？Second-order（C）。半径（D）保留观察，不施工。
