# FINAL_NOVELTY_DECISION (2026-09-21, 通俗中文)

## 各包判决
- WP1: UPDATE_GEOMETRY_NOT_SUPPORTED（composition-as-update 单项成立）
- WP2: BOUNDARY_ASSOCIATION_ONLY（coverage 注脚，无对齐修复故事）
- WP3: 未执行（无 geometry 可推广；不记 verdict，记 killed-by-WP1）
- WP4: STATIC_AND_UPDATE_SIMILAR（Level D 被拒）
- WP5: PARK（G1 未过）

## 十三问
1. composition blindness 是否是 missed semantic update 的特例？是。111/111
   的 AB-miss 背后都有零转折路径，且无"转过去再转回来"——路径级直接证据。
2. 84% vs 44% 能连成一条规律吗？不能。emergent 全程平坦；切向/法向匹配
   SMD 结构性失败。维持 E1 原判：两个真数、两个分母。
3. 严格 matching 后切向更易失败吗？方向性有（+14pp）但 SMD 不过门——不许 claim。
4. model 与真实边界有可测量的 mismatch 吗？有，且很 stark：90% 的语义边界点
   附近 r 内无模型边界；对齐均值≈0。
5. flipmine 改善 mismatch 吗？只改善 coverage（9.6%→16.9%，CI 不含 0），
   不改善朝向/偏移。coverage-not-precision 再添一条。
6. geometry 在第二种 relation 上复现吗？未测（WP1 死后 WP3 已杀）。
7. flat boundary 更容易吗？未测。同上。
8. transition 指标比 static 更能预测 action 吗？不能。模型级 0.54 vs 0.94，
   static 完胜。场景级有混杂下的受控贡献，但不够升级指标。
9. geometry-aware repair 优于 flipmine 吗？未测（G1 未过，WP5 parked）。
10. 值得进正文的：composition-as-update 路径证据（111/111＋零转折形态），
    可作 Figure 2 的 panel（端点 headline 的路径展开）；coverage 注脚一句话。
11. 只进 appendix 的：WP2 对齐/偏移分布表；WP4 六臂统一表；WP1 平坦曲线
    （诚实阴性）。
12. 必须彻底放弃的：incidence 定律、84/44 一统、boundary-alignment repair、
    update-指标升级（Level D）、跨 family 推广。
13. 最终 novelty 定级：**update（一级残）**——"static correctness != correct
    updating" 主线不变，composition 降级为 update 的特殊构造（有路径证据），
    repair 仍是 boundary-coverage（添 coverage-shift 注脚）。不是 composition，
    不是 boundary geometry，不是 repair升级。

## Claim freeze
- 主 spine 不动（标题/摘要/引言已是 update 线，无需改）。
- 03 节可加一段 composition-as-update 路径证据（ цифры 111/111＋196/237 零转折），
  出处 reports/update_geometry/WP1_FINAL_REPORT.md（写稿时由用户确认措辞）。
- 04/07 节可加 flipmine coverage-shift 一句＋WP2 附录引用。
- 局限加两行：incidence 阴性（诚实）、update-指标未胜 static（诚实）。
- `geometry_confirm` 从未铸造——好事：one-shot 额度还在，以备审稿人要求
  composition-path 的 fresh 确认（属新 claim 的 confirm，需新协议）。
- paper/ 本轮零修改已执行。
- Stop rule（§17）：不再开第六个大方向。回主线投稿。
