# NOVELTY_AUDIT

日期：2026-09-11

状态：`INCOMPLETE_FOR_PILOT_DECISION`

核验方式：使用项目 `docs/07_PRIOR_ART_AND_SOURCES.md` 指定的三个 arXiv HTML 原始页面，检查摘要、相关工作、实验设置、主要表格和结果段；未核验所有附录、release artifact 与代码，因此不作穷尽式新颖性结论。

| 最近邻 | 同 H 分叉且所有臂不可回读 H | 同 final actual/cap | direct vs staged | 同 K 不同 schedule | 固定 B rewrite | query-blind 写入 | 信息类型/reader 分离 | 任务/策略含义 | 定位 |
|---|---|---|---|---|---|---|---|---|---|
| [EXT-RD](https://arxiv.org/html/2607.08032v1) | 否：reversible 臂保留 archive 并在 query 时 retrieval | 报告共同 budget/BPT，未证明实际文本长度相等 | 否：主要是 irreversible summary vs reversible archive | 部分：改变 compaction frequency/events，不是同 K 中间 budget schedule | 否 | 未明确到本项目协议 | 部分：single-turn needle 与 repeated agent facts；非本项目四类配对 | 是：重复不可逆摘要与可回读 memory 的部署差异 | §§13–14，尤其 repeated-compaction experiment |
| [EXT-CLIFF](https://arxiv.org/html/2608.22752v1) | 部分：主 compaction 对照不回读原 H，但另有 retrieval operator | 50/25/10% cap/ratio；不是同 actual length 证明 | 部分：single-round 与 five-round compaction，不是 direct/staged 分叉 | 否：固定五轮 50%，未做同 K schedule 对照 | 否 | 任务上 query-blind，但写入协议细节仍需代码核验 | 是类型化 constraint/procedure/belief/preference/episodic；reader-family 不是本项目主对比 | 是：typed retention、verifier、decomposition/retrieval 有明确策略含义 | §§3–4，Tables 5–9 |
| [EXT-RESTORE](https://arxiv.org/html/2609.08279v1) | 否：restore 臂在 read time 显式注入 gold evidence | 8k/30k/80k store caps；非同 actual memory 证明 | 否：FIFO/random/redundancy-aware/LLM-importance eviction | 否 | 否 | retention decision 对 question/gold blind；read retrieval 可 query-aware | 部分：两 retrieval regimes、两 readers、按 evidence layout；不是 direct/staged | 是：区分 destruction、retrieval miss、reader utilization，并给出 restore/placebo 诊断 | §§3–5，尤其 §4 setup、§5.2–5.4 |

## 本项目尚可检验、但本轮未测的新知识

在同一完整 H、query-blind 写入、同一最终 cap 下，直接压缩与不可逆 staged/rewrite schedule 的 paired utility difference；同 K 变中间预算；以及在严格实际长度敏感性、reader 分离和自然数据设置下这些差异是否仍有策略意义。由于正式 pilot 没有运行，这些是待测设计，不是结果。

## 不能再声称的创新

- 不能声称首次研究 repeated compaction、知识类型保护、retention/accessibility split、restore counterfactual 或统一 budget frontier。
- 不能把本项目的 fixed-budget path 说成已有工作未覆盖；三篇近邻的实际覆盖范围仍需 release/code/appendix 级核验。
- 不能把上述论文的数值直接当作本项目效应、正对照或确认结果。

## 未核验范围

未逐项检查三篇文章所有附录、代码实现、数据版本与其 exact final actual memory lengths；也未完成更广泛文献检索。依据 docs/05，本门只能标为 `INCOMPLETE`，不能通过 novelty gate。

结论：`INCOMPLETE`；允许保留为后续 pilot 前的查重输入，不足以 GO。
