# CONFIRMATION_USAGE — confirm1007 逐 claim 使用表（初版）

> R9.3 要求。原则：训练 parent 独立 ≠ 后续一切假设"从未见过"；读取归档结果 ≠ 污染。
> 每次使用记录：使用者 / 输入 bank SHA / 输出 / 是否新增选择自由度。

| # | 使用者 | 输入 | bank SHA | 输出 | 新增选择自由度 | 状态 |
|---|---|---|---|---|---|---|
| 1 | P1 首分析（p1b_confirm/P1B.json） | confirm1007 bank + dev 冻结阈值 | 28365ae3…（以 manifest 为准） | P1B.json：分位曲线/CV/保留分析 | 无（阈值 dev 冻结） | 首用 |
| 2 | P1G 纠错重算（P1G_CORRECTED.json） | confirm1007 bank + dev 冻结 t25 | 同上 | 修正残留 0.32–0.62 | 无（同阈值纠正 bug，非新选择） | 重算（非新确认） |
| 3 | relfeat/relflip 确认评价（RELFLIP_CONFIRM.json） | confirm1007 bank + 已有模型 | 同上 | relflip J/R_full/M | 有（新模型族评价）— 但属开发中已定路线的确认，非事后挑选 | 确认 |
| 4 | p1b_res_v2 / p1b_confirm_v2（paired refit） | 同 1 输入 | 同上 | 配对 OOF 区间 | 无（同数据重分析） | 重算 |

禁区：holdout_909 自 E1 后零新读取；confirm1007 不得用于新一轮候选选择（candidate selection）；任何新训练/新选样需用户授权的新协议事件。
