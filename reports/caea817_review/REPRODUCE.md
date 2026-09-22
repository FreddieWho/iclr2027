# REPRODUCE.md — caea817 整改后复算入口（R9.2）

> 与 `reports/final_closure/REPRODUCE_FINAL.md`（冻结链全文）互补：本表只收**主文每条主结果**
> 的运行入口。约定：先 `export LD_LIBRARY_PATH=/opt/anaconda3/lib:$LD_LIBRARY_PATH`；
> torch 线程按各脚本固定；banks/checkpoints 只读（先对各自 manifest 验 sha）。
> ⚠️ E1（holdout_909）是预注册单次读取，**不可重跑**——其复算=直接读归档
> `artifacts/last15h/E1/holdout_909/result.json`（对照 REPRODUCE_FINAL 首部的核对命令）。

| 主文结果 | 命令 | 必需输入/权重/版本 | 输出 |
|---|---|---|---|
| §3 headline 151/176、turn 28/64、C1/C0、action λ2（fresh） | （冻结单次运行，不重跑） | `artifacts/last15h/E1/holdout_909/result.json`（sha 见该目录 manifest） | — |
| §3 discovery turn 84.4%（108/128） | `python3 experiments/last15h/n01_scan.py --out artifacts/last15h/N01/round1` | dev pools train_101/eval_202（归档） | N01/round1（对比归档值 0.844） |
| §3 P1 分位曲线（dev，6 模型） | `python3 experiments/next_novelty/p1_unified.py --bank dev512` 然后 `python3 experiments/next_novelty/p1b_analysis.py --indir artifacts/next_novelty/p1_unified --out artifacts/next_novelty/p1b_res` | bank dev512（sha 21786139…）、9 冻结 checkpoints | `artifacts/next_novelty/p1b_res/P1B.json` |
| §3 P1 确认曲线（confirm1007） | `python3 experiments/next_novelty/p1_unified.py --bank confirm1007 --p1old artifacts/p123_upgrade/p1_confirm --out artifacts/next_novelty/p1_unified_confirm` 然后 `python3 experiments/next_novelty/p1b_analysis.py --indir artifacts/next_novelty/p1_unified_confirm --out artifacts/next_novelty/p1b_confirm --freeze artifacts/next_novelty/p1_freeze.json` | bank confirm1007（sha 28365ae3…）、dev 冻结阈值 | `p1b_confirm/P1B.json` |
| §3 P1 配对 OOF 区间（整改后口径） | 同上两条命令（整改版 `p1b_analysis.py` 输出 `p1b_res_v2/`、`p1b_confirm_v2/`，paired refit） | 同上 | `artifacts/next_novelty/p1b_res_v2/`、`p1b_confirm_v2/` |
| §3 P1G 修复残留（修正版） | `python3 experiments/next_novelty/p1g_repair.py --indir artifacts/next_novelty/p1_unified_confirm --out artifacts/next_novelty/p1g_confirm_CORRECTED --freeze artifacts/next_novelty/p1_freeze.json --mode confirm` | confirm bank + dev 冻结 t25（生产 loader，缺键即报错） | `P1G_CORRECTED.json`（勿覆盖归档 `p1g_confirm/`） |
| §4 P3 迁移分解（dev） | `python3 experiments/ccm_audit/p3_final_v2.py --bank dev512` | dev512 bank、frozen clean/flipmine ckpts（r04b_s*） | `artifacts/next_novelty/p3_v2/P3_FINAL.json`（8×8 全矩阵+M/R_endpoint） |
| §4 relfeat J（dev/confirm） | `python3 experiments/ccm_audit/relfeat_eval.py --bank dev512` / `--bank confirm1007` | relfeat checkpoints（r04b_s*_relfeat） | `relfeat/RELFEAT.json` 同结构 |
| §4 relflip 确认（四臂同 bank） | `python3 experiments/ccm_audit/relflip_confirm_eval.py --bank confirm1007 --out artifacts/next_novelty/relflip_v2` | confirm1007 bank + 四臂 checkpoints | `relflip_v2/RELFLIP_CONFIRM.json` |
| §5 足球 E2 三件套 | `python3 experiments/last15h/e2_n01_t5r3.py --out artifacts/last15h/E2/n01`；`e2_n03_t5r3.py --out artifacts/last15h/E2/n03`；`e2_n06_t5r3.py --out artifacts/last15h/E2/n06b --hard`（seed 变体 `--seed 23|47`） | T5R3 冻结模型（artifacts/phase3/task_semantic_repair_v1） | `artifacts/last15h/E2/`（对照 HEADLINE_FREEZE） |
| §5 像素（u03/u03b_fixed，冻结） | （冻结运行，不重跑） | `artifacts/next6_ef0f7a3/u03/result.json`、`u03b_fixed/result.json`（sha 见 SHA256SUMS_UNTRACKED） | — |
| Fig 2/3 导出 | `python3 scripts/figures/fig_p1_p3.py` | `p1b_res/P1B.json`、`p3/P3_FINAL.json` | `paper/figures/fig2_p1_dual.{pdf,png}`、`fig3_p3_migration.{pdf,png}` |
| Fig 4 四臂导出 | `python3 scripts/figures/fig_fourarm.py` | `relfeat/RELFEAT.json`、`p3/P3_FINAL.json`、`relflip/RELFLIP_MIG.json`、`relflip_v2/RELFLIP_CONFIRM.json` | `paper/figures/fig4_fourarm.{pdf,png}` |

## 说明
- 整改后新增入口均**不覆盖**冻结产物：v2 输出到 `*_v2/` 或 `*_CORRECTED` 路径；历史 JSON 保持原字节。
- P2 v2（`experiments/repair_decomposition/p2_operating_v2.py` → `artifacts/next_novelty/p2_v2/`）入口待该 JSON 全 seed 落盘后由 ledger 引用（见 RECTIFICATION_LOG R5）。
- 设计完整性测试（历史 manifest/config 锁）与可执行依赖校验是两类验收，分开记录于 `RELEASE_MANIFEST_NEW.md`。
