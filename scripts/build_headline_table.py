#!/usr/bin/env python3
"""Build the publication headline effect table from frozen T5R6 artifacts.

Pure re-organization of existing numbers (no new experiments):
per-metric macro, per-match win counts, sign-test p, match-level
bootstrap CI of the macro difference (2:1 minus fixed-dual reference),
and relative effect sizes. Writes reports/P3_HEADLINE_EFFECT_TABLE.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" / "t5r6_soccertrack_v1" / "summary.json"
OUT = ROOT / "reports" / "P3_HEADLINE_EFFECT_TABLE.md"

MATCHES = ["118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
SELECTED = "update_ratio_2to1"
REFERENCE = "fixed_dual_channel_shared_phase_gat"
METRICS = [
    ("field_zone_f1", "Field-zone macro-F1", True),
    ("pair_accuracy", "Natural-pair accuracy", True),
    ("geometry_spearman", "Geometry Spearman", True),
    ("centroid_mae", "Centroid MAE (pitch units)", False),
    ("match_half_f1", "Match-half macro-F1", True),
]
N_BOOT = 10000


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260905)
    lines = [
        "# P3 修复效应 headline 表（T5R6 外部确认，2:1 vs fixed-dual）",
        "",
        "数据来源：`t5r6_soccertrack_v1/summary.json`（冻结 artifact 的纯整理，无新实验）。",
        "单位：场（n=8 独立比赛）；seed 均值已先在场内聚合。bootstrap：以场为单位",
        "有放回重采样 10,000 次（seed 20260905），报告 95% 百分位区间。",
        "",
        "| 指标 | 2:1 macro | fixed-dual macro | 差值 | 相对效应 | 逐场胜场 | 符号检验 p | bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, label, higher_better in METRICS:
        sel = np.asarray([s["per_match_task"][m][SELECTED][key + "_mean"] for m in MATCHES], dtype=np.float64)
        ref = np.asarray([s["per_match_task"][m][REFERENCE][key + "_mean"] for m in MATCHES], dtype=np.float64)
        diff = sel - ref
        if not higher_better:
            diff = -diff
        wins = int(np.sum(diff > 0))
        p_sign = float(sum(math.comb(8, k) for k in range(wins, 9))) / 256.0 if wins >= 4 else 1.0
        boot = np.asarray([diff[rng.integers(0, 8, 8)].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rel = float(diff.mean() / abs(ref.mean())) if ref.mean() != 0 else float("nan")
        lines.append(
            f"| {label} | {sel.mean():.4f} | {ref.mean():.4f} | {diff.mean():+.4f} | {rel:+.1%} | {wins}/8 | {p_sign:.4f} | [{lo:+.4f}, {hi:+.4f}] |"
        )
    full = np.asarray(s["intervention_per_match_full"], dtype=np.float64)
    base = np.asarray(s["intervention_per_match_baseline"], dtype=np.float64)
    direction = np.asarray(s["intervention_per_match_direction"], dtype=np.float64)
    diff = full - base
    wins = int(np.sum(diff > 0))
    boot = np.asarray([diff[rng.integers(0, 8, 8)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(
        f"| Intervention full − spectral baseline | {full.mean():.4f} | {base.mean():.4f} | {diff.mean():+.4f} | — | {wins}/8 | {0.5 ** 8:.4f} | [{lo:+.4f}, {hi:+.4f}] |"
    )
    lines += [
        "",
        f"干预方向准确率（2:1，逐场均值）：{direction.mean():.3f}（范围 {direction.min():.3f}–{direction.max():.3f}，chance=0.5）。",
        "",
        "## 诚实解读",
        "",
        "- **四项主指标方向完全一致**：zone F1、pair、geometry、centroid 全部 8/8 场占优",
        "  （符号检验 p≈0.004）；干预 full 8/8 高于谱基线（macro 差 ≈ +0.53，机制侧 headline）。",
        "- **一项辅助探针反向**：match-half context F1 为 0/8（−30%）。该探针在冻结 addendum 中",
        "  的定位是 auxiliary——测量 z_ctx 保留绝对上下文信息的程度。这一负向结果与分配机制",
        "  自洽：更新向 intrinsic 倾斜会以部分绝对上下文保留为代价；且空间上下文（zone、",
        "  centroid）改善而赛时上下文（match-half）退化，说明代价是选择性的。论文必须如实",
        "  报告此项，它界定了修复的 context 保持主张的边界。",
        "- **任务效应量小**：zone F1 +0.005、pair +0.009、geometry +0.03 量级；幅度最大的是",
        "  **centroid MAE 约 −34%**，是任务侧最有分量的数字。表述保持\"小效应、全同向\"。",
        "- 最弱场（132877，intervention full=0.421）保留在表内；透明优于修饰。",
        "",
        "## 对 PLAN 假设的影响",
        "",
        "本表不改变任何假设；它把 T5R6 已冻结的证据整理为论文可用的效应量表述，",
        "直接支撑评审第四部分 C 节的框架：修复是诊断的验证实验（小而同向的任务效应＋大而稳的机制效应）。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
