# e832 复现入口

本轮没有新租 GPU。重量级旧 GPU 压缩包仍在本机，不在 GitHub 单文件上限内。下面的入口是已经跑过的脚本，不是用哈希代替数据。

## 结构对照

```bash
python3 -m unittest experiments/e832_focus/structure/test_a_definitions.py -v
python3 experiments/e832_focus/structure/expressivity_repo_audit.py
python3 experiments/e832_focus/structure/a_source_compare.py --stage all
```

主张分母是 `artifacts/e832_focus/structure/bank_quartets.npz`，2188 个 E、871 个 parent，sha256 `15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c`。数字表是 `artifacts/e832_focus/structure/metrics_continued.json`。结论是 `reports/e832_focus/STRUCTURE_DECISION.md`。训练名单只读 `artifacts/f095_campaign/D01/scenes_N/scenes.npz`。不读封存池，不读 dev512。线程数 4。

## 视觉表

```bash
python3 experiments/e832_focus/vision/build_canonical_fourarm_table.py
```

输入是已完成的 `artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix/`。不重新训练。结论是 `reports/e832_focus/VISUAL_DECISION.md`。

## 算力判断

不训练。复算笔记是 `experiments/e832_focus/sampling/mac_ratio_audit.md`。裁决是 `reports/e832_focus/SAMPLING_DECISION.md`：不启动。

## 五路线新证据

```bash
# 路线1：source/T1/T2；Round 2设置已冻结在 artifacts/e832_focus/route1/round2_manifest.json
python3 -m unittest discover -s experiments/e832_focus/route1_cross_task -p 'test_*.py' -v

# 路线2：fresh visual data + tests + CPU PILOT_ONLY + data-bound GPU runner
python3 experiments/e832_focus/route2_visual/run_route2.py
# 合同修复后的正式v2入口（GPU host）
CUDA_VISIBLE_DEVICES=0 python3 experiments/e832_focus/route2_visual/gpu_run.py --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json --data artifacts/e832_focus/route2/data --out artifacts/e832_focus/route2/gpu_run_v2
CUDA_VISIBLE_DEVICES=0 python3 experiments/e832_focus/route2_visual/static_baseline.py --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json --data artifacts/e832_focus/route2/data --out artifacts/e832_focus/route2/gpu_run/static_baseline_v2

# 路线3：解析/集合/关系/容量基线
python3 experiments/e832_focus/route3_baselines/strong_baselines.py --out artifacts/e832_focus/route3

# 路线4：新parent与刚体坐标条件
python3 experiments/e832_focus/route4_robustness/robustness.py --out artifacts/e832_focus/route4
```

路线1报告：`reports/e832_focus/route1/REPORT.md`；source Round 2为1414 E/389 parents，T1/T2不构成跨任务确认。路线2 v2结果在`artifacts/e832_focus/route2/gpu_run_v2_remote_20260925/corrected_results/`，状态为有界阴性/不稳定；不要使用旧v1的J3字段。路线3解析器在230 E/65 parents上J3=1.0，因此source学习比较是parser-bounded diagnostic。路线4只支持207 E/66 parents的刚体坐标稳健性。

## 论文

口径迁移说明是 `reports/e832_focus/PAPER_MIGRATION.md`。真实Figure 1选择回执是 `reports/e832_focus/FIGURE1_SELECTION_RECEIPT.json`；统一fallback表是 `reports/e832_focus/MAIN_TABLE_FALLBACK.csv`。本轮新增结构句只在附录 `paper/sections/app_l015.tex`，没有挤进已经排满的结论页。
