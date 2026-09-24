"""Verify every recorded witness against scalar production oracle; write report."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from d06_oracle import channel_label
OUT=ROOT/'artifacts/e1a933_review/football'
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args();OUT=args.out
r=json.loads((OUT/'FOOTBALL_RESULTS.json').read_text());checked=0
for line in (OUT/'events_candidates.jsonl').open():
 row=json.loads(line);q=row['probe']
 if not q.get('E_found'):continue
 w=q['witness'];p=np.array(row['passer_xy']);rr=np.array(row['receiver_xy']);d=np.array(row['defenders_xy']);newr=rr+w['receiver_delta'];newd=d.copy();newd[q['defender_index']]+=w['defender_delta']
 labels=[channel_label(p,a,b,.5,1.)['label'] for a,b in [(rr,d),(newr,d),(rr,newd),(newr,newd)]]
 assert labels==w['labels_0_A_B_AB'] and labels[0]==labels[1]==labels[2] and labels[3]!=labels[0]
 cap=r['matches'][row['match']]['motion']['cap_m'];assert np.linalg.norm(w['receiver_delta'])<=cap+1e-9 and np.linalg.norm(w['defender_delta'])<=cap+1e-9
 checked+=1
(OUT/'WITNESS_VERIFICATION.json').write_text(json.dumps({'checked_all_found_witnesses':checked,'invalid_witnesses':0,'scalar_production_oracle':True},indent=2))
lines=['# R08 / O05 足球可构性 v2','',
'状态：R08 真实生产修复与三场有限探针完成；**自然 E 发生率已测**（见 `R08_NATURAL_E_RATE.md`，主格 8/1449 = 0.55% 情境级）。O05 已完成真实坐标候选代理、受控缺测的因果基线与 12 臂学习模型（学习不超过恒速+解析基线）；O06 已执行 272 次原生 VLM 请求（可选桥，基础能力不足）。本文件是旧快照，现行状态以 `R_FIXES_REPORT.md` 为准。','',
'旧 `paired` 文件原样保留，只能称 supported_single_keep_and_flip，不能作 E 可构率。新的 E 严格要求 y0=yA=yB != yAB。oracle 为所有防守者均不阻塞的 AND；固定端点分别移动两个防守者不能使 open 的两个保持单编辑共同变 blocked。本轮选择接球端点与真正最近通道约束防守者，49 个解析法向引导幅度对 + 128 随机对，找到 witness 后 24 步边界二分，不扩大 cap。','',
'事件先按比赛时间排序，用明确半场事件边界 forward-fill 缺失 period；取 match+period 内唯一 timestamp，双侧等距选较早帧，重复 person ID 拒绝。只保留 player，排除球/裁判；检查 pass/recipient 同球队、event team 一致、两队与 11 个防守者。每个候选记录角色/坐标/时间/实际传球与否、原始 outcome（仅实际传球）和抽样权重。', '',
'cap 来自同 player+match+period 的连续差分（0<dt≤.12s），速度/加速度 p95 按 .2s 窗口换算。分别约1.214/1.199/1.174米；这是各场完整轨迹的几何扰动包络，含位置量化噪声，不是独立部署标定或运动学共同可实现性证明。新率不能与旧 .15m cap 的3–7%直接比较。','',
'每场等间隔抽80个已选传球事件，共240；通过角色审计162。候选为当时同队所有2–45m且场内的接球者，事件内全枚举，条件 inclusion probability/weight=1。角色不合格（recipient 实为对手、非11名对方球员）单独记录，不能全解释成追踪缺测。','',
'| 比赛 / 预设用途 | 审计后事件 | 实际所选合法接球 E可构 / 分母 | 全合法候选 E可构 / 分母 | E编辑对 / 单编辑均保持对 |', '|---|---:|---:|---:|---:|']
for match,v in r['matches'].items():
 a=v['observed_selected_passes'];b=v['all_legal_receiver_candidates'];lines.append(f"| {match} / {v['split']} | {v['passes_used']} | {a['E_found_n']}/{a['candidate_n']} | {b['E_found_n']}/{b['candidate_n']} | {b['E_pair_n']}/{b['single_keep_pair_n']} |")
lines += ['',f'合计实际已选合法候选 48/155、全部候选 670/1449 有至少一个 E；全 {checked} 个 witness 经标量生产 oracle 独立重算，标签合同和 cap 全满足。引导与随机命中率分别存 JSON，不能将177对搜索视为177个独立样本。','',
'三个分母须区分：①自然轨迹中 E 的真实发生率**已测**（`R08_NATURAL_E_RATE.md`：情境级 8/1449 = 0.55%、事件级 8/162 = 4.94%、帧对 10/7117；窗口依赖 0.2s→2.0s 为 0.55%→16.98%；标签为几何 oracle，非传球成功率）；②在真实已选接球者与全合法候选上的受控可构率如表；③E编辑对 / 两个单编辑均保持对如表。候选扩展的富集是模型无关几何搜索，不是自然传球策略频率；两者相差约 84×。','',
'完整坐标标签与解析基线是同一 oracle，准确率1是定义恒等式。O05 额外把接球者与约束防守者的当前坐标遮挡 .2 秒，只用过去的最近值或两历史帧恒速外推，再解析判别；其余实体当前可见。标签始终完整坐标几何，不使用未来帧填输入。','',
'| 比赛 | 候选数 | Last value | Constant velocity |','|---|---:|---:|---:|']
for match,v in r['matches'].items():
 a=v['controlled_mask_baselines']['last_value'];b=v['controlled_mask_baselines']['constant_velocity'];lines.append(f"| {match} | {a['n']} | {a['accuracy']:.4f} | {b['accuracy']:.4f} |")
lines += ['',
'这支持继续独立评估 D07/U09，而不构成“学习结构迁移到不完整tracking”的主贡献。**仍未做**：自然缺测模式下（而非人工有定义遮挡）的学习模型与状态估计拟合、自然时间路径的提前/滞后秒数与误报评估、真实传球成功预测。当前人工遮挡是明确受控压力测试。按比赛train/dev/test命名用于以后开发；O05 的 12 臂已训练但未超过恒速+解析基线，不应包装成已完成三路泛化实验。','',
f"预算：既有3场数据，无下载/付费/API/GPU；运行约 {r['wall_seconds']:.1f}s。命令：`.venv/bin/python experiments/e1a933_review/football_repair.py --events-per-match 80`；全 witness 检查与报告：`.venv/bin/python experiments/e1a933_review/football_summarize.py`。",'',
'证据：`artifacts/e1a933_review/football/FOOTBALL_RESULTS.json`、`events_candidates.jsonl`、`WITNESS_VERIFICATION.json`。未新增外部生信数据。']
(ROOT/'reports/e1a933_review/FOOTBALL_E_FEASIBILITY_V2.md').write_text('\n'.join(lines)+'\n')
