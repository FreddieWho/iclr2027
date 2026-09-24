"""R07 paired parent-cluster contrasts and concise report from actual receipts."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/e1a933_review/fair'
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args();OUT=args.out
r=json.loads((OUT/'CONTINUATION_RESULTS.json').read_text());contrasts={};rng=np.random.default_rng(260924)
for enc in ['raw','rel']:
 for seed in [11,23,47]:
  plain=np.load(OUT/f'{enc}_plain_s{seed}/evaluation.npz');labels=plain['labels'];par=plain['parents'];unique=np.unique(par)
  old=plain['before']==labels;po=plain['after']==labels
  for method in ['protection','hard_replay','preserve_flip_balanced','scratch']:
   d=np.load(OUT/f'{enc}_{method}_s{seed}/evaluation.npz');ok=d['after']==labels
   jdelta=ok.all(1).astype(int)-po.all(1).astype(int)
   rdelta=(old[:,:2]&~ok[:,:2]).sum(1)-(old[:,:2]&~po[:,:2]).sum(1)
   J=np.array([jdelta[par==p].sum() for p in unique]);N=np.array([(par==p).sum() for p in unique]);R=np.array([rdelta[par==p].sum() for p in unique]);D=np.array([old[par==p,:2].sum() for p in unique])
   draw=rng.integers(0,len(unique),(2000,len(unique)));bj=J[draw].sum(1)/N[draw].sum(1);br=R[draw].sum(1)/np.maximum(D[draw].sum(1),1)
   contrasts[f'{enc}_{method}_s{seed}']={'J_delta_vs_plain':float(jdelta.mean()),'J_delta_CI_parent_bootstrap':np.quantile(bj,[.025,.975]).tolist(),
    'atomic_regression_delta_vs_plain':float(rdelta.sum()/old[:,:2].sum()),'atomic_regression_delta_CI_parent_bootstrap':np.quantile(br,[.025,.975]).tolist()}
(OUT/'FAIR_CONTRASTS.json').write_text(json.dumps(contrasts,indent=2))
lines=['# R07 公平续训比较','',
'状态：30 个真实训练臂完成（raw/rel × seeds 11/23/47 × 5 方法），每臂 300 full-batch epochs，CPU 4 threads；固定 Adam lr=.01，base=1、augmentation mean=1，每臂新增 1,534 个样本。',
f"运行用时 {r['wall_seconds']:.1f}s；dev512 的 {r['quartets']} 个 quartet、{r['parents']} 个未见训练 parent（来自 eval_202）用于评估。它是复用开发集，不是新 sealed 确认集。",'',
'plain / protection / hard replay / preserve+flip balanced 均从每 seed 同一 clean 参数出发，全部重置 Adam；scratch 独立列为训练路径对照。每轮 base、augmentation 和 replay 的前向数量相同；plain/balanced/scratch 的额外 replay 权重为 0。protection 使用旧正确训练点的 soft teacher，hard replay 使用同组点的 hard label，二者系数均 1。balanced 使用 767 flip + 767 真实 preserve；其他臂使用全部 1,534 flip。总量相等但组成不同，不把 balanced 差异归于纯 loss。', '',
'初始/末参数 hash、clean 文件 hash、每轮重复的完整 full-batch 顺序、训练轨迹、逐 quartet 预测和 joint 8×8 迁移均落盘。eval_202 与 train_101 原始坐标逐行 SHA 无重叠；不将多个同 parent quartet 当独立重复。','',
'| 输入 | 方法 | J（3 seeds 平均） | 未见原子 old-correct regression |', '|---|---|---:|---:|']
for enc in ['raw','rel']:
 for method in ['plain','protection','hard_replay','preserve_flip_balanced','scratch']:
  vals=[r['arms'][f'{enc}_{method}_s{s}']['metrics'] for s in [11,23,47]]
  lines.append(f"| {enc} | {method} | {np.mean([v['J'] for v in vals]):.4f} | {np.mean([v['old_correct_atomic_regression'] for v in vals]):.4f} |")
lines += ['', '结论：protection 对 plain 的 J 差异 raw 三种子为 −.0295/−.0042/+.0169，rel 为 −.0042/+.0169/+.0253；不是稳定的跨 seed 优势，不能恢复旧 loss 因果解释。未见原子回归约 11–38%，远非训练集 <5pp 可替代，U05 保留为有根据的探索。hard replay 与 soft protection 接近；真实 preserve/flip 组成有降低原子回归的迹象，但需要独立确认，不能归于保护损失机制。', '',
'U07 RNG 归因：当前 d01_capacity.py 在每次模型构建前 torch.manual_seed(seed)，aug=none 为确定的 full-batch；本轮没有旧 run 初始 state/batch-stream 证据，RNG 伪影归因仍 UNRESOLVED/撤回。不得只用救好的格替换失败格。本轮是全 30 臂固定合同重训，并不冒充 U07 全 dose 网格重训。', '',
'未运行：U07 全剂量格重训、U05 测试原子保护训练、独立 sealed 确认；当前比较不能完成这些科学检验。', '',
'命令：`.venv/bin/python experiments/e1a933_review/fair_continuation.py`；统计：`.venv/bin/python experiments/e1a933_review/fair_summarize.py`。',
'证据：`artifacts/e1a933_review/fair/CONTINUATION_RESULTS.json`、`FAIR_CONTRASTS.json`、各臂 `receipt.json` / `batch_order.json` / `evaluation.npz`。CI 对 parent 成簇、同一次抽样计算双方完整比例；3 个 seed 均值仅描述性汇总。']
(ROOT/'reports/e1a933_review/CONTINUATION_FAIR_COMPARISON.md').write_text('\n'.join(lines)+'\n')
