#!/usr/bin/env python3
"""Summarize the predeclared three-seed frozen N02 diagnostic."""
import csv, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import torch, torchvision
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/e1a933_review/N02_sampling_summary'
OUT.mkdir(parents=True,exist_ok=False)
rs={}; samples=[]
for seed in [803,805,806]:
    d=ROOT/f'artifacts/e1a933_review/N02_sampling_seed{seed}_n64'
    r=json.loads((d/'result.json').read_text());rs[seed]=r
    sm=np.load(d/'samples.npz');samples.append(sm)
    assert r['n_selected']==64
    for cp in r['checkpoints']:assert cp['stored_seed']==seed
    pp=np.load(d/'predictions.npz')
    for k,v in r['metrics'].items():
        assert pp[k].shape==(64,4) and np.isfinite(pp[k]).all()
        assert v['CCM_denom']<=64
        ok=(pp[k]>0)==pp['labels']
        assert np.isclose(ok[:,1:].all(1).mean(),v['J'])
        assert int((ok[:,1]&ok[:,2]).sum())==v['CCM_denom']
    if len(samples)>1:
        for key in ['bank_index','coords','labels','parent_hash','base_images']:
            assert np.array_equal(sm[key],samples[0][key]),key
    # This is a fixed E-quartet bank; do not relabel it as new static-only or flip training.
    yy=sm['labels']; assert ((yy[:,0]==yy[:,1])&(yy[:,1]==yy[:,2])&(yy[:,2]!=yy[:,3])).all()
with (OUT/'sample_manifest.csv').open('w') as f:
    fields=['bank_index','parent_sha256','quartet_coords_sha256','x_image_sha256','A_image_sha256','B_image_sha256','AB_image_sha256']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    sm=samples[0]
    for i,idx in enumerate(sm['bank_index']):
        row=dict(bank_index=int(idx),parent_sha256=str(sm['parent_hash'][i]),quartet_coords_sha256=hashlib.sha256(sm['coords'][i].tobytes()).hexdigest())
        for j,state in enumerate(['x','A','B','AB']):row[state+'_image_sha256']=hashlib.sha256(sm['base_images'][i,j].tobytes()).hexdigest()
        w.writerow(row)
rows=[];pr=[]
for seed,r in rs.items():
    for k,v in r['metrics'].items():rows.append(dict(seed=seed,arm=k,**v,**r['forward_budget'][k]))
    for k,v in r['paired'].items():pr.append(dict(seed=seed,contrast=k,**v))
for path,rr in [('observation_matrix.csv',rows),('paired_statistics.csv',pr)]:
    with (OUT/path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(OUT/'structure_matrix.json').write_text(json.dumps(rs[803]['structure'],indent=2)+'\n')
(OUT/'runtime.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,threads_per_run=2,maximum_simultaneous_runs=2,checkpoint_count=12,total_forward_images=64*4*8*12,total_measured_forward_seconds=sum(x['forward_seconds'] for x in rows)),indent=2)+'\n')
r=rs[803];lines=['# N02 Sampling Access Report','', '**状态：已完成有界、零重训前向诊断；旧 bank 再分析，机制未决。**', '',f"三个已有 seed（803/805/806），每 seed 四个冻结 checkpoint（64/224 训练 × pretrained/random）。使用相同的 64 个 quartet，共 {r['n_unique_parent']} 个不同 base-coordinate parent；原 bank 有 {r['n_bank']} 行，其中 {r['n_boundary_eligible']} 行满足平移不裁掉前景条件。rng2924 选样，未依据模型结果选样。seed803 random 的塌缩保留；补入805/806不改变首批样本。",'', '## 观测和计算合同','', '- A：原生产 renderer 的 64×64 RGB，lw=2，即整数方形笔刷宽5像素。B：恰为A的双线性224上采样（align_corners=False），不增加观测信息。每个状态固定 rng=50000+旧bank行号；四状态共享背景 nuisance。', '- 相位：先在64像素光栅上平移(0,0)、(1,0)、(0,1)、(1,1)，原背景填充，再分别按64/224输入。仅选原图最右列/最下行无前景的quartet，因此不丢失前景；这是平移敏感性探针，不能独立确定aliasing。', '- 原生224 C及低通下采样D：NOT_RUN。现有renderer依赖整数坐标/整数线宽，直接改IMG/lw不能保证相同物理笔刷、量化和遮挡合同；本轮没有用未经审计的新版render替代。', '- 低stride/抗混叠结构：NOT_RUN；四格完整重训、关键格pretrained/random公平开发、新parent确认：NOT_RUN。本轮没有优化器步骤。', '- 本bank入选行均为E quartet (y0=yA=yB≠yAB)；指标static是同一quartet的x准确率，不是独立static-only训练模型。逐原子、AB、J、CCM分母和110→111/111退化见CSV。', '', '## 同一checkpoint输入切换（phase00）','', '| seed | checkpoint | J@64 | J@224 | ΔJ row (224−64) | ΔJ parent | parent bootstrap 95% CI |', '|---|---|---:|---:|---:|---:|---|']
for seed,r in rs.items():
    for c in r['checkpoints']:
        name=c['name'];pa=r['paired'][name+'__224_minus_64'];ci=pa['parent_bootstrap_95pct']
        lines.append(f"| {seed} | {name} | {r['metrics'][name+'__input64_phase00']['J']:.4f} | {r['metrics'][name+'__input224_phase00']['J']:.4f} | {pa['delta_J']:+.4f} | {pa['parent_mean_delta_J']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
switch64=[rr['metrics'][f'train64_{mode}__input224_phase00']['J'] for rr in rs.values() for mode in ['pretrained','random']]
lines+=['', '64训练的6个checkpoint直接换224输入后，phase00的J分别为 '+str(switch64)+'。这与“只把输入放大就改善同一模型”不相容；历史224结果包含在224输入下训练这一变化，不能由纯前向缩放替代。seed803的224-random零J保留，没有删除塌缩seed。', '', 'row ΔJ以64个quartet为权重；parent ΔJ先在每个parent内均值，再对58个parent等权。CI对应parent ΔJ，为每个训练checkpoint条件下4000次parent bootstrap区间；未多重比较校正，64 quartet/58 parent子集不代表整个任务分布，也不计作三次独立任务验证。', '', '## 平移相位敏感性','', '| seed | checkpoint | J@64 四相位 min–max | J@224 四相位 min–max |', '|---|---|---|---|']
for seed,r in rs.items():
    for c in r['checkpoints']:
        name=c['name'];ranges=[]
        for res in (64,224):
            vv=[r['metrics'][name+f'__input{res}_phase{ph}']['J'] for ph in ['00','10','01','11']]
            ranges.append(f'{min(vv):.4f}–{max(vv):.4f}')
        lines.append(f'| {seed} | {name} | {ranges[0]} | {ranges[1]} |')
lines+=['','## 解释与未决边界','', '固定checkpoint的输入尺寸切换和在另一尺寸上重新训练是两个不同实验。本结果只识别前者；不同训练尺寸checkpoint的差距混合了优化、早期采样、特征/归一化统计与预训练适配。不得把输入切换效应直接当重训收益、不得排除ImageNet先验。', '', '相位曲线证明对具体光栅平移的敏感性，不能证明早期下采样是因果机制。尚无低stride消除放大优势的干预，也没有在新parent命中的预先固定结构预测，所以N02保持未决。', '', '这是已暴露旧bank的子集再分析。训练parent重叠的完整审计由共同数据路线负责，本诊断不声称未见parent泛化。原图裁剪/遮挡和不同状态光栅相同的问题没有全面解决；入选样本中“不同label而整图完全相同”的parent索引为 '+str(rs[803]['identical_raster_label_contradiction_parent'])+'，这只是精确相同光栅检查，不是完全可观测性证明。', '', '## 计算预算与复现','', '使用CPU，每进程2线程，最多2进程重叠（共4线程）。共12个checkpoint，每个8个输入条件、256幅图；24,576次图像前向。实际耗时见runtime.json与observation_matrix.csv；这是本机诊断计时，不是公平速度benchmark。Conv/Linear MACs不含BN、ReLU与pool开销；224的中间网格和MACs在structure_matrix.json。', '', '原实际seed803命令（首批固定803；当前脚本--seed默认803）：', '```bash', 'python experiments/e1a933_review/sampling_access.py --out artifacts/e1a933_review/N02_sampling_seed803_n64 --n 64 --threads 2', 'for seed in 805 806; do', '  python experiments/e1a933_review/sampling_access.py --out artifacts/e1a933_review/N02_sampling_seed${seed}_n64 --n 64 --threads 2 --seed "$seed"', 'done', 'python experiments/e1a933_review/sampling_report.py', '```', '', '输出目录exist_ok=False；复跑请改用新的输出目录。脚本仅从已有checkpoint加载weights，无权重下载。每seed result.json保存源commit、bank/checkpoint/sample/prediction/renderer/hash、固定信息张量hash、命令和配对统计；samples.npz保存原64图与坐标，predictions.npz保存全部逐parent四状态logits。summary目录保存跨seed相同样本检查后的矩阵。', '', '未增加外部生信数据，无需新增bioinf-data-index条目。']
path=ROOT/'reports/e1a933_review/N02_SAMPLING_ACCESS_REPORT.md';path.write_text('\n'.join(lines)+'\n')
print(path)
