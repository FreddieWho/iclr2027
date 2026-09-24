"""Interpret immutable N03 runs and measure frozen-head noise on saved dev latents."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
torch.set_num_threads(2)
receipt = json.loads((a.results / 'receipt.json').read_text())
assert receipt['status'] == 'MATRIX_COMPLETE' and len(receipt['completed']) == 6
assert hashlib.sha256(a.data.read_bytes()).hexdigest() == receipt['data_sha256']
data = np.load(a.data)
labels = torch.tensor(data['dev_labels'])
rows = []
for arm in receipt['arms']:
    for seed in receipt['seeds']:
        folder = a.results / f'{arm}_s{seed}'
        result = json.loads((folder / 'result.json').read_text())
        assert hashlib.sha256((folder / 'model.pt').read_bytes()).hexdigest() == result['checkpoint_sha256']
        checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        state = checkpoint['state']
        head = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
        head.load_state_dict({k[len('relation.'):]: v for k, v in state.items() if k.startswith('relation.')})
        head.eval()
        perms = state['rel_perms']
        latent = np.load(folder / 'dev_aux_predictions.npz')
        history = json.loads((folder / 'history.json').read_text())

        def classify(x):
            with torch.no_grad():
                return head(x[:, perms]).mean(1).squeeze(-1)

        predicted = torch.tensor(latent['prediction'])
        clean = classify(predicted)
        bce = float(F.binary_cross_entropy_with_logits(clean, labels))
        recorded_bce = history[result['best_epoch'] - 1]['dev_BCE']
        assert abs(bce - recorded_bce) < 1e-5, (bce, recorded_bce)
        generator = np.random.default_rng(26092403)
        noise = generator.standard_normal((16, *predicted.shape)).astype(np.float32)
        sensitivity = []
        for sigma in [0., .01, .03, .1, .3, 1.]:
            accuracies = []
            changes = []
            for sample in noise:
                z = classify(predicted + sigma * torch.tensor(sample))
                accuracies.append(float(((z > 0) == labels.bool()).float().mean()))
                changes.append(float(((z > 0) != (clean > 0)).float().mean()))
            sensitivity.append(dict(sigma=sigma, mean_dev_accuracy=float(np.mean(accuracies)),
                                    mean_prediction_change=float(np.mean(changes))))
        oracle = json.loads((folder / 'oracle_replacement_DIAGNOSTIC.json').read_text())
        rows.append(dict(arm=arm, seed=seed, J=result['J'], best_epoch=result['best_epoch'],
                         dev_matched_mse=float(latent['matched_error'].mean()),
                         oracle_same_head_J=oracle['J'], noise_sensitivity=sensitivity,
                         checkpoint_sha256=result['checkpoint_sha256']))
paired = json.loads((a.results / 'paired_analysis.json').read_text())
summary = dict(status='COMPLETE', results=str(a.results), data_sha256=receipt['data_sha256'],
               rows=rows, paired=paired,
               noise_contract='Post-training descriptive diagnostic on 707 saved dev latents; isotropic Gaussian noise in standardized 10d relation space, 16 fixed draws. No image perturbation or threshold tuning; no physical-validity claim.',
               decision='Weak gain versus generic bottleneck; no consistent gain over strong visual baselines; oracle substitution fails and geometry error is not comparable to N01. No confirmed geometric-decision mechanism.')
(a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['# N03 决策瓶颈结果', '', '六臂GPU训练及完整产物已回收校验。几何监督相对同容量通用瓶颈三seed的J均提升，但对强视觉基线没有稳定收益，不能确认为几何决策机制或主方法。', '',
         '| 瓶颈 | seed | J | dev matched MSE | 真几何替换同head J（诊断） |', '|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['arm']} | {row['seed']} | {row['J']:.4f} | {row['dev_matched_mse']:.4f} | {row['oracle_same_head_J']:.4f} |")
lines += ['', '## 配对解释', '']
for name, value in paired.items():
    seed_text = '; '.join(f"{r['seed']}: {100*r['J_difference']:+.2f}pp, parent CI [{100*r['parent_bootstrap95'][0]:+.2f}, {100*r['parent_bootstrap95'][1]:+.2f}]" for r in value['seeds'])
    lines.append(f"- {name}: 均值 {100*value['mean_J_difference']:+.2f}pp；{seed_text}。")
lines += ['', '几何瓶颈的dev MSE为0.60–2.62，明显高于N01 matched的0.118–0.297，因此B/C不是“几何误差相当”的干预；不能将差异归因于信息被迫进入决策。三个几何瓶颈将真几何代入冻结head时J均为0，这表明head没有在真几何输入下形成正确关系计算；替换也改变了输入分布，不能据此将错误完全分解成感知误差。', '',
          '经典颜色分割/PCA像素基线J=114/159=0.7170，高于三个几何瓶颈的描述性点估计；该比较不作独立显著性宣告。所有实验使用159 quartet/74 parent的已暴露bank，不是新确认。', '',
          '## 噪声诊断', '', summary['noise_contract'], '', '| 瓶颈 | seed | sigma=.1决策变化率 | sigma=1决策变化率 |', '|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['arm']} | {row['seed']} | {row['noise_sensitivity'][3]['mean_prediction_change']:.4f} | {row['noise_sensitivity'][5]['mean_prediction_change']:.4f} |")
lines += ['', '不再据此重启架构搜索；本瓶颈未取得可靠任务增益，不触发N03向足球迁移。O05独立的部分tracking学习比较不依赖此路线成功。', '', f"逐臂原始结果/完整修复、迁移及111退化分母：`{a.results}/paired_analysis.json`。本地解释与噪声结果：`{a.out}/summary.json`。"]
Path('reports/e1a933_review/N03_DECISION_BOTTLENECK_REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'status': 'COMPLETE', 'arms': len(rows), 'report': 'reports/e1a933_review/N03_DECISION_BOTTLENECK_REPORT.md'}))
