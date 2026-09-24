"""Read completed O05/O06 GPU receipts; report without changing frozen claims."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--football',type=Path);ap.add_argument('--vlm',type=Path);ap.add_argument('--reports',type=Path,default=ROOT/'reports/e1a933_review');a=ap.parse_args();a.reports.mkdir(parents=True,exist_ok=True)
 if a.football:
  r=json.loads((a.football/'summary.json').read_text());receipt=json.loads((a.football/'receipt.json').read_text());assert receipt['status']=='MATRIX_COMPLETE' and len(receipt['completed'])==12
  for enc in ['raw','typed']:
   for seed in [11,23,47]:
    one=r['arms'][f'{enc}_static_s{seed}'];two=r['arms'][f'{enc}_change_s{seed}']
    for key in ['initial_sha256','batch_order_sha256','train_examples','steps']:assert one[key]==two[key],(enc,seed,key)
  lines=['# O05 GPU真实学习比较','',f"状态：MATRIX_COMPLETE，12/12训练臂；实际GPU运行 {r['wall_seconds']:.1f}s，峰值allocated CUDA {r['peak_cuda_allocated_bytes']/2**30:.3f}GiB。固定合同见 `O05_O06_GPU_FROZEN.md`，输入、批次和初始化配对hash验收通过。",'',
   '测试为J03WN1单场2794个自然轨迹快照与134个受控E quartet；训练/开发是另外两场。三seed不是三场独立重复；自然路径和受控路径分别统计。当前数据角色完整，缺测由明确因果遮挡产生，不冒充自然缺测机制。','',
   '| 方法 | test accuracy | 真实转移匹配/210 | 漏报 | 误报警 | median lag秒 |','|---|---:|---:|---:|---:|---:|']
  for name,v in r['baselines'].items():
   t=v['natural_temporal'];lines.append(f"| {name}+oracle | {v['accuracy']:.4f} | {t['matched_n']}/{t['true_transition_n']} | {t['missed']} | {t['false_alarm_n']} | {t['lag_seconds_median']} |")
  lines+=['','| 编码 | 增广 | accuracy 3seed | J(A,B,AB) 3seed |','|---|---|---|---|']
  for enc in ['raw','typed']:
   for aug in ['static','change']:
    arms=[r['arms'][f'{enc}_{aug}_s{s}']['metrics'] for s in [11,23,47]]
    ac='/'.join(f"{v['accuracy']:.4f}" for v in arms);js='/'.join(f"{v['controlled_E']['J_A_B_AB']:.4f}" for v in arms);lines.append(f'| {enc} | {aug} | {ac} | {js} |')
  best=max(v['metrics']['accuracy'] for v in r['arms'].values());cv=r['baselines']['cv']['accuracy']
  lines+=['',f'最好的单训练臂自然快照准确率 {best:.4f}，恒速估计+解析基线 {cv:.4f}；这是描述性对照，不按test挑选主模型。'+(' 学习臂未超过该简单因果基线，不能把足球提升为结构迁移主贡献。' if best<=cv else ' 个别臂高于基线仍需跨更多比赛独立确认，不能用三seed替代比赛复制。'),'',
   '静态/变化臂同编码同seed拥有相同初始化、样本总量和batch序列；变化只用train角色的历史一致A/B/AB编辑。选择checkpoint只依据自然dev BCE。typed参数18177、raw16769，不称严格参数量匹配。',
   'JSON逐臂包含自然转移误报、漏报、提前/滞后全部秒数、受控E原子正确分母和条件AB错误。解析全坐标基线等于真值定义，不能用“神经网络未超oracle”作否定。','',
   '边界：真实传球成功/未发生传球结果未测试；自然缺测机制学习未测试；仅一个test match，不形成广泛足球泛化结论。',f'证据目录：`{a.football}`；输入合同：`artifacts/e1a933_review/football_learning_frozen/manifest.json`。']
  (a.reports/'O05_LEARNED_PARTIAL_TRACKING_GPU.md').write_text('\n'.join(lines)+'\n')
 if a.vlm:
  r=json.loads((a.vlm/'summary.json').read_text());receipt=json.loads((a.vlm/'receipt.json').read_text());assert receipt['status']=='MATRIX_COMPLETE' and len(receipt['completed'])==272
  m=json.loads((a.vlm/'run_manifest.json').read_text());lines=['# O06 原生VLM外部桥结果','',
   f"状态：MATRIX_COMPLETE，272/272独立图像请求。模型 `{m['model']}`，revision `{m['revision']}`；transformers `{m['transformers']}`，greedy max_new_tokens16，固定prompt与解析，无按准确率换模型。",'',
   f"64个quartet来自已复用的corrected visual bank，不是fresh confirmation；另16个基本几何sanity。J(A,B,AB)={r['J_A_B_AB']:.4f}，4状态J={r['J_base_A_B_AB']:.4f}；parent bootstrap 95%CI={r['J_95CI_parent_bootstrap']}。",'',
   f"原子A/B准确率={r['atomic_A_accuracy']:.4f}/{r['atomic_B_accuracy']:.4f}，双原子正确分母={r['atomic_both_correct_n']}/64；条件AB错误率={r['conditional_AB_error_given_atoms_correct']}。其中可解析模型错误={r['conditional_AB_model_error_n']}，AB非响应={r['conditional_AB_nonresponse_n']}。原子能力不足时不可解释为组合专属机制。",'',
   f"sanity准确率={r['sanity_accuracy_all_requests']:.4f}（n={r['sanity_n']}）；parse failure={r['parse_failure']}、拒答={r['refusal']}。总体准确率/J保留全部分母，将非响应视为未答对，并单独披露，不能静默删除。",'',
   f"峰值allocated CUDA={r['peak_cuda_allocated_bytes']/2**30:.3f}GiB；本次调用推理墙钟={r['wall_seconds_this_invocation']:.1f}s（resume场景不含之前调用）。首次8项输入/解析收据单独留存，准确率不作为继续/模型选择门槛。",'',
   '边界：这是单一3B原生VLM对合成64像素上采样448观测的外部桥，不是所有VLM/自然视觉场景结论。没有可比置信度，不套用小MLP logit P1。每状态独立请求，未输入其他状态、标签或历史回答。',f'证据：`{a.vlm}` 下responses.jsonl、run_manifest.json、summary.json、receipt.json；冻结图像/提示见 `artifacts/e1a933_review/vlm_frozen_272/manifest.json`。']
  (a.reports/'O06_NATIVE_VLM_GPU.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
