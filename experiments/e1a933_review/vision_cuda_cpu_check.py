"""CPU-only equivalence check for CUDA port; this does not validate CUDA kernels."""
import json,sys
from pathlib import Path
import numpy as np,torch
import vision_protocol as original
import vision_cuda_protocol as port
out=Path('artifacts/e1a933_review/vision_cuda_cpu_check');out.mkdir(exist_ok=False)
data=dict(np.load('artifacts/e1a933_review/vision_canonical224_v2/data.npz'))
for k in data:data[k]=data[k][:2] if k.startswith('quartet_') else data[k][:8]
torch.set_num_threads(4);port.DEVICE=torch.device('cpu');checks={}
for arm in ['pretrained_static','pretrained_flip','matched','shuffled_matched']:
    for name,module in [('original',original),('port_cpu',port)]:module.train_arm(data,out/(arm+'_'+name),arm,991,epochs=1,batch=4)
    a=np.load(out/(arm+'_original')/'predictions.npz')['logits'];b=np.load(out/(arm+'_port_cpu')/'predictions.npz')['logits']
    checks[arm]={'max_logit_difference':float(np.abs(a-b).max()),'exact_equal':bool(np.array_equal(a,b))}
    assert np.array_equal(a,b)
(out/'result.json').write_text(json.dumps({'status':'CPU_PORT_EQUIVALENCE_PASS','cuda_validation':'NOT_RUN_NO_GPU','checks':checks},indent=2))
