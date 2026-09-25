# Route 2 remote GPU runbook

状态：`DATA_READY_BLOCKED_GPU`，等待用户提供已授权实例的登录方式。未保存或请求任何凭据，也未主动连接远程主机。

## 已冻结输入

- manifest：`artifacts/e832_focus/route2/gpu_bundle/manifest.json`
- data：`artifacts/e832_focus/route2/data/data.npz`
- data SHA256：`8e5128e51499b32bc5082a010a27d0e19542a1979325c31969780594f1e221ee`
- 4 arms × seeds `803,805,806`
- 224×224、batch32、FP32、20 epochs
- 12 formal runs + 1 aggregate report
- 当前没有视觉结果，formal claim 为 `none`

## 登录后先做

1. 确认CUDA可见、显存至少8GB（推荐12GB）。
2. 将仓库和 `artifacts/e832_focus/route2/data/` 同步到远端工作目录。
3. 先运行：

```bash
python experiments/e832_focus/route2_visual/gpu_preflight.py \
  --data artifacts/e832_focus/route2/data \
  --out artifacts/e832_focus/route2/gpu_bundle
python -m unittest discover -s experiments/e832_focus/route2_visual -p 'test_*.py' -v
```

4. 确认preflight仍为 `DATA_READY_BLOCKED_GPU` 或CUDA可用，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/e832_focus/route2_visual/gpu_run.py \
  --manifest artifacts/e832_focus/route2/gpu_bundle/manifest.json \
  --data artifacts/e832_focus/route2/data \
  --out artifacts/e832_focus/route2/gpu_run
```

## 回收与审阅

完成后回收：

- `gpu_run/status.json`
- `gpu_run/results.json`
- 12个 `*/model.pt`、`history.json`、`result.json`、`predictions.npz`
- `exposure_indices.json`

先核对manifest/data hash、每个seed/arm是否完整、是否出现OOM或NaN，再更新 `CLAIM_LEDGER`。在独立review前，状态保持 `COMPUTED_NOT_YET_INTERPRETED`，不能写视觉方向。
