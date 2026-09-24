# Independent CUDA FP32 vision matrix

Planning artifact only. No instance rental or remote execution is authorized by this bundle.
Keep the current CPU run running. Run the complete 3 seeds × 7 arms on one GPU backend; never mix CPU and GPU arms/seeds into the primary matrix.

Same frozen data, seed803/805/806, 20 epochs, batch32, Adam3e-4, native64 bilinear224, dev-BCE checkpoint selection, FP32. CPU preprocessing stays identical; GPU training has TF32 and AMP disabled, deterministic algorithms enabled. Backend numerics and runtime can still differ. Cached ImageNet weights included, no model download required. No CPU trained checkpoints are included.

From this directory on an authorized provisioned GPU host:
```bash
python vision_cuda_run.py --bundle . --out /path/to/new_cuda_preflight
```
This defaults to CUDA preflight only: validates payload hashes, device, full224/batch32 forward and matched auxiliary backward, finite gradients, memory and time. A fresh output directory is required. A no-CUDA host writes NOT_RUN_NO_CUDA.

Only after an actual authorization to execute the remote GPU matrix:
```bash
python vision_cuda_run.py --bundle . --out /path/to/new_cuda_matrix --execute
```
This runs the complete21-arm matrix and paired parent-bootstrap analysis. Every output contains backend metadata. No partial CPU/GPU mixing and no silently changing batchsize/precision after OOM; a failed preflight must be reported. Existing output directories are never overwritten. Check scientific conclusions separately after MATRIX_COMPLETE.

Environment is recorded in manifest.json. Matching Python/PyTorch/torchvision/NumPy builds on a provisioned CUDA-capable host remain untested. GPU memory, real throughput, numeric CPU-CUDA equivalence, driver compatibility, and remote durability are NOT_RUN on the current no-GPU machine. The CPU port check only tests the refactored code path on CPU.
