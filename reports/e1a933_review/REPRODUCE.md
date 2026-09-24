# 复现入口（执行中）

基准：e1a933e6b32dd07c6895c7925cbccd9004da7f54；当前修改尚未提交。
代码变更将在本轮SOURCE_RECEIPT记录内容哈希；旧权重与bank不覆写。

```bash
cd /home/huyudi/012_conference/iclr2027
python -m pytest -q experiments/e1a933_review/test_data_repairs.py
```

上面只验证R02/R04生产函数，不是科学复现。各工作组的真实运行命令和产物将在执行回执完成后登记；当前不提供假设已经运行的命令。多seed运行使用for循环或脚本明确列表，不使用`--seed {11,23,47}`伪装多个运行。

## 已执行的公平续训和readout再分析

```bash
.venv/bin/python experiments/e1a933_review/fair_continuation.py --out /tmp/e1a933_fair_reproduce_01
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_objective_check.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/e1a933_review/readout_refit.py --output artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY
python experiments/e1a933_review/readout_report.py --run artifacts/e1a933_review/readout_refit_FRESH_DIRECTORY --report reports/e1a933_review/O04_readout_FRESH.md
```

readout实际目录为readout_refit_20260924；FRESH_DIRECTORY是重新执行时的新输出目录，不应覆盖旧结果。公平续训脚本的实际写入/恢复行为见专报，不要将重跑旧bank称新确认。

## 其他已执行路线

- 数据/跨任务：以本目录数据组命令专报及U10/R03/R04/O01报告为准；已有目录不得覆盖。
- 足球/前沿：WORKER_C_RUN_COMMANDS.md列真实原命令及新目录重跑方法，20项针对性检查。
- 视觉合同：`python experiments/e1a933_review/vision_regression.py`；旧factorial：`python experiments/e1a933_review/vision_factorial.py`。
- N02：见N02_SAMPLING_ACCESS_REPORT.md的三seed for循环和新目录命令。
- 论文：PAPER_PATCH_AUDIT.md记录TinyTeX及新PDF，旧paper/main.pdf未替换。

## 已停止的 CPU 兜底训练（保留命令供审计，勿重启）

```bash
python -u experiments/e1a933_review/vision_run.py --out artifacts/e1a933_review/vision_canonical224_v2 --epochs 20
```

**该 CPU 兜底任务已于 2026-09-24 14:53Z 按用户指示停止**（SIGTERM，2 秒内退出并核实无残留进程）：它在远端 GPU 矩阵完成、回收、校验并进稿后已无用途，而本项目规定 CPU/GPU 结果不得混表，故即使跑完也不进任何报告数字。
停止时完成 **19/21 臂**（`matched_s806` 仅有 history.json、无 model.pt；`shuffled_matched_s806` 未启动），**未删除任何产物**。
现行状态：`artifacts/e1a933_review/vision_canonical224_v2/receipt.json` = `STOPPED_BY_USER`；
原始 RUNNING 回执按字节保留为 `receipt.RUNNING_asof_stop.json`（sha256 `8dcc9e7f…`）；停止记录见 `TERMINATION_RECEIPT.json`。

报告中的视觉矩阵是**远端 GPU** 那一套（`remote_vision_20260924_ssh30891` + `gpu_finish_20260924`），不是这个 CPU 目录。


## 远端 GPU 执行（已完成）
用户部署实例并授权 21 臂执行，具体命令/版本/源和数据哈希见 `REMOTE_GPU_EXECUTION.md`。**全部 21 臂已完成、回收、逐文件 SHA256 校验并已进稿**；自动回收基于 inotify 文件事件等待。不可再次启动同目录，不可把 CPU/GPU 臂混为同矩阵。追加矩阵（N03/O05/N02/O06）状态见 `GPU_RELEASE_RECEIPT.md`。

## 本轮GPU补完（2026-09-24）

远端冻结argv、输入/输出、环境与监督器合同分别保存在`artifacts/e1a933_review/gpu_finish_20260924/{n03,football_v2,vlm,sampling}/`的task/spec文件；不含登录密码。四项使用同一GPU文件锁串行计算。N03/O05/O06已完成并校验；N02仍运行，完整矩阵不能由当前部分结果替代。

本地结果解释命令：
```bash
.venv/bin/python experiments/e1a933_review/n03_finish_analysis.py --results artifacts/e1a933_review/gpu_finish_20260924/n03/extracted/n03_results --data artifacts/e1a933_review/vision_cuda_bundle/data.npz --out /tmp/n03-interpretation-reproduction
python experiments/e1a933_review/football_vlm_summarize.py --football artifacts/e1a933_review/gpu_finish_20260924/football_v2/extracted/football_results_v2 --vlm artifacts/e1a933_review/gpu_finish_20260924/vlm/extracted/vlm_results
```

N03噪声诊断仅对已保存dev relation latents经过冻结head进行，不需GPU。原生VLM固定revision、transformers4.51.3、torch2.5.1+cu124、BF16；两个官方权重hash和镜像来源在远端VLM_MODEL.json及脚本remote_vlm_mirror_download.py。

本地自动完成处理器`gpu_finish_release.py`基于inotify，无远端定时查询；仅当全部GPU产物本地校验通过才产生SAFE_TO_SHUTDOWN，不把释放GPU等同科学全部完成。
