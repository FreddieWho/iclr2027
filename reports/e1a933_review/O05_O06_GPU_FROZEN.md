# O05 / O06 已冻结 GPU 合同

用户追加授权：本轮剩余 GPU 工作在已有3080Ti 12GB实例完成后再关闭；必要公开权重下载已获总控确认，不新增付费API/实例。远端运行由总控统一调度。本文件是输入与实施合同，不是训练结果。

## O05：受控缺测下的真实tracking关系学习

固定数据 `artifacts/e1a933_review/football_learning_frozen/`；入口 `experiments/e1a933_review/football_learning.py`。

- J03WOY训练、J03WMX开发、J03WN1测试；每候选取事件前2秒自然轨迹，.2秒一帧。自然快照6314/6831/2794。每个.4秒history逐40ms检查身份与时间连续，跨缺测/半场断开，不用未来位置补输入。
- 接球者和一个根据**过去**位置选出的通道最近防守者遮挡.2秒；其余实体当前可见。隐藏坐标取过去最近值，速度取更早两帧；无任意高斯噪声。该人工遮挡是受控压力测试，不冒充原生缺测机制。
- raw MLP / permutation-invariant typed relation网络 × static / change × seeds11/23/47，共12臂。300 epochs、Adam .001、batch512，每臂7076训练样本、4200steps；同编码同seed初始化和批次顺序一致。参数量分别记录，不称严格容量匹配。
- static追加762个原始快照重复；change追加相同parent的762个A/B/AB状态。受控位置编辑应用到整个.4秒history，确保变化可观测，不对隐藏当前状态单独改标签。mask identity在一组受控quartet内固定。
- checkpoint每10epochs按自然dev BCE最小选；不按test/E表现选。test有134个受控E quartet；自然轨迹和受控E分开报告。
- 比较last-value+解析几何、恒速状态估计+解析几何；完整坐标oracle是标签定义。自然轨迹分类转移按.2秒网格统计真假报警和提前/滞后秒数（同目标类别、±.6秒一对一匹配），缺帧断开。

测试只来自一场比赛，三seed不是三场独立复制；多个候选、时间帧同事件相关。结果是探索性真实tracking代理，不是反事实传球成功，也不能宣称广泛足球泛化。

```bash
python experiments/e1a933_review/football_learning.py --data artifacts/e1a933_review/football_learning_frozen --out NEW_OUTPUT --device cuda
```

预计<2GB显存，实际峰值由GPU运行记录。完成 `receipt.json` 包含 `status=MATRIX_COMPLETE` 与12项`completed`列表；`summary.json`含完整指标、轨迹和运行预算。中途重启须相同输入/代码/torch合同；已完成目录拒绝覆盖。

## O06：一个小型原生VLM外部桥

固定 `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3`，官方文件约7.52GB。原生视觉语言生成模型，非CLIP分类替代。官方Qwen Research License用于本轮学术测试；独立venv `transformers==4.51.3`、`accelerate==1.6.0`，复用原torch，SDPA，不引入flash-attn。来源：[官方模型](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[官方使用说明](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md)、[许可](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE)。

输入 `artifacts/e1a933_review/vlm_frozen_272/` 已冻结：本轮corrected visual bank里seeded permutation选64个不同parent，每个base/A/B/AB，共256图；另16个明显交叉/不交叉 sanity。没有根据模型输出筛选。全部native64同源观测统一uint8及bilinear448；上采样不增加信息。此bank已用于本轮视觉评估，属于exploratory reused bank，不称fresh confirmation。

统一prompt询问红蓝线段中心线是否严格内部相交，固定YES/NO解析。每个状态独立新请求，输入中没有标签/parentID/其他状态/先前回答。BF16、batch1、greedy、max_new_tokens16。保存完整响应、tokenIDs、parse failure/拒答和模型错误分离；输出J、4状态J、原子分母和条件AB错误分解。先8项记录输入/解析可用性，不以准确率换模型或prompt；不设85%门槛，不强套MLP-logit P1。

```bash
vlm_venv/bin/python experiments/e1a933_review/vlm_native.py --data artifacts/e1a933_review/vlm_frozen_272 --model-path SNAPSHOT_PATH --out NEW_OUTPUT
```

完成 `receipt.json` 包含 `status=MATRIX_COMPLETE` 和272项`completed`列表；原文`responses.jsonl`支持显式`--resume`且校验输入/checkpoint/版本/源码hash。这是单一3B模型测试，不推广成所有VLM结论。

本地实施检查：raw/typed前向反向、typed防守者置换不变、causal mask取历史坐标、YES/NO/parse/refusal分类通过；完整GPU矩阵仍以远端receipt为准。
