# O06 原生VLM外部桥结果

状态：MATRIX_COMPLETE，272/272独立图像请求。模型 `Qwen/Qwen2.5-VL-3B-Instruct`，revision `66285546d2b821cf421d4f5eb2576359d3770cd3`；transformers `4.51.3`，greedy max_new_tokens16，固定prompt与解析，无按准确率换模型。

64个quartet来自已复用的corrected visual bank，不是fresh confirmation；另16个基本几何sanity。J(A,B,AB)=0.0000，4状态J=0.0000；parent bootstrap 95%CI=[0.0, 0.0]。

原子A/B准确率=0.2031/0.2031，双原子正确分母=13/64；条件AB错误率=1.0。其中可解析模型错误=13，AB非响应=0。原子能力不足时不可解释为组合专属机制。

sanity准确率=0.5000（n=16）；parse failure=0、拒答=0。总体准确率/J保留全部分母，将非响应视为未答对，并单独披露，不能静默删除。

峰值allocated CUDA=7.222GiB；本次调用推理墙钟=57.8s（resume场景不含之前调用）。首次8项输入/解析收据单独留存，准确率不作为继续/模型选择门槛。

边界：这是单一3B原生VLM对合成64像素上采样448观测的外部桥，不是所有VLM/自然视觉场景结论。没有可比置信度，不套用小MLP logit P1。每状态独立请求，未输入其他状态、标签或历史回答。
证据：`artifacts/e1a933_review/gpu_finish_20260924/vlm/extracted/vlm_results` 下responses.jsonl、run_manifest.json、summary.json、receipt.json；冻结图像/提示见 `artifacts/e1a933_review/vlm_frozen_272/manifest.json`。
