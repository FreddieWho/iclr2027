# DeepSeek C1 替换尝试：两个模型均未解除阻塞

当前结论：`NO_GO / TECHNICAL`。按用户偏好先尝试 `deepseek-v4.1-flash`，失败后尝试 `deepseek-v4-flash`，各一次；两者均耗尽输出且无可见memory。完整P0、P1/P2与科学假设验证仍未完成。没有重试、截断、隐藏reasoning回填或科学评分，不自动改变reasoning档位。

## 本轮合同与证据

授权记录：`configs/deepseek_switch_authorization_20260912.json`。模型替换仅用于P0技术校准，不是按科学效果选模型。R1保持OpenCode Go / GLM原合同，未调用或修改。所有C1请求均为同一原始完整history `syn-20260910-00000`；输入hash仍为 `80c3a2f04bb8a2a77358ceb496c616262af0772a24f7191e63505e5e3e56de48`，未读取未来问题或gold。

使用前轮已有的中性长度校准提示 `prompts/compress_neutral_calibration.txt`，hash `a736947d980951aea4852903ff70510f3f30021b138f5ed13f0fd1e01a42248f`。它只规定memory上限和内容保留要求，不要求精确填满。不是与旧GLM原始提示完全等价的实验；这里没有做跨模型效应对比。

两个模型均通过同一Go Chat Completions endpoint调用：`thinking.type=enabled`、`reasoning_effort=max`、`top_p=0.95`、总输出上限24,576、memory上限1,024、无tools/web。根据DeepSeek官方说明，thinking模式不支持temperature控制，本轮不发送temperature并记录provider默认，不能称与GLM temperature=0.2等价。

V4.1官方编码规范将字符串max映射至最高数值100；Go网关的内部映射未暴露。记录“请求max”，不把API接受参数当作内部档位已独立验证。来源：[V4.1固定版本编码说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md)、[DeepSeek thinking参数](https://api-docs.deepseek.com/guides/thinking_mode/)。

## 实测结果

| 模型 | provider输入 | provider输出/reasoning | 可见memory | finish | 费用保守上界 |
|---|---:|---:|---:|---|---:|
| deepseek-v4.1-flash | 32,903 | 24,577 / 24,577 | 0 | length | $0.03936330 |
| deepseek-v4-flash | 32,903 | 24,576 / 24,576 | 0 | length | $0.03936210 |

两次返回model字段均与请求ID一致。V4.1于2026-09-11T16:22:57.305097+00:00开始，耗时103.277秒；V4于16:26:03.205391+00:00开始，耗时103.688秒。北京时间均为2026-09-12。两个tokenizer对该history均计32,693 tokens，完整request正文均32,873；tokenizer文件hash不同，并非混用。

V4.1的reported output比请求上限多1 token，单列为provider预算越界；不将其忽略、不当有效完成。两次content均为空字符串，因此无可供reader使用的memory。1,024预算未通过，2,048和8,192能力探针按预定门控 `NOT_RUN`；本轮最多3次中的剩余1次未使用，不为用完额度增加请求。

## Tokenizer与实现

- V4.1：`deepseek-ai/DeepSeek-V4.1-Flash`，完整revision `dba1be0a40aa45a94ad051997016db3960a90277`，tokenizer.json SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。
- V4：`deepseek-ai/DeepSeek-V4-Flash`，完整revision `60d8d70770c6776ff598c94bb586a859a38244f1`，tokenizer.json SHA256 `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`。
- 旧目录 `work/tokenizers/deepseek-v4.1-flash-60d8d707` 实际来自V4，不能作为V4.1 tokenizer；旧目录保持不动，新目录按真实模型分别下载并冻结。
- 按HF CLI技能先尝试CLI；本机Typer/Click冲突后使用已有Hub SDK回退，仅下载公开tokenizer/说明，未下载权重、未修改全局依赖。
- 修复 `hf_json` 分支明确使用 `add_special_tokens=False`，并为探针按角色选择tokenizer。新增回归覆盖特殊token排除，当前51项离线测试通过。未宣称完整混合C1/R1正式runner已验证。
- session隔离：V4.1本次为独立新ID `iclr-memory-pilot/b1024`；V4回退显式使用含模型名的不同ID，防止两个同名b1024目录误用同一session。每次manifest保存实际ID/code/prompt/config hash，未重跑旧请求。

## 费用与状态

共享账本仍为 `work/p0_repair_accounting_v1/cost_ledger.sqlite`，没有清空旧费用。使用Go公布的DeepSeek较高时段费率input/output/cache=$0.30/$1.20/$0.006每百万tokens作保守估算，并预留完整65,536输入和24,576输出。实际usage已测，但美元数字是保守上界，不冒充provider账单；来源：[OpenCode Go价格](https://opencode.ai/docs/go/)。

本轮新增2次请求，费用上界$0.07872540，低于本轮$0.15和DeepSeek $3上限。连同历史保守占用$0.16927795，P0累计$0.24800335，剩余$0.25199665。无后台API作业。累计21次尝试包含历史不确定请求，不等同确认送达次数。

最新机器证据：`artifacts/deepseek_switch_result.json`。每个模型的 `work/p0_deepseek_switch_20260912_v1/<model>/` 保存preflight、实际request config、result、cache和费用快照。模型列表保存解析后的JSON及其canonical digest，同时记录原HTTP响应字节的SHA256；两类hash不混称。

结论仅限这些模型/输入/提示/输出额度的技术失败。它提示阻塞不只出现在GLM，但不能证明所有模型都不能完成任务，也不能证明max是唯一根因。若下一步调整C1 reasoning，必须由用户明确修改此前“不降低reasoning”的约束；新合同须在P1前对所有路径统一冻结，不能从失败或隐藏reasoning中拼出科学结果。
