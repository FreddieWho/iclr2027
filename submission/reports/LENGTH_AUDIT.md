# LENGTH_AUDIT

状态：`NOT_RUN_FORMAL_LENGTH_AUDIT`

## 已核验的输入长度

| 数据 | n history | tokenizer | canonical tokens | 结果 |
|---|---:|---|---:|---|
| synthetic P0 | 8 | `tiktoken:cl100k_base` | 32,722–32,753 | 可用于 calibration，不计推断 |
| LongMemEval cleaned 抽样 | 32 | `tiktoken:cl100k_base` | 112,845–118,608 | 完整保留；超过默认 32K 施工配置 |

`configs/pilot.json` 的目标是 `L=32,768`、`B∈{1,024,2,048}`；中间最大目标为 `4B`。这只证明计划可构建，不证明目标模型能看见自然 history。自然数据需要已验证的 128K 级窗口或协议允许的、与 H/Q 无关的完整短历史子集；本轮没有自行截断、删 session 或改用 oracle。

## 未执行项目

- C1 的 native tokenizer 已核验；API/native usage 仅有 probe：32K 首个请求在 provider `2048` 与授权 `10240` 下均 `finish_reason=length`、visible `content=null`，未形成 memory row。
- 正式每个压缩 step 的 actual visible memory、clip fraction 和 reader 长度：`NOT_RUN`；隐藏 reasoning 只作为阻塞证据，不作为 memory。
- `>5%` 输出被裁剪的技术门槛：`NOT_TESTABLE`。
- cap-matched 与共同实际长度 `t=min(actual lengths,B)` 重读：`NOT_RUN`。
- reader context 截断/服务商隐藏 compaction 证明：`NOT_RUN`。
- formal path 的 length-sensitive CI：`NOT_RUN`。

mock smoke 的 `work/smoke_contrast.json` 仅显示 `MOCK_ONLY`，不可用作长度或效应证据。

结论：长度门槛尚未关闭；不能宣称同一实际 memory budget 下的 direct/staged 比较成立。
