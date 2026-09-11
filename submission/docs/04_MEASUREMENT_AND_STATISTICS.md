# 04｜测量、混杂控制与统计

## 1. 必存字段

每个压缩步骤：history_id/group_id/source/split，path_id，B、step_index、target budget、真实 input/output canonical tokens、native/API usage、完整输入/输出 hash、前一步 memory hash、compressor ID/prompt hash/replicate、finish reason、clip token 数、原始输出、实际送入下一步文本、latency、estimated/actual cost。

每题：question_id/type/gold reference，reader ID/prompt，所读 memory hash，raw answer，score/scoring_version，证据审计状态，失败原因。future question/answer 不能反流到压缩请求。key 不落盘。

## 2. Budget matching 不可糊弄

统一 canonical tokenizer 并存版本；对同一 C 的不同路径还报告 native tokens。B 是包含所有持久文本、标签、日期、source ID 的总额，而不是只计算正文。固定的系统提示和当次问题单独计 read cost。

建议提示输出90–100%预算。真实运行不因输出偏短就按分数反复再生成。长度修补也是一次 re-encoding，不能免费藏进某臂。默认允许 deterministic cap 截断，但记录 clipped fraction；如果 >5% 的 memory 需要截断超过5%原输出，视为长度协议不合格，先校准。因服务商输出上限截断/半个 JSON 不能当正常完成。

确认报告必须同时给 cap-matched 与实际长度敏感性结果。预设方法：同 history 上所有被比较 memory 取共同长度 t=min(actual lengths,B)，按共同截断操作重读；这是额外诊断 estimand，不偷偷替代原结果。记录 t/B 并报告极短情形。若裁剪改变方向，只能声明长度/排版敏感，不支持严格 endpoint-path 主 claim。不能用 padding 声称信息预算相等。

## 3. 有效读出与失败状态

没有隐式会话：每一步 fresh request，仅给当前 memory；模型缓存只能是字节相同前缀复用，不可携带隐藏 history。开新 chat 并不足以保证平台无 global memory，需查服务设置。

API 超时、rate limit、格式失败、拒绝、context overflow、服务端 truncation 单独报告；不要默认为错误答案，也不要从分母静默删除。报告 intent-to-run 与 valid-pair 数量。若一条路径技术失效率>5%或路径间差异>3pp，先修或标不可比。

确认主比较按完整 matched histories，缺失必须报流图。存在大量路径相关缺失时，不用 complete-case 的漂亮结果通过 GO。

## 4. Retention / accessibility 的操作定义

证据审计必须识别语义内容，而不是只匹配原句。exact ID 可以 exact；关系、更新与撤销必须检查实体、谓词、时间、scope、polarity、provenance 是否共同正确。

| 证据状态 | 读出 | 解释（只在已审计范围） |
|---|---|---|
| 完整、语义正确 | 对 | 可用记忆 |
| 完整、语义正确 | 错 | reader 使用失败候选；仍需控制问题本身难度 |
| 缺失/扭曲 | 错 | 存储失败候选，不自动证明不可恢复 |
| 缺失/扭曲 | 对 | 猜测、先验或未标证据；不能当完整保留 |
| 不可判定 | 任意 | unknown，不能强制塞进前面类别 |

用 retain-only highlighting（只突出 memory 中已有内容，不加答案）和 gold restoration（明确引入外部已丢内容）区分恢复方式；加等长度非gold placebo控制位置/长度收益。所有被增添的文本计入诊断读出长度。restore 后答对不自动证明“memory 原来绝对没有答案”，必须结合证据审计。

已有 restore-counterfactual 工作 [EXT-RESTORE]；这里是适配 measurement，不宣称首创。不能把不同路径子集上的“错误中不可逆比例”直接比较而不报告总错误率：主报告使用全体问题中的错误数/损失率，附条件比例及分母。

## 5. 统计定义

每个历史先对同类问题、压缩重复和 reader 配置内求平均；路径对比 d_i=U_i(path A)−U_i(path B)，再均匀平均历史，不能让问题多的历史权重大。不同 reader/compressor 的结果先分别报告，不以平均掩盖反转。

探索：点估计、95% history-cluster bootstrap CI、分布与失败例，均标 exploratory。

确认：至多两个从探索锁定的对比或 interaction，family 内采用 Bonferroni-adjusted CI（m=2 时97.5%双侧），不对探索几百个格子事后挑一个显著值。分析脚本 `--family-size` 明确写入。统计单位 n=history；seed不能增加 n。bootstrap推荐5000次，正式10000可作为敏感性，不要求重算到显著。

interaction 应直接 bootstrap d_update−d_atomic 或 d_B1−d_B2，而不是“这边显著那边不显著”。类型效应允许整体效应为零，但要确认该类型和对照类型来自同 histories/可比来源。

## 6. 等价与噪声

p>0.05 不是无效应。若要说实践等价，预设 ±0.03 utility 带，并要求相应全CI落在带内；所有相关预设对比都满足才可推广到已测范围。无法排除±10pp时只能说证据不足。no-memory 接近oracle或两者均地板，说明仪器没分辨率。

同配置独立重运行只刻画生成波动，不能证明 API 温度0真正可复现。保存 fingerprint。bootstrap CI会利用真实 replicate，不凭空制造确定性。多个有限预算点只能支持响应曲线/交互，不支持物理相变。

## 7. Rewrite 对照的解释

R3、S3、W3同时改变中间文本和累计 token 工作量。S3−R3不是一个无假设的“纯压缩路径效应残差”，也不能由相减得严格因果中介比例。正确说法是相同调用次数的干预对比支持/不支持某种解释。总call数、总input/output tokens、price、latency都列出，价格缓存折扣单列。

## 8. 实验费用与可部署策略费用

跨路径共享前缀/缓存会降低这次实验的实际账单，不能把这种跨候选共享当成某个策略部署时的成本优势。分别报告实际实验消费和按单一路径逻辑调用/usage重建的独立策略成本；cost-saving判据使用后者，并固定每条历史服务的未来问题数。API缓存折扣和provider差异单列。
