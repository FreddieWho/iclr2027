# D02-aug｜对称增广训练（raw w64×N，同预算，判决 NARROW-POSITIVE）

- 问题：D02 群平均（测试时集成）增益大；训练时随机合法重标号能否学到不变性/抬高 J？
- 运行合同：`--aug g8`（逐 epoch 随机 G8 元，坐标层置换后重算标准化，同前向数）vs `--aug repeat`（同 batch 拼两份，同信息双倍计算对照）；raw w64f32 × clean/flipmine × 3 种子；dev512 同 quartet＋orbit 验收。输出 `D02/aug_{g8,rep}_s{seed}/`＋`D02_AUG_EVAL.json`。
- 结果表（identity J / groupavg / disagreement；基线 raw clean .055/.051/.055，flip .068/.059/.076）：

| arm | g8 ident | g8 avg | g8 disagr | rep ident | rep avg |
|---|---|---|---|---|---|
| s11 clean/flip | .076/.363 | .110/.418 | .50/.71 | .055/.089 | .038/.106 |
| s23 clean/flip | .194/.308 | .173/.414 | .62/.69 | .051/.097 | .055/.122 |
| s47 clean/flip | .080/.245 | .118/.350 | .46/.76 | .055/.038 | .055/.093 |

- 主 claim：增广训练大幅抬高 J（flip .24–.36 ≈ N-relflip .38–.44 的六成差距弥合；s47 塌缩被治愈，6/6 收敛），但**没有学会不变性**（disagreement .46–.76，all8≈0）——增益来自标注多样性/覆盖，不是学到的对称保证。repeat 无增益→非计算效应。
- 排序更新：rel-flip ≥ six-flip > g8-raw-flip > raw-flip。命名/任务形式解释 raw 缺陷的**重要部分**（承认），几何组织仍居首（残差 all8=0 且绝对值仍差）。
- 不能说的 claim：不能说"对称性解决"（不变性检验失败）；g8-clean s23 .194 单格偏高，复述区间不 cherry-pick。
- 下一步：g8×sixdist（增广＋几何是否叠加）列为可选；主文不动（附录 D02 表加两行，批量更新时）。
- 判决：SUPPORTED（增广有效）/ NARROW（机制归属：覆盖多样性，非不变性）。
