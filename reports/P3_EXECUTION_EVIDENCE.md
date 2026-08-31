# P3 执行证据索引

日期：2026-09-01

## 状态

P3-T0–T5 已完成工程施工；科学路由为 `T4_T5_COMPLETE_RESPONSE_SHAPING_ONLY_P4_GATE_NOT_MET`。P4 AMR 未启动。该文件只索引 P3 证据，不改写 P2。

## 冻结 provenance

- P2 freeze HEAD：`f44be1e9526796a76b58ca2bd44dc562ab7841a3`
- 文档迁移 commit：`10d0711` (`docs(p3): integrate support-conditioned geometry refinement`)
- 当前 P3 config SHA-256：`e2842cea410a6fc2170d3fb750f81f2a03b64b9d510975c2f438a2eed8f6243b`
- `scripts/p3_support_geometry.py`：`66e5e9cd3353870eb4499ad4cb86951964cee8b7a3f92dbf88da862d3aa9431f`
- `scripts/p3_causal_switch.py`：`d50dcc947562333c0c860d33a6055d417da78d02d88ce5ced4a5ba1bf9220514`
- `scripts/p3_node_localization.py`：`456d16814c52137abdcb51af39e1d28e926df744e913699f6ac2bd0fb92ca574`
- samples、模型 manifest、embedding manifest、P2 matching/response/statistics manifest 和 9 个 checkpoint hash：见 `configs/phase3_support_geometry_v1.yaml`
- 环境：Python 3.11.5、PyTorch 2.11.0、NumPy 1.26.4、pandas 2.3.3、SciPy 1.13.1、PyArrow 22.0.0、PyYAML 6.0.1、scikit-learn 1.5.1；CPU、float32。
- 各任务运行时 config snapshot hash 可能早于最终同步配置：T0=`831d972b...`、T1=`1372659b...`、T2=`994e5459...`、T3=`0e708427...`、T4/T5 repair=`9808b0d...`；完整值见 `artifacts/phase3/P3_FINAL_EXECUTION_MANIFEST.json`。这一区别被保留，未把后写入的结果锁伪装成运行前锁。

## 结果 receipts

| task | artifact | summary/receipt SHA-256 |
|---|---|---|
| T0 | `artifacts/phase3/support_geometry_v1/t0_summary.json` | `8c9359653a8072a889bdd781060b5f7366b125911ed1dcb7d2611c1db16c8073` |
| T0 | `artifacts/phase3/support_geometry_v1/t0_receipt.json` | `2ce53ffa7f850134e90f6f041e597b4ffaaf61c7ac04d7cd6b7f1a4c39168c81` |
| T1 | `artifacts/phase3/support_geometry_v1/retrospective_summary.json` | `32a5f78ef26055a3bf4ec48daf4acafa128f52209fc196803a90856891fcf4a5` |
| T1 | `artifacts/phase3/support_geometry_v1/retrospective_receipt.json` | `cf433739c10f2e68af5d731369264bd19a9fc9234abdfb27674ada116794c5bc` |
| T2 | `artifacts/phase3/support_geometry_prospective_v1/intervention_manifest.json` | `ee16159c64929f874ef50d0d2f3be87769a79d569d80a635ff2eb5e36c64fb6a` |
| T2 | `artifacts/phase3/support_geometry_prospective_v1/prospective_receipt.json` | `5c24abf24cab69a6700df497a2cdc7310ce1420b3bc0c50c842f662c4eeef62a` |
| T3 | `artifacts/phase3/support_geometry_v1/layerwise_summary.json` | `cf55830edea5f1a80719a9f82833b70b6c7716c35db5be4d3b44ab47be06c51a` |
| T3 | `artifacts/phase3/support_geometry_v1/layerwise_receipt.json` | `c3578ed7b645d1c543ddb8fd76941061bae91efec211bd2b4c89723ddf809569` |
| T4–T5 | `artifacts/phase3/selected_causal_switch_v1/switch_summary.json` | `da04a16a0a51b3eb06cf65e62958612eedb8b0a4ef9a5c9a2a95b08720b73092` |
| T4–T5 | `artifacts/phase3/selected_causal_switch_v1/switch_receipt.json` | `e37557e253b5152dcbdbdb5aa778d19655abc79edb52870249f3f2c7f390cfaa` |

T5 nodewise localization receipt：`artifacts/phase3/selected_causal_switch_v1/node_sensitivity_localization_v2_receipt.json`，SHA-256 为 `5d0130743b1f249fbcc3bb6009097f4b6b11a5ba03939f3482508b5bad0e9816`。T5 checksum：`artifacts/phase3/selected_causal_switch_v1/t5_localization_SHA256SUMS`，SHA-256 为 `dac6d7ec87ec5014965150397febe722f1c5fddcffe9692a29ba62f4d5bb853d`。

每个 artifact root 另外保留 `SHA256SUMS`、`known_limitations.md`、execution contract 和 manifest。T0/T1/T3 共用 `support_geometry_v1` 根目录，但各任务的 summary、receipt 和 checksum 独立保留；没有覆盖 P2 文件。

## 统计和结果摘要

- T1：272,169 arm rows、121,293 pair rows、10 matches；full geometry mean Spearman `0.8466`，baseline `0.1222`。
- T2：5,700 sets 中 4,475 完整；9 模型共 90,432 arm rows、40,275 pair rows；full geometry mean Spearman `0.8526`，baseline `0.1294`；response-blind generation 为 true。
- T3：按 canonical order 每个 match 前 5 个样本，共 50 个 layer-wise samples；diagonal 为主，off-diagonal 随架构在 message passing/pooling 形成或保留。
- T4–T5：Phase-GAT `team_mean` vs `relational_pairwise`，3 seeds、matched parameter count `149,639`；开关改变 geometry/response，但 heldout macro-F1 均值 `0.1861→0.1839`，context response `0.01324→0.01571`，所以是 response shaping only。
- T5 localization：六个 switch checkpoint、53,700 个 complete arms、60 个 match-group rows；node-sensitivity top-k recall 为 relational `0.1546`、team mean `0.1487`，uniform support-size baseline `0.1458`。q-full parity 最大绝对误差 `8.9e-9`，但定位增益接近 uniform，不能作为任务修复证据。

完整指标、失败条件和最终十问回答见 `reports/P3_SUPPORT_CONDITIONED_GEOMETRY.md`；P3 claim 状态见 `CLAIM_LEDGER.md`。
