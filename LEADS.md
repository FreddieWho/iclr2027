# LEADS

2026/9/5
## L-001  smoke 场 117092/117093 可作 parser 调试但禁入正式统计
manifest 明确排除。其余 8 场的 GSR 结构若与 smoke 一致，parser 可先在 smoke 上验证。
最小验证：只解析 smoke 场结构、不跑指标，约 0.5h。
状态：已放弃（2026-09-18 清理：parser 随 T5R2 审计通过，无残留工作）

2026/9/5
## L-002  ball 轨迹与 BAS 事件暂未使用
当前任务只需要 20 人位置；ball/BAS 可能支撑未来的事件对齐分析，但会引入新语义。
最小验证：暂不做，等 T5R6 主线闭合后再评估。
状态：待挖掘

2026/9/5
## L-005  GRF（Google Research Football）模拟干预臂
外部检索建议：模拟环境的干预有 ground truth，可彻底消除"输入干预导致内部表征 OOD"的审稿批评。
若成立可替代/补强 IDSSE 受限访问的可复现性负债，也是 rebuttal 阶段的现成预案。
最小验证：GRF 导出轨迹→跑冻结箅子与预测器，约 1-2 天。
状态：待挖掘（本期投稿冲刺不启动，rebuttal 或下一篇用）

2026/9/18
## L-006  训练类主张的 ~10 seeds/arm 确认门（E6）
小 MLP 重训噪声地板 ±15pp（同配置两次运行差 17pp，见 reports/last15h/N04.md），单种子训练对比不可 headline。
若开：唯一能让 N04 课程 / N05 迁移类主张上正文的通道；先导须 ≥3 种子同向且均值效应 ≥12pp 才值得烧。
最小验证：30 次小训练（3 臂 × 10 种子，同 recipe，eval-only 分母），约数小时 CPU。
状态：待挖掘（等 E2 之后有候选主张再开；同一时间只开一条）

2026/9/18
## L-007  转折位置精度的全新机制（E7，v3）
Bracket 双侧标签（v1）＋方向 margin（v2）在积分尺子下双种子一致输 flipmine 4pp："敢转"易、"转准"难，位置精度需要别的机制。
若成立就是主发现里唯一未解机制问题的新方法位；先做零训练诊断（转折处梯度/曲率结构分析），再谈训练，避免重蹈 v1/v2。
最小验证：坐标单转折路径诊断小试，零训练，约 1h。
状态：待挖掘（v3 网 2026-09-18 打完：12 臂×5 种子全败/持平，斜率非 binding constraint；下一假设应在表示侧而非监督侧，见 reports/last15h/E7_v3_verdict.md）

2026/9/19
## L-008  DINOv3 S/B/L 同家族 size sweep
门禁未解（HF gated manual，无 token；GitHub 只给 Meta 接受页，绕不过；S/B/L ID 已确认）。
若许可就绪，同一批 v2 数据、同一协议 S→B→L，B1 先行、holdout 一次性，禁 VLM prompting/finetune。
最小验证：先通 S 的权重下载＋dev atomic。
状态：待挖掘（blocked：外部许可；DINOv2 S/B/L sweep 先行，见 TODO）
