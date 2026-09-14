# 06｜Agent 执行清单

## 开始前
- [ ] 读取当前 progress summary 与 V3 partial artifacts
- [ ] 确认旧 V3 不再续跑
- [ ] 复制/生成 V4 config，不覆盖 V3
- [ ] 复用 tokenizer revision/hash 与 cost ledger 机制
- [ ] 运行现有 tests；V4 最小新增测试通过

## P0
- [ ] 2 histories
- [ ] C1 thinking off
- [ ] direct/staged2/rewrite natural-stop 检查
- [ ] actual visible length 记录
- [ ] query-blind 检查
- [ ] R1 oracle/raw/no-memory
- [ ] 最多一次允许调整

## P1
- [ ] 12 histories
- [ ] 冻结 memory 后统一 reader
- [ ] history-level utility
- [ ] ΔSD / ΔRD / ΔSR
- [ ] 4 info types
- [ ] length-balanced + regression
- [ ] 不因结果改变配置

## P2
- [ ] 24 new histories
- [ ] frozen config/hash
- [ ] R1 score
- [ ] R2 复读同一 frozen memory
- [ ] bootstrap 90% CI
- [ ] route decision

## 结束
- [ ] `V4_FAST_PILOT_REPORT.md`
- [ ] `V4_ROUTE_DECISION.json`
- [ ] 结果可追到 row/hash
- [ ] GO/NO-GO 无 MAYBE
- [ ] 未把 route-selection 结果冒充 publication-grade proof
