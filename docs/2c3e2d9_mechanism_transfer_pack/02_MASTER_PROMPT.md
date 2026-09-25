# 总控 Prompt｜机制可检验、模块可迁移

仓库：`FreddieWho/iclr2027`
基准：`2c3e2d9640607c32aacf86c8758cd1e2f8ef317b`
资源：用户已有2080 Ti实例，允许使用；不自行新增租机、扩大付费资源或关停实例。

## 目标

不要再开二十路。主攻M1“保留对应关系的不变表示”与M2“冻结关系模块的跨观测迁移”；M3计算/空间资源分配是独立备选。

机制的证据至少区分：数学可表达性/信息可辨识性、有限数据下学得怎样、具体干预是否改善、换观测或对象后是否仍成立。不把代码修复、架构内建等式或相关性直接当新机制。

## 先读

`00_REVIEW.md`、`01_CRITICAL_REPAIRS.md`、本包03–06；仓库最新`reports/e832_focus/`、`reports/e1a933_review/`及相应源代码。不要恢复过时的“所有排序跨任务失败”“所有关系迁移关闭”标签，也不要把它们一律改阳性。

## 执行

1. CPU worker修R1–R5共享原语；用既有bank量化错误影响。不要重训不受影响臂。
2. CPU/短GPU做M1信息碰撞与角色保真对照。先复现一个已有有效锚点，再做新改动。
3. GPU worker可并行修mask接口、准备M2感知数据与强视觉基线；不必等待M1每项都结束。冻结模块迁移使用一个已验证、有明确输入语义的source模型。
4. M3在前两线需要解释N02或已有容量允许时并行小规模运行，不能抢占主要GPU队列。
5. 每阶段向用户汇总“新事实、对旧解释的影响、下一步”，不报一串SUPPORTED标签。

## 现场决策权

你可调整优化器、训练时长、模型宽度、抽样方式与实现细节；调整必须有训练曲线、信息合同或可辨识性依据，写简短`DECISION_NOTES.md`。禁止仅因为测试J不漂亮而换任务/分母/seed。

允许约20%的训练预算用于现场新lead，条件是能提出具体竞争解释和一个有辨别力的实验。不要求每个新点预注册，也不以预算/期限为理由放弃重要可修复错误。

## 证据规则

- shared dataset与model implementations优先；不再复制不等价的`Typed`类。
- 新训练只用clean与single edit，目标测试AB不进训练/选型；任何使用其他组合训练的支线另立claim。
- 选择用singleton-dev，最终确认按parent隔离。开发时可以反复分析已暴露银行，明确身份即可。
- 三seed是通常起点，主结果读effect/CI/方向；不因一个任意accuracy gate判整个原则失败。
- 解析器是任务上界/强基线，不能作为学习贡献；提供坐标特权监督时必须披露其足以决定标签。
- 每个“迁移”标签说明：设计原则复现、权重迁移、跨观测迁移、跨任务迁移中的哪一种。换任务重新训练不是同一模块迁移。

## GPU与工程

`nvidia-smi`实测显存、驱动、设备与可用空间；保留已有可用环境，不盲升级库。先FP32短探针；需要混合精度时使用设备支持的FP16+scaler并做数值对照，不默认BF16/FP8。以实测显存选batch，较小batch与累积梯度须记录，BN状态明确。

单GPU默认一条训练队列，CPU并行生成/审计。独立data/model/shuffle RNG；seed、batch列表、checkpoint含optimizer状态。新目录fail-closed，不覆写旧权重。SSH等凭据不进入文件或报告。

新增目录建议：
```
experiments/mechanism_transfer_v3/{common,m1,m2,m3}/
artifacts/mechanism_transfer_v3/
reports/mechanism_transfer_v3/
```

## 交付

`FIX_IMPACT.md`、`M1_MECHANISM.md`、`M2_TRANSFER.md`、`M3_COMPUTE.md`（未执行也说明）、`CLAIM_LEDGER.csv`、`REPRODUCE.md`、`PAPER_MIGRATION.md`、`FINAL_DECISION.md`。

最后只选择最强的1–2个贡献。允许改论文，必须依据本轮结果；不把某条阴性强行塞进中心。没有新方法成功时，保留精确诊断与边界，比把修bug改名为方法更可靠。
