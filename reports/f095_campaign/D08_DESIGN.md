# D08｜设计（最近邻公平对打，2026-09-23）

## 目标
回答"相对于最接近的训练思想，flipmine多解决什么"。对照组锚定已有存档，缺项按原配方补齐，不发明新loss名。

## 军备现状（已核查）
- 复用（不重训，记hash）：raw flipmine×3、raw fliprand×1(s11)、relflip×3、raw clean×3、relfeat×1(s11)。
- 新训（同配方：train_101＋mine-seed=5挖掘集＋全批量300ep＋Adam1e-2＋BCE）：raw fliprand×2(s23/s47)、relfeat fliprand×3、balanced×6（raw/rel×3种子）、pct-preserve×6。
- 免训分析：temp-calib（flipmine两encoder×3种子，T在train静态场景上拟合）。

## 五臂定义（同基础训练集＋同单编辑候选池＋各自可见标签）
1. **flipmine**：clean＋全部1534-flip BCE（lam=1.0）。复用。
2. **fliprand**（普通等量增广）：clean＋随机一半flip（同数对照，r04b原配方，seed+777划分）。s11复用，余新训；rel域三种子全系新训（该对照本就不存在）。
3. **balanced**：flip池内类别平衡（337 pos＋337 neg，seed固定下采样，lam=1.0）。检验"flip增益是否只是多数类灌水"。unique样本数如实少一截，不补量（合同要求列出，不要求等量）。
4. **pct-preserve**：从同种子clean checkpoint做continuation（estimand注明非从头训练）；loss＝cleanBCE＋flipBCE＋preserveBCE(lam_pres=1.0固定，零搜索、如实记录)；preserve集＝baseline在train静态上判对的样本，teacher只读train预测软目标；不碰AB/测试oracle。
5. **temp-calib**：flipmine模型的输出温度T（train静态NLL拟合），纯输出控制。预期：只动工作点，不动S/J*（单调变换不变性，作为sanity验证）。

## 超参预算
全臂固定配方零搜索（lam_pres=1.0是单点取值不是调优，如实记录）。梯度步数全相同（300ep全批量）；encoder全程可训（与flipmine一致）；unique样本/抽样概率/lam记入manifest。

## 评价（dev512，统一分母）
R_endpoint/R_full/M/J（B110沿用U1的raw_clean整网基线，保证与U1表可比）＋static/atomic覆盖＋train旧正确子集回归率。完整E结果不挑分母。

## 可证伪分支（冻结）
- Q1：rel域上任一对照臂的J ≥ relflip－0.05 → flipmine非特殊（novelty收窄到诊断贡献）。
- Q2：pct臂的旧正确回归率低于flipmine ≥5pp且J不差 → preserve约束有独立价值（U05开绿灯，否则U05不开）。
- Q3：temp-calib动J不动S/J* → 输出控制与表示/训练效应分离成立（sanity）。
- 判决：Q1成立→方法贡献让位诊断；Q1不成立∧任一对照接近→有限支持；全远离→flipmine特殊性成立。
