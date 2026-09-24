# U04｜设计（冻结编码器交叉重训，2026-09-23）

## 目标
把"表示降第一段、flip降第三段"（U03行为描述）做因果定位：排序增益是否在encoder里、flip增益是否在readout即可复制。

## 合同
- **Encoder**：四臂已有权重 raw_clean / raw_flipmine / relfeat / relflip × 3种子（路径复用`u1_factorial_eval.arm_checkpoint`），取`net+feat_head`输出的32维z并冻结。主干sha256在重训前后各记一次（冻结不断言、只验证）。
- **Readout**：全新`Linear(32,1)`，与原`cls`同结构。关键事实（先记录）：`feat_head`与`cls`之间无非线性，二者可合并为单层线性——因此**冻结z上的线性readout与原head函数容量完全一致**，不存在"head更弱"的混杂。
- **监督**：静态train-only；singleflip覆盖（clean＋全部mined-flip BCE，lam=1.0）。训练数据与原配方逐项相同：train_101静态场景＋oracle标签、mine-seed=5挖掘集（五份mined.npz字节一致，已验md5）、全批量、300ep、Adam 1e-2、wd=0。
- **无dev选择**：原配方固定预算无选择；readout沿用固定预算（线性＋BCE是凸问题，初值不敏感）。初值种子固定并记录（`torch.manual_seed(1000+seed_idx)`）。
- **不直接跨接原head**：不同隐空间坐标不对齐，只比较"各encoder上重训的同样head"。
- **小MLP对照**：`Linear(32,16)→ReLU→Linear(16,1)`，仅flip监督，用于辨别线性可达性；同预算。
- **评价**：dev512银行（pool=eval_202，与训练不交叠），逐state独立预测；S/J*/J（`threshold_certificate`）、atomic_pass、AB_correct；R_full/M以**同encoder静态readout的B110**为分母（镜像U1的repair_vs_rawclean结构）；原整网数从`U1_SUMMARY.json`并表（不重算、不覆盖）。

## 可证伪分支（冻结，跑前写定）
- P1：relflip-encoder＋静态readout的S ≈ 原relflip整网S（±0.03内）→ 排序增益在encoder里。
- P2：任一encoder＋flip-readout的J − 同encoder静态-readout的J ≥ +0.10 → flip增益readout充分。
- P3：小MLP-readout在任一encoder上相对线性-readout的J增益 < +0.05 → 线性可达（表示已把问题变线性）。
- 判决：P1∧P2∧P3 → 层间互补SUPPORTED；任一失败 → 按失败项收窄（保留U2行为描述）。
