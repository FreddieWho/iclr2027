# FINAL_EVIDENCE_TABLE（正文数字唯一来源，2026-09-18）

规则：正文只允许从本表取数字。条件（discovery vs fresh vs 真数据）不得混用。
`status`：confirm=E1 fresh holdout 或多种子确认；discovery=开发阶段发现数（只许配 fresh 数并列引用）；
pending=重跑中（U02-calib、U03b-fixed）；park=阴性诊断保留。

| claim | number | numerator | denominator | dataset | split | seeds | status | artifact |
|---|---|---|---|---|---|---|---|---|
| compositional miss (fresh headline) | 0.858 | 151 | 176 | synthetic coord | fresh holdout_909 | fixed model | confirm | artifacts/last15h/E1/holdout_909/result.json |
| compositional miss (discovery, 并列引用) | 0.939 | 62 | 66 | synthetic coord | dev eval_202 | fixed | discovery | N04/round1c |
| flip repair emergent (fresh) | 0.773 | — | 176 | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| flip repair C1 miss (fresh) | 0.524→0.347 | 130→86 | 248 | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| C0 false alarm (fresh, 持平) | 0.262→0.238 | — | — | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| static error (fresh) | 0.180→0.107 | — | — | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| action invalid lam2 (fresh) | 0.750→0.625 | — | — | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| action invalid lam2 (discovery, 并列) | 0.770→0.683 | — | — | synthetic coord | dev eval_202 | fixed | discovery | N06/round1+round1b |
| turn miss, 随机编辑分母 (限定措辞) | 0.844 | 108 | 128 | synthetic coord | dev, 3072随机编辑筛单转折 | fixed | discovery | N01/round1; E1_VERDICT 分母调查 |
| turn miss, 瞄准路径分母 (fresh) | 0.438 | 28 | 64 | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| turn repair (fresh) | 0.438→0.313 | 28→20 | 64 | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| event cond miss (discovery) | 0.181 | 23 | 127 | synthetic coord | dev | fixed | discovery | N03/round1d |
| event cond miss (fresh, n小弱确认) | 0.042 | 1 | 24 | synthetic coord | fresh holdout_909 | fixed | confirm | artifacts/last15h/E1/holdout_909/result.json |
| 真数据 turn miss frozen→cover | s11 0.277→0.129; s23 0.383→0.184; s47 0.336→0.176 | — | — | IDSSE football | 跨场, 3 seeds | s11/s23/s47 | confirm | reports/last15h/HEADLINE_FREEZE.md (E2-N01) |
| 真数据 event cond miss frozen→cover | s11 0.255→0.092; s23 0.362→0.180; s47 0.347→0.168 | — | — | IDSSE football | 跨场, 3 seeds | s11/s23/s47 | confirm | HEADLINE_FREEZE (E2-N03) |
| 真数据 action lam2 invalid frozen→cover | s11 0.309→0.102; s23 0.410→0.184; s47 0.348→0.141 | — | — | IDSSE football | 跨场, 3 seeds | s11/s23/s47 | confirm | HEADLINE_FREEZE (E2-N06-hard) |
| pixel emerg clean→pixflip (同风格) | 0.537→0.351 | — | — | rendered coord | seen style | 1 train seed | discovery | artifacts/next6_ef0f7a3/u03/result.json |
| pixel 5风格 pixflip emerg (nuisance锁定确认, 3训练seeds全同向) | s804增益−18/−27/−12/−4/−15pp; s805增益−10/−10/−16/−8/−11pp; s806增益−30/−30/−16/−21/−28pp | — | 547 quartets | rendered | 5 unseen styles, 同quartet同seed新鲜RNG | 3 train seeds | confirm | artifacts/next6_ef0f7a3/u03b_fixed + u03b_s805 + u03b_s806/result.json |
| U02 flipmine覆盖增益 (不受calib bug影响) | 拒动 0.43→0.118; score λ≥1 全胜 | — | 93 test scenes | synthetic coord | dev/test split | fixed | confirm | artifacts/next6_ef0f7a3/u02/result.json |
| U02 calibration T (修正已落地) | clean 8.14 / flipmine 8.43 (旧T→20为符号bug产物, 作废; T>1=常规过自信压平) | — | — | synthetic coord | dev/test split | fixed | confirm | artifacts/next6_ef0f7a3/u02_fixed/result.json |
| X03 低秩补丁交换≈整换≈随机子空间 | 0.544/0.491/0.484; anti 0.147 | — | 450 pairs (anti 150) | synthetic coord | fresh bindings eval_202 | fixed | park | artifacts/next6_ef0f7a3/x03/result.json |
| X03b 拼接读出anti vs 补丁anti | 0.87–0.89 vs 0.23–0.27 | — | 150 anti | synthetic coord | anti-only train | fixed | park | artifacts/next6_ef0f7a3/x03b/result.json |

## 措辞红线（审计结论，写正文时强制遵守）
1. 主 headline 用 fresh 151/176=85.8%，62/66=93.9% 只许并列“开发发现→fresh确认”。
2. 84.4% 必须限定："Among single-turn paths discovered by random edits, 84% induced no
   model transition; on directly targeted fresh paths, the miss rate was 44%." 不得平均，不得泛化。
3. X03 只许写 "learned rank-4 low-rank patch map"（禁 projection）；结论只许
   "hybrid truth readily decodes from [z_b,z_d], but could not be installed into the frozen
   classifier by the tested rank-4 patch at this layer"（禁"fundamentally cannot compose"）。
4. U02 "confidence anti-information" 段删除/以 fixed-T 重写；score/lexico/u02b/argmax机制结论保留。
5. U03b 在 fixed 重跑落地后口径升级为“同quartet三状态新鲜同seed RNG，只改变几何状态”；
   落地前只许“同批渲染图三臂比较”。（2026-09-18 已落地，见 u03b_fixed。）
6. T5R3 真数据：以3种子+跨场方向一致为准，无 pristine holdout，局限如实写。
