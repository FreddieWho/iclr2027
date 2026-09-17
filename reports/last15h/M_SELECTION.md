# M 分支收敛（E2/E3/E4，2026-09-18）：双数据故事升级通过

## E2：真数据三件套 3/3 成立 → 主文升级为双数据故事
- N01 转折：三种子 turn_miss frozen→cover：0.277→0.129 / 0.383→0.184 / 0.336→0.176；积分与精度全同向。**进主文方法证据**（cover 修转折位置）。
- N03 路径事件：剂量反应 frozen 0.255→cleanonly 0.185→cover 0.092；FA 代价 +3pp。**进主文现象证据**。
- N06-hard 行动：lam2 invalid frozen→cover：0.309→0.102 / 0.410→0.184 / 0.348→0.141；lam0 三方全 ~0（置信机制复现）。**进主文后果证据**。
- 升级含义：主文从"坐标故事＋真数据注脚"变为"坐标＋T5R3 双数据故事"；R02-Track6（miss|flip 0.305）＋E2（转折/事件/行动）构成真数据侧完整链条。

## E3：关闭（诚实 null）
- 端点规则零中途违例，但全库验算 violators=0（含 ±0.8 大旋转，2318 干净有效）——线性路径在此任务族下不重入，实验空洞。边界结论：端点有效在此自动路径有效；N03→N06 链条需可重入动作族（future）。

## E4：死（3 种子全平）
- joint≡state_only≡shuffled（turn_miss 0.44–0.53，积分 0.31–0.34）；shuffled 对照证明 pair 信号零贡献。N10 线全关。

## 方法位终局
flipmine（坐标）＋cover（真数据，lr3e-4）——同一思想（oracle 重标记覆盖）两侧各自三种子成立。新方法位仍空缺（E7 在 LEADS 待挖掘）。

## 可运行命令
```
python3 experiments/last15h/e2_n01_t5r3.py --out artifacts/last15h/E2/n01 [--seed 23|47 --out .../n01_s23|s47]
python3 experiments/last15h/e2_n03_t5r3.py --out artifacts/last15h/E2/n03
python3 experiments/last15h/e2_n06_t5r3.py --out artifacts/last15h/E2/n06b --hard [--seed ...]
python3 experiments/last15h/e3_fullpath.py --out artifacts/last15h/E3/fullpath_bigrot --bigrot
python3 experiments/last15h/e4_joint.py --out artifacts/last15h/E4/joint
```

## 红线
J03WQQ 未碰；holdout_303 未碰；新输出 artifacts/last15h/E{2,3,4}＋reports/last15h/E*，旧证据只读。
