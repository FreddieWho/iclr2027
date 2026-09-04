# 数据与环境指南

> 一句话摘要：足球主数据可直接从公开仓库获取；书法采用“矢量笔画控制数据 + 自然书法图像”双层结构；篮球和 MathWriting 是低风险扩展与 fallback。

## 1. 数据优先级

| 优先级 | 数据 | 角色 | 是否阻塞主线 |
|---|---|---|---|
| P0 | SkillCorner Open Data | 足球主要仪器域 | 是 |
| P0 | Make Me a Hanzi | 书法矢量干预与 stroke/component graph | 是 |
| P0 | 开放书法图像集 | 自然图像 smoke test | 否，MCCD 可替换 |
| P1 | Metrica sample-data | 足球外部格式/比赛验证 | 否 |
| P1 | MCCD | 大规模自然书法主数据 | 否，需审批/密码 |
| P1 | CCSE | 书法 stroke pseudo-support | 否 |
| P2 | TrackID3x3 | 篮球复现 | 否 |
| Fallback | MathWriting | 客观隐式二维结构替代书法 | 否 |

## 2. 足球数据

### 2.1 SkillCorner Open Data

仓库：

```text
https://github.com/SkillCorner/opendata
```

当前公开内容包括：

- 10 场 2024/25 澳大利亚 A-League 比赛；
- 10 fps 广播跟踪；
- 球员和球的坐标；
- lineup、球场尺寸；
- dynamic events；
- phases of play。

下载：

```bash
git clone --depth 1 https://github.com/SkillCorner/opendata.git \
  data/raw/sports/skillcorner
```

预期结构：

```text
data/raw/sports/skillcorner/data/
  matches.json
  matches/<match_id>/
    <id>_match.json
    <id>_tracking_extrapolated.jsonl
    <id>_dynamic_events.csv
    <id>_phases_of_play.csv
```

许可：仓库 MIT；使用数据时按 README 要求致谢 SkillCorner。提交前重新读取仓库 LICENSE 和 README，保存快照到 `data/licenses/`。

### 2.2 Metrica Sports sample-data

仓库：

```text
https://github.com/metrica-sports/sample-data
```

特点：

- 3 场 sample games；
- tracking 与 event 同步；
- 不同格式可测试 parser 鲁棒性；
- 可通过 `kloppy` 读取。

下载：

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git \
  data/raw/sports/metrica
```

使用：只作为第二来源复现，不允许因格式适配阻塞 SkillCorner 主线。

## 3. 篮球数据

### 3.1 TrackID3x3

仓库：

```text
https://github.com/open-starlab/TrackID3x3
```

数据文件夹：

```text
https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX
```

数据包括 Indoor、Outdoor、Drone 三个子集；所有帧有 6 名场上球员 bbox，部分帧有每名球员 10 个关键点。数据集许可为 CC BY 4.0；仓库主代码 Apache 2.0，但子模块存在其他许可，尤其 jersey-number pipeline 为非商业许可。

推荐路线：

1. 先 clone 仓库和读取现成 ground truth；
2. 不运行完整检测、ReID 或 pose pipeline；
3. 只用 bbox 中心或现成追踪结果形成 2D 节点；
4. 优先 Indoor fixed-camera 子集；
5. 仅在足球主结果存活后下载视频大文件。

下载文件夹可尝试：

```bash
gdown --folder \
  https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX \
  -O data/raw/sports/trackid3x3_drive
```

Google Drive 文件结构可能变化，脚本失败时按网页人工下载，不要调试超过半天。

## 4. 中国书法数据

### 4.1 Make Me a Hanzi：受控结构数据

仓库：

```text
https://github.com/skishore/makemeahanzi
```

提供 9,000+ 简繁字符：

- 每一笔的 SVG path；
- stroke medians；
- 笔顺；
- IDS decomposition；
- stroke 到 component 的 matches。

下载：

```bash
git clone --depth 1 https://github.com/skishore/makemeahanzi.git \
  data/raw/calligraphy/makemeahanzi
```

关键文件：

```text
dictionary.txt
  character
  decomposition
  matches

graphics.txt
  character
  strokes
  medians
```

用途：

- 构建 stroke graph；
- 构建 component graph；
- 对页面、偏旁和单笔施加无接缝矢量干预；
- 重新渲染，避免 Photoshop 边缘伪影；
- 预训练/验证书法端 Action-Mode Probe。

许可不是一个统一 MIT 文件：dictionary 和 graphics 来源许可不同，必须保存 `COPYING`、`APL`、`LGPL` 文件并按具体派生数据遵守。

### 4.2 MCCD：自然书法主数据

仓库：

```text
https://github.com/SCUT-DLVCLab/MCCD
```

规模：

- 约 329,715 张字符图像；
- 7,765 个字符类；
- 10 种书体；
- 15 个历史时期；
- 142 位书法家。

下载入口在仓库 README 的 Baidu/OneDrive。数据已发布，但需要申请并获得解压密码；仅限非商业研究，许可 CC BY-NC-ND 4.0。

施工要求：

1. 立即提交申请，但不要等待批准才启动项目；
2. 先 clone repo：

```bash
git clone --depth 1 https://github.com/SCUT-DLVCLab/MCCD.git \
  data/raw/calligraphy/MCCD_repo
```

3. 将申请记录写入 `data/manual/MCCD_APPLICATION.md`；
4. 未获批时使用开放替代集完成 pipeline；
5. 不能重新分发 MCCD 原图到公开仓库。

### 4.3 开放替代一：zhuojg/chinese-calligraphy-dataset

仓库：

```text
https://github.com/zhuojg/chinese-calligraphy-dataset
```

规模：138,499 张、19 位书法家、7,328 个字符。

字符组织版：

```text
https://drive.google.com/file/d/1k849yUZhkUfbupZT0kRR2ZzZj5g89yLw/view
```

下载：

```bash
gdown 1k849yUZhkUfbupZT0kRR2ZzZj5g89yLw \
  -O data/raw/calligraphy/zhuojg_characters.zip
```

按书法家组织版 ID：

```text
10QJrw0Qdk4O1bIrehCLmdiCkwpLbGVe8
```

仓库代码 Apache 2.0；数据来源为互联网收集，正式使用前需要在数据卡中记录来源与潜在版权边界。优先只用于内部研究和非商业论文实验。

### 4.4 开放替代二：kirosc/chinese-calligraphy-dataset

仓库直接包含约 14,537 张图：

```bash
git clone --depth 1 https://github.com/kirosc/chinese-calligraphy-dataset.git \
  data/raw/calligraphy/kirosc
```

许可 GPL-3.0。规模较小，适合 smoke test，不适合作为唯一大规模证据。

### 4.5 CCSE：stroke instance segmentation

仓库：

```text
https://github.com/lizhaoliu-Lec/CCSE
```

论文提供 CCSE-Kai 和 CCSE-HW 两个公开 stroke instance segmentation 数据集，COCO 风格标注。

已知 Google Drive 文件 ID：

```text
CCSE-HW:  1U8mLLb_qWSqC4yRnJlzoVaI2ELF2lGAH
CCSE-Kai: 1-2VFuiWHSd3fzl9qYMoEi0mlO_BoSgCd
```

下载：

```bash
gdown 1U8mLLb_qWSqC4yRnJlzoVaI2ELF2lGAH \
  -O data/raw/calligraphy/ccse_hw.zip

gdown 1-2VFuiWHSd3fzl9qYMoEi0mlO_BoSgCd \
  -O data/raw/calligraphy/ccse_kai.zip
```

用途：只提供 pseudo-support 或 stroke segmentation baseline。AMR 主方法不得完全依赖一个重型分割模型，否则“隐式结构”会名不副实。

## 5. MathWriting fallback

完整数据：

```text
https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
```

轻量 excerpt：

```text
https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz
```

代码：

```text
https://github.com/google-research/google-research/tree/master/mathwriting
```

规模：约 230k 训练人写样本、15k valid、7k test、396k synthetic。完整包约 2.9 GB，许可 CC BY-NC-SA 4.0。

用途：

- 公式整体平移 vs 符号相对上下移动；
- 上标、下标、分数位置等客观二维关系；
- 若书法 stroke support、授权或自然验证严重阻塞，可快速切换。

只在触发 fallback 后下载 full；默认下载 excerpt 验证 parser。

## 6. 数据处理规范

### 6.1 足球

1. 解析 tracking；
2. 按球队分离节点；
3. 优先使用 10 名外场球员，GK 单独作为 context 或另做实验；
4. 对短缺节点帧进行过滤或 mask，不做随意插值；
5. 用 0.5–1 秒均值窗减少 tracking 抖动；
6. 保存 raw coordinates 和 centered coordinates；
7. 攻击方向标准化只作为可选 readout，不覆盖原始上下文；
8. split 按 match，不按 frame。

### 6.2 篮球

1. bbox 中心作为节点；
2. 队伍和球员身份来自标注；
3. 优先 fixed camera；
4. split 按视频；
5. 不运行完整视频 backbone。

### 6.3 书法

1. 矢量层生成 interventions；
2. 统一画布、线宽和 anti-aliasing；
3. 全局、component、stroke 干预使用同一重渲染路径；
4. 生成无语义变化但有同等重采样的 artifact control；
5. 自然图像 split 按书法家和字符双重考虑；
6. 不用审美标签定义主要真值；
7. 保存原始许可和来源元数据。

## 7. 处理后目录规范

```text
data/
  raw/
    sports/
    calligraphy/
    mathwriting/
  interim/
    sports_frames/
    calligraphy_vectors/
    render_cache/
  processed/
    canonical_samples/
    interventions/
    splits/
  licenses/
  manual/
  checksums/
```

所有数据处理脚本必须：

- 输入路径参数化；
- 不改 raw；
- 输出 manifest；
- 记录版本/commit；
- 记录随机种子；
- 支持 dry run。

## 8. 环境

### 8.1 推荐版本

- Python 3.10 或 3.11；
- PyTorch 2.4+；
- CPU/MPS 首先；
- Linux/macOS 均可。

核心依赖：

```text
numpy scipy pandas pyarrow
networkx scikit-learn
matplotlib pillow opencv-python-headless
shapely svgpathtools cairosvg
pyyaml tqdm rich
pytorch-lightning or accelerate
torch torchvision timm
transformers open_clip_torch
gdown requests kloppy
```

`bootstrap_env.sh` 创建 `.venv`，不安装固定 CUDA wheel。

### 8.2 硬件策略

#### CPU/MPS 可完成

- 数据下载和解析；
- 图谱和干预生成；
- DeepSets/GNN 小模型；
- 绝大多数早期探索检查点；
- DINO/CLIP 小批量冻结特征抽取。

#### 可选租 GPU

- 多个视觉 backbone 全量特征抽取；
- AMR 图像模型训练；
- 大规模 MCCD 实验。

GPU 申请前必须形成：

```text
GPU_REQUEST.md
  blocker
  current CPU/MPS benchmark
  exact model
  expected VRAM
  expected GPU hours
  minimum machine
  resource boundary and fallback
```

默认不允许因为“可能更快”而租卡。

## 9. 下载与校验脚本

```bash
bash scripts/download_public_data.sh --core
bash scripts/download_public_data.sh --basketball
bash scripts/download_public_data.sh --ccse
bash scripts/download_public_data.sh --math-full
python scripts/verify_data.py --manifest configs/data_manifest.yaml
```

下载脚本不会自动申请 MCCD，也不会绕过访问控制。

## 10. 许可与匿名投稿

- 每个数据源保存 LICENSE/README 快照；
- 不把受限数据打进代码仓库或补充材料；
- 论文中只提供下载指引和处理脚本；
- MCCD 仅非商业研究且禁止衍生再分发时，公开仓库只放索引和处理代码；
- Google Drive 链接可能变化，提交前验证；
- ICLR 双盲期仓库、W&B 和文件元数据不得暴露作者身份。

## 11. P3-T5R 数据与访问状态（2026-09-04）

当前数据入口由 `configs/data_manifest.yaml`（version 2）和 `configs/dataset_acquisition_manifest_v2.yaml` 共同描述：

| 数据 | 当前状态 | 允许用途 |
|---|---|---|
| IDSSE（7 场） | 本地原始文件已提供，来源登记与转换待完成 | 暂代 SNGAR 的开发、任务构造和候选选择 |
| SNGAR train/valid | gated access blocked/deferred | 当前不阻塞 IDSSE 替代分支，后续可升级独立样本量 |
| IDSSE 保留 match | same-source holdout | candidate lock 后一次性读取，不称独立 external confirmation |
| SkillCorner | existing historical asset | dynamic-support 规则开发和人工审计，不作最终确认 |
| SNGAR test / 其他 provider | not available in current branch | candidate lock 后才可承担独立确认 |
| SoccerTrack v2 | parser smoke only | 先做两场转换 smoke，正式 acquisition-shift 需候选锁 |

旧 heldout 只允许 `exploratory_audit_only`。跨比赛 split 是强制的，不能跨 match 随机切 frame；原始数据不进入 Git，必须保留 source revision、raw-to-canonical mapping、license snapshot、receipt 和 SHA-256。IDSSE 已用于开发后，不得把同一批数据再次写成独立 external confirmation。未获得访问权限时使用当前固定替代路线，但不能把 event-only 数据冒充 continuous geometry evidence。
