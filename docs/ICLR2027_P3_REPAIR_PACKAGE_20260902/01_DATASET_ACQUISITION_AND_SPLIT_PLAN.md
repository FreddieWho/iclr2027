# 数据获取、转换与冻结分割方案

## 0. 决策摘要

当前最缺的不是更多合成 pair，而是三类证据：

1. **更多独立比赛**：减少 10 场比赛和 32 个已暴露 heldout 标签样本导致的不稳定；
2. **任务语义正确的数据**：将绝对部署信息与内部构型信息分开；
3. **真正未暴露的确认集**：验证 geometry、task repair 和 dynamic support 结论是否可迁移。

建议采用四层数据体系：

| 层级 | 数据 | 主要角色 | 是否参与搜索 |
|---|---|---|---|
| A | SNGAR whole-match tracking，45 train + 9 valid | 扩大开发样本、构造任务、有限 autoresearch | 是，仅 train/valid |
| B | SNGAR 10 test matches | 最终同来源未见确认 | 否，candidate lock 后一次性运行 |
| C | IDSSE 7 matches | 不同联赛/供应商的外部确认 | 否，主候选冻结后运行 |
| D | SoccerTrack v2 | 不同采集方式和竞技水平的 acquisition-shift 检验 | 仅转换 smoke；正式结果锁后运行 |

现有 SkillCorner 10 场继续承担历史探索和动态 support 规则开发，但不再作为最终确认集。书法数据只在体育端预测冻结后进入。

---

# 1. 体育数据优先级

## A0. SNGAR-Action-Spotting-Tracking：首选扩展主数据

### 地址

- 数据页：`https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking`
- 同源窗口化版本：`https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR`

### 数据特征

- 64 场完整比赛的 player/ball tracking；
- 约 29.97 Hz；
- 11,849,815 tracking rows；
- 87,939 个事件，10 类 action labels；
- Parquet，约 3.0 GB；
- 官方分割：45 train / 9 valid / 10 test；
- 每帧包含 home/away players、角色/位置组、坐标、球、事件字段；
- 自带逐文件 SHA-256 manifest。

### 为什么最适合当前 P3

1. 比当前 10 场 SkillCorner 大一个数量级，且仍是结构化 tracking；
2. 有完整比赛，不只是事件窗口，可构造自然 frame pairs、动态局部协同和 context/intrinsic 两类任务；
3. 官方 match-level split 可以直接防止帧级泄漏；
4. 10 场 test 可以真正替代已经暴露的旧 heldout；
5. 事件和 tracking 同时存在，能构造自然任务，而不必继续依赖合成 intervention label。

### 重要限制

- 访问是 gated，需在 Hugging Face 页面接受条款并登录；
- test 在候选冻结前不得下载；
- 时间索引必须使用 `videoTimeMs`，不能用 `row_index / fps`；数据存在 halftime/outage gaps；
- 球在全体帧中的可见率并非 100%，需要报告缺失并设置任务级降级策略；
- 事件标签是 single-label priority resolution，不应把 rare class 缺失简单解释为模型失败。

### 开发数据下载

```bash
python -m pip install -U huggingface_hub hf_xet
hf auth login

# 先检查体积和文件，不下载 test
hf download OpenSportsLab/SNGAR-Action-Spotting-Tracking \
  --repo-type dataset \
  --local-dir data/raw/sports/sngar_tracking \
  --include README.md \
  --include MANIFEST.sha256 \
  --include annotations_train.json \
  --include annotations_valid.json \
  --include 'train/videos/*' \
  --include 'valid/videos/*' \
  --dry-run

# 确认后删除 --dry-run
```

### 最终测试下载

只允许在仓库中存在已提交且 hash 固定的 `candidate_lock.json` 后执行：

```bash
hf download OpenSportsLab/SNGAR-Action-Spotting-Tracking \
  --repo-type dataset \
  --local-dir data/raw/sports/sngar_tracking \
  --include annotations_test.json \
  --include 'test/videos/*'
```

下载完成后：

```bash
cd data/raw/sports/sngar_tracking
sha256sum -c MANIFEST.sha256
```

### 推荐 split 用法

- train 45：模型训练与无标签/弱标签结构任务；
- valid 9：候选搜索、早停、固定指标比较；
- test 10：候选锁定后一次性确认；
- 统计单位：match；event/frame 只作为 match 内观测。

---

## A1. SkillCorner Open Data：动态“语义小组”规则开发

### 地址

- 仓库：`https://github.com/SkillCorner/opendata`

### 现有资产

仓库已有 10 场 A-League 2024/25 tracking，每场包含：

- `{id}_tracking_extrapolated.jsonl`
- `{id}_dynamic_events.csv`
- `{id}_phases_of_play.csv`
- `{id}_match.json`

### 为什么仍然重要

虽然这些比赛已经被当前 P1–P3 使用，不能作为新的确认集，但 dynamic event 表包含许多比静态角色更接近真实协同的信息，例如：

- `pressing_chain`、`pressing_chain_index`、`index_in_pressing_chain`；
- `simultaneous_defensive_engagement_same_target`；
- `associated_off_ball_run_event_id`、`n_simultaneous_runs`；
- `passing_option`、`n_simultaneous_passing_options`；
- `push_defensive_line`、`break_defensive_line`；
- phase、line break、defensive structure、inside defensive shape 等。

这些字段可以用来构造 response-blind 的动态 support，而不再把 defender/midfielder/forward 当作最终语义真值。

### 下载/更新

```bash
git clone https://github.com/SkillCorner/opendata.git data/raw/sports/skillcorner_open
# 已存在时：记录当前 commit，不要无记录地 git pull
```

### 用途边界

- 用于定义和调试 dynamic support schema；
- 用于人工审计样本；
- 不再用于候选最终确认；
- 任何 support 规则必须在读取模型 response 前冻结。

---

## B0. IDSSE：首选外部确认数据

### 地址

- Figshare 下载页：`https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177`
- 论文说明：`https://www.nature.com/articles/s41597-025-04505-y`

### 数据特征

- 7 场德国 Bundesliga 一/二级联赛完整比赛；
- 官方 match info、event、position 三类文件；
- tracking 25 Hz，所有球员和球的 x/y 坐标；
- 11,137 events，1,002,644 position frames，207 players，10 teams；
- XML，约 2.45 GB；
- CC BY 4.0；
- 可由 floodlight/kloppy 类工具转换。

### 为什么适合外部确认

- 与 SkillCorner/SNGAR 不同联赛、不同数据提供体系；
- 有完整位置与同步事件；
- 许可开放；
- 足以检验 geometry-response 关系和固定候选方向是否依赖特定 provider。

### 下载方法

最稳妥方法是通过 Figshare API枚举文件并下载，避免把临时 `ndownloader` 地址硬编码：

```python
import json
from pathlib import Path
import requests

article_id = 28196177
out = Path('data/raw/sports/idsse')
out.mkdir(parents=True, exist_ok=True)
metadata = requests.get(
    f'https://api.figshare.com/v2/articles/{article_id}', timeout=60
).json()
(out / 'figshare_metadata.json').write_text(
    json.dumps(metadata, indent=2), encoding='utf-8'
)
for item in metadata['files']:
    target = out / item['name']
    if target.exists() and target.stat().st_size == item['size']:
        continue
    with requests.get(item['download_url'], stream=True, timeout=120) as r:
        r.raise_for_status()
        with target.open('wb') as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
```

也可以在 Figshare 页面点击 `Download all`。

### 使用规则

- 在主候选、任务定义、指标和超参数冻结前不读取结果；
- 转换 schema 可提前开发，但不能用 IDSSE 指标选择候选；
- 七场比赛整体作为 external-confirmatory set；
- 若必须调适 provider-specific parser，只允许使用文件结构/单位，不得根据模型得分调参。

---

## B1. SoccerTrack v2：采集域迁移验证

### 地址

- 项目页：`https://atomscott.github.io/SoccerTrack-v2/`
- GitHub：`https://github.com/AtomScott/SoccerTrack-v2`
- 数据页：`https://huggingface.co/datasets/atomscott/soccertrack-v2`

### 数据特征

- 10 场完整大学足球比赛，约 900 分钟；
- panoramic 4K 全场视野；
- per-frame GSR：2D pitch coordinates、persistent track IDs、jersey、roles、teams；
- BAS：12 类球事件；
- 数据许可 CC BY 4.0；
- Hugging Face 当前为 gated，需要申请/登录。

### 价值

它与 broadcast-derived SkillCorner/SNGAR 的采集条件不同，可以回答：

> support-conditioned geometry 和双通道机制是否只在某一种 tracking pipeline 中成立？

### 下载命令

```bash
git clone https://github.com/AtomScott/SoccerTrack-v2.git external/SoccerTrack-v2
cd external/SoccerTrack-v2
python -m pip install -U huggingface_hub
hf auth login

# 先用两场开发 parser，不读取正式模型结果
./scripts/download.sh \
  --dest ../../data/raw/sports/soccertrack_v2_smoke \
  --match 117092 --match 117093
```

主候选锁定后再下载余下比赛。parser smoke 比赛不得进入最终 acquisition-shift 统计。

---

## C0. SoccerNet-GAR：事件窗口快速原型

### 地址

`https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR`

它与 SNGAR whole-match tracking 来自同一批 64 场比赛，但提供 4.5 秒、16 个采样点的事件窗口。适合快速测试 action classification 和数据接口，不构成独立外部验证。

用途：

- 快速训练 task head；
- 验证 event-centered dual-channel 接口；
- 不与 SNGAR whole-match 结果当作两个独立数据集计数。

---

## C1. 事件数据降级方案

当 gated tracking 暂时无法获取时，可用以下公开事件数据完成任务/标签接口，但不能替代连续 tracking 几何验证。

### Wyscout open event logs

- `https://figshare.com/collections/Soccer_match_event_dataset/4415000/5`
- 覆盖五大联赛、世界杯和欧洲杯的大规模 event logs；
- 包含 position、time、outcome、player、event attributes；
- 用途：event label pretraining、类别频率审计、natural-event task 原型；
- 不可用于完整 Jacobian/support geometry 结论。

### StatsBomb Open Data / 360

- `https://github.com/statsbomb/open-data`
- event、lineups，以及部分比赛的 360 freeze-frame；
- 用途：稀疏事件时点的 spatial task 或 parser fallback；
- 不是连续 tracking。

### Metrica sample data

- `https://github.com/metrica-sports/sample-data`
- 少量同步 tracking + event；
- 只用于转换 smoke 和单元测试，不承担统计结论。

---

# 2. 书法数据

书法不是当前 P3 task-repair 的训练数据。它只能在体育端机制和预测冻结后承担跨域确认，不能反向选择模型、band、support 或 loss。

## D0. Make Me a Hanzi：矢量受控实验

- 地址：`https://github.com/skishore/makemeahanzi`
- 结构：`graphics.txt`、`dictionary.txt`，含笔画路径、顺序和字符结构；
- 用途：严格的 stroke/component intervention、无接缝重渲染、等能量控制；
- 已在当前仓库使用；继续保留为受控 instrument，不承担作者/水平真值。

## D1. HCSU：自然风格/作者/结构确认首选

### 地址

- Hugging Face：`https://huggingface.co/datasets/Tongji209/HCSU`
- GitHub：`https://github.com/209-Tongji/HCSU`
- ModelScope：`https://www.modelscope.cn/datasets/Tongji209/HCSU_Fine-Grained_Historical_Calligraphy_Style_Understanding_Dataset`

### 数据特征

- 总体说明为 39,307 张字符图像，49 位书法家、10 个朝代、5 种书体、Tie/Bei/Wild 三个 domain；
- 当前公开高质量 Tie/Bei 约 7,020 条，Hugging Face viewer 约 7.4k rows；
- 元数据包括 character、author、dynasty、source type、quality、ink style、stroke style、character structure 和专家描述；
- CC BY-NC 4.0。

### 当前项目中的合理用途

- 同字符、不同作者/结构的 retrieval；
- 不同字符、同作者的 style consistency；
- page/source modality 与 intrinsic structure 分离；
- 检验体育端冻结的“context/mode 双通道”是否迁移。

它不能替代 Make Me a Hanzi 的笔画级结构真值。

### 下载

```bash
python -m pip install -U huggingface_hub hf_xet
hf download Tongji209/HCSU \
  --repo-type dataset \
  --local-dir data/raw/calligraphy/hcsu \
  --dry-run
# 核对体积、许可证和文件后移除 --dry-run
```

## D2. MCCD：作者/书体多属性大规模备选

- GitHub：`https://github.com/SCUT-DLVCLab/MCCD`
- 近 330,000 张 isolated character images；
- 7,765 个字符类别、10 种书体、15 个朝代、142 位书法家；
- PNG / LMDB；
- 需要提交申请并获得解压密码；
- 仅限非商业研究，CC BY-NC-ND 4.0。

用途：当 HCSU 的作者×字符交叉覆盖不足时，用于大规模作者/书体分类和 retrieval。由于申请可能耗时，立即提交申请，但不让它阻塞体育主线。

---

# 3. 动态“语义小组”构造

## 3.1 为什么不能继续用静态角色当真值

`defender / midfielder / forward` 只是球员的长期职位，不等于某一时刻真实共同执行动作的单位。现有 P2 已经否定“静态角色组普遍同方向更特殊”，但没有否定真实动态协同单元。

## 3.2 高置信 dynamic support 类型

优先只使用可审计、response-blind 的 support：

1. **Pressing chain support**  
   同一 `pressing_chain_index` 内的防守球员；可加入 `simultaneous_defensive_engagement_same_target`。

2. **Off-ball coordinated run support**  
   同一 possession window 中 actor、ball carrier、`associated_off_ball_run_event_id` 及同时 run 的球员。

3. **Passing-option unit**  
   player in possession + passing option targets；只在字段完整且时间重叠时使用。

4. **Defensive-line movement support**  
   由已冻结的角色/位置和短窗速度一致性共同定义；不能只按 defender 标签。

5. **Movement-coherence fallback**  
   在事件前固定窗口内，基于速度方向一致、距离/图连通和同队约束生成 support。规则在读取任何模型 response 前冻结。

## 3.3 必须保存的记录

```text
DynamicSupportRecord
  dataset_id
  match_id
  event_id
  anchor_time_ms
  support_rule_version
  support_type
  player_ids
  node_indices
  actor_id
  team_id
  pre_window_ms
  post_window_ms
  event_label
  source_columns
  confidence_level
  support_size
  graph_density
  graph_cut
  velocity_alignment
  manual_audit_status
  source_checksum
```

## 3.4 人工小审计

最低充分审计：

- 分层抽取 200 个 dynamic supports；
- 两名评阅者独立判断“是否构成合理的共同动作单元”；
- 报告 agreement、无效原因和规则版本；
- 只用于验证 support construction，不根据模型 response 修改规则。

人工审计无法完成时，将这些 support 明确标为 `algorithmic_dynamic_support`，不得直接称为 tactical ground truth。

---

# 4. Context 与 intrinsic 任务

## 4.1 Context tasks：应该保留绝对位置

- high / medium / low block；
- team centroid / deployment zone；
- phase/deployment classification；
- absolute field-location prediction。

对这些任务，global translation 可能是信号。评价目标是 `z_ctx` 可访问性，而不是一味降低 global response。

## 4.2 Intrinsic tasks：应该忽略整体位置但保留内部组织

优先使用：

1. **Intrinsic formation retrieval**  
   基于 centered/Procrustes internal geometry 定义自然近邻；正负样本按 match 隔离。

2. **Natural pair ranking**  
   正例：内部构型近、绝对位置远；负例：绝对位置近、内部构型远。只使用自然帧，不用 intervention label。

3. **Dynamic support type / event-context classification**  
   使用 event-centered 局部结构预测 pass/tackle/run 等，但需做 context shortcut audit。

4. **Natural dynamic support localization**  
   用冻结 dynamic support labels，而不是合成 intervention support。

## 4.3 不能再使用的错误总门

禁止继续用：

```text
phase macro-F1 越高越好
AND
全局平移 response 越低越好
```

作为同一个 embedding 的统一优劣判据。应分别评价 `z_ctx` 和 `z_mode`，并报告二者的交叉泄漏。

---

# 5. 冻结 split 协议

## 5.1 旧数据状态

- SkillCorner 原 10 场：`EXPLORATORY_HISTORICAL`；
- 旧 heldout：`EXPOSED_DURING_CANDIDATE_SEARCH`；
- 旧 heldout 指标可以报告，但不得作为新的确认性 evidence。

## 5.2 新 split

```text
SNGAR train 45    → training / task construction
SNGAR valid 9     → candidate selection / early stopping
SNGAR test 10     → candidate-lock 后一次性 run
IDSSE 7           → external provider/league confirmation
SoccerTrack smoke 2 → parser only
SoccerTrack remaining → optional acquisition-shift confirmation
```

禁止跨 match 随机切 frame。禁止把同一比赛的不同 event 分到 train/test。

## 5.3 Candidate lock 前必须冻结

- 数据 commit/revision 与文件 hash；
- task definitions；
- parser and canonical schema；
- model architecture；
- candidate hyperparameters；
- seeds；
- primary/secondary metrics；
- noninferiority margin；
- statistical unit；
- failure criteria；
- figure source tables。

---

# 6. Canonical schema 与转换验收

所有体育源转换为：

```text
CanonicalMatch
  dataset_id
  source_match_id
  provider
  competition
  pitch_length
  pitch_width
  native_fps
  time_axis
  source_revision
  source_checksums

CanonicalFrame
  dataset_id
  source_match_id
  timestamp_ms
  period
  player_ids[N]
  team_slots[N]
  roles[N]
  positions[N,2]
  velocities[N,2] | null
  node_mask[N]
  ball_position[3] | null
  adjacency[N,N]
  context_labels
  event_labels
  support_metadata
```

验收：

- x/y 单位统一为 meters 和 normalized pitch 两份；
- 攻击方向标准化但保留原坐标；
- 时间以数据原生时间列为准；
- 身份、球缺失率按比赛报告；
- 不允许静默插值跨越长 gap；
- 每个源保留 raw→canonical mapping、checksum 和 parser version；
- 生成 `conversion_receipt.json` 与 `data_card.md`。

---

# 7. 数据获取失败时的降级顺序

1. SNGAR access 未批准：先用现有 SkillCorner + IDSSE 开发 task semantics 和 parser；
2. SNGAR 仍不可用：用 SoccerNet-GAR tracking windows 做 task prototype，但最终结论必须由 IDSSE/SoccerTrack continuous tracking 支撑；
3. SoccerTrack gate 未批准：IDSSE 作为唯一外部源，明确 external n=7 限制；
4. MCCD 未批准：使用 HCSU + Make Me a Hanzi；
5. 任何 gate 失败都不能用 Wyscout/StatsBomb event-only 数据冒充 continuous geometry evidence。

---

# 8. 许可证与发表注意

- SNGAR：遵守 gated access 条款，不重分发；记录接受日期和用户；
- SkillCorner：按仓库许可与引用说明署名；
- IDSSE：CC BY 4.0；
- SoccerTrack v2：dataset CC BY 4.0，code MIT；
- HCSU：CC BY-NC 4.0；
- MCCD：CC BY-NC-ND 4.0，需申请；
- Make Me a Hanzi：不同文件来源许可不同，保留 COPYING 与来源说明；
- 双盲稿件中不能把私人申请邮件、用户名或绝对本地路径写进正文/补充材料。
