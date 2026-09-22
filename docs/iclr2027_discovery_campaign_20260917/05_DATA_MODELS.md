# 05｜数据和模型：避免新数据工程吞掉探索

## 路线A：现有坐标数据与权重，立即复用
IDSSE、SoccerTrack的已授权本地canonical数据是第一选择。现有数据入口见04；只读使用。若大文件不在git，先在运行环境原项目目录查找，不要求用户重复上传已存在的数据。
原任务的几何/质心读出仍可作工程入口，但其结果只称访问代理；不能把一个坐标函数任务换名为战术理解。

## 路线B：本地生成的关系场景，不依赖下载
包内 `scripts/generate_relation_scenes.py` 生成四端点、两段线的相交/不相交关系，标签来自解析oracle。`core/relations.py` 返回几何margin，支持避开任意微小越界造成的假戏剧性。
```bash
python "$PACK/scripts/generate_relation_scenes.py" --n 256 --seed 101 --output artifacts/discovery_campaign/scenes/train_101
python "$PACK/scripts/generate_relation_scenes.py" --n 128 --seed 202 --output artifacts/discovery_campaign/scenes/eval_202
```
这是新受控场景，不是天然分布或人类标注benchmark。原始场景、其所有±编辑和所有渲染归同一split。颜色/形状/相机不与标签绑定。

agent为视觉分支实现一个固定渲染器：显示可识别A/B/C/D角色及两段连接，或用稳定的形状/颜色编码角色。文字本身不能泄露标签。固定画布、抗锯齿、线宽与分辨率后，先验证屏幕上可观察的关系与oracle一致。
不要从颜色直方图、对象数目、绝对位置得到标签捷径；正负类的这些分布在生成时平衡。若任务太简单或太难，可改成包含、相对次序、关系组合，但应更改科学任务而非扫美术参数。

## 路线C：一类冻结视觉模型，作为重要性/接口检验
优先选择已有可运行编码器；没有时使用DINOv2 ViT-S/14。官方README已核验模型入口和权重地址[S13]；**本次未下载权重或实测运行**。不把它称为2026最先进模型，不以过时的模型代表全部基础模型。

源码/模型卡入口：
```text
https://github.com/facebookresearch/dinov2
https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

当前官方接口：
```python
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
```
对正式实验固定下载的源码commit和权重文件，阅读许可与模型卡；不要在运行中无提示自动切模型。PyTorch Hub加载repo代码，首次使用先检查来源，其他依赖在隔离环境安装。使用官方预处理；CLS和patch token不同输入权限分别标记。

小规模冻结前向可在已有设备上分批运行；性能不够时先缓存特征，不上来训练大型VLM。不需要付费API，不需要以LLM judge替代几何真值。若已具备另一类视觉编码器，可增加一类复现；没有时不阻塞坐标实验。

## 路线D：可再仿真的动态数据（R06）
已核验的官方来源[S9,S14]：
```text
https://github.com/ethanfetaya/NRI
https://github.com/ethanfetaya/NRI/tree/master/data
https://arxiv.org/abs/1802.04687
```
README给出 `data/generate_dataset.py` 和 `--simulation charged` 选项，原环境为很旧的PyTorch/Python。推荐复用独立模拟器或有记录地移植现代NumPy实现，别先重装古老训练栈。下载源码后核验许可证、实际生成器接口和积分单位；本次未逐个验证data目录文件的现代兼容性。

可本地生成5–8粒子初态、速度、图和轨迹。每个初态生成反事实时必须真的再积分。小规模先排除数值不稳定，不扩建外部数据湖。

## 不自动开展的域
点云ModelNet/ScanObjectNN、完整书法识别和篮球管线不作为默认并行新数据任务。其思想/文献仍可作为对照；若已有可直接运行的数据/权重，总控可用它替代视觉或动态分支，**不另建第四套管线**。

## 下载记账
来源URL、实际commit/文件hash、许可和下载日期写入新run的一份source记录就够。已有授权数据无需再次审批；新的受控/收费来源不绕过授权。不可取得的来源不成为全部分支的阻塞。
