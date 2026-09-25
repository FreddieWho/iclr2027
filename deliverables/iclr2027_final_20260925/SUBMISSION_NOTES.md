# 当前投稿文件

- `main.pdf` / `main.md`：正文、摘要、关键词、声明与参考文献。正文 7 页，PDF 合计 8 页。
- `supplementary.pdf` / `supplementary.md`：独立补充材料，4 页，页码从 1 开始。
- 项目根目录 `tmp.md`：优化后的题目、摘要、关键词与 TL;DR；其题目、摘要、关键词已同步正文。
- `metadata.json`：本目录构建所用元数据。
- `archive/tmp_before_metadata_optimization.md`：修改前的根目录 tmp.md 备份。
- `submission.pdf` / `submission.md` 是旧合并版，当前投稿使用上方 main 与 supplementary 文件。

当前题目：**Repair or Relocation? Compositional Generalization Beyond Endpoint Accuracy**

AI 声明保留两句。两个 PDF 均无 overfull 与未定义引用。实验结果未复算，图表数据未改，未提交或推送 GitHub。

重建：`python3 build.py`。构建会同时生成分开的 PDF 与 Markdown，不依赖 pandoc-citeproc。
