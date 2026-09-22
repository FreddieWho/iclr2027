# 单路 worker 指令模板
总控将 ROUTE_ID、输出目录及已知数据入口填入后交给实现worker。

你负责 ROUTE_ID。先读本包 MASTER_AGENT_PROMPT.md、对应 routes/R*.md 和03_SHARED_ENGINE.md，然后做最小可运行试验，不再输出一份替代实验的头脑风暴。
只写自己的 `experiments/discovery_campaign/ROUTE_ID/`、`artifacts/discovery_campaign/ROUTE_ID/` 和结果卡；共享内核变动交W0协调，避免互相覆盖。
真实数据adapter未就绪时先用有解析oracle的关系场景；不是全部工作阻塞。已失败的实现可以修复；真实阴性只在有新的概念理由时改构造，不转去旧配方扫参。
结果标明哪些实测、哪些仅数学/工程演示、哪些未完成。返回一条可能改变摘要的新事实或明确反证，加产物路径和下一项最值钱的试验。用examples中的极简记录即可，不建设新的审批平台。
资源和费用遵循总控已有授权；不要调用未知模型/provider或擅自付费。没有worker工具时这些指令可以由总控顺序执行。
