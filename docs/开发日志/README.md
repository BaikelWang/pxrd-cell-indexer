# 开发日志

> **用途**：记录**工作性质**的内容——做了什么、优化了什么、设计方案、决策、周报、复盘等。  
> **习惯**：Agent 与 PM 协作过程中**随手记录**，每次有意义的进展都应落盘。

## 写什么 / 不写什么

| ✅ 放这里 | ❌ 不放这里 |
|---|---|
| 任务目标确认、方案讨论、设计决策 | 单次实验的完整配置与数值结果 |
| 优化思路、代码重构说明 | loss 曲线、指标表、checkpoint 路径 |
| 周报、里程碑总结、问题清单 | 训练命令的完整参数 dump |
| 与 `起点.md` 类似的历史复盘 | 可复现实验的原始日志 |

实验的配置、过程、结果、分析 → 见 [`../实验记录/`](../实验记录/)。

## 命名建议

```
开发日志/
├── README.md           ← 本说明
├── 起点.md             ← 项目背景与历史复盘
├── 20260706-任务初始化.md
├── 20260713-周报-W27.md
└── ...
```

- 单次记录：`YYYYMMDD-主题.md`
- 周报：`YYYYMMDD-周报-Wxx.md`

## 索引

| 文档 | 说明 |
|---|---|
| [`../references/README.md`](../references/README.md) | 外部论文/代码本地镜像：OpenAlphaDiffract、DeepLPnet、AIdex/AIdex-R2、Attention…Diffraction |
| [`起点.md`](起点.md) | Cell Indexing 全历程复盘（2026-05～07 历史背景） |
| [`20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md`](20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md) | Without-L MP100 复测、与 Mc/NN 口径澄清、可迁移启示 |
| [`20260714-RealPXRD-纯XRD-lattice消融综合报告.md`](20260714-RealPXRD-纯XRD-lattice消融综合报告.md) | A2 消融：掐断 formula 后 MP100 elementwise 归零 |
| [`20260715-CellIndexing-可执行优化方案v3.md`](20260715-CellIndexing-可执行优化方案v3.md) | **当前执行方案**：A0 评测 → E1c@100k → Peak Transformer → reciprocal metric → 训练课程/放量 → 全局搜索与排序 |
| [`20260714-CellIndexing-Top1逐步优化方案v2.md`](20260714-CellIndexing-Top1逐步优化方案v2.md) | 已由 v3 取代；保留历史决策与阶段演进 |
| [`20260706-任务初始化.md`](20260706-任务初始化.md) | 脚手架与文档约定 |
| [`20260710-周报-W28.md`](20260710-周报-W28.md) | 周报 W28（2026-07-06~07-10） |
| [`20260717-周报-W29.md`](20260717-周报-W29.md) | 周报 W29（2026-07-13~07-17）：Peak Transformer + cs_cond + gstar6 → ~42.5% |
| [`20260706-训练数据调研-alex_aflow_oqmd_mp.md`](20260706-训练数据调研-alex_aflow_oqmd_mp.md) | 训练集 LMDB 规模、字段、PXRD 口径与子集候选 |
| [`20260707-PM决策与待确认清单.md`](20260707-PM决策与待确认清单.md) | D1–D3、D5–D8、D10 确认；D4/D9 待定 |
| [`20260707-输入形态调研-实验PXRD与Indexing工具对比.md`](20260707-输入形态调研-实验PXRD与Indexing工具对比.md) | 实验上传+λ 场景；RealPXRD/Mc/JADE/GSAS 输入对比与 D4 建议 |
