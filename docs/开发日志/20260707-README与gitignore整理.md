# 2026-07-07 — README 润色与 gitignore 整理

## 做了什么

1. **README 重写**为独立仓库风格（项目名 **PXRD Cell Indexer**）：
   - 任务一句话、I/O 表
   - 架构图 + 数据流图（mermaid）
   - 目录结构、Quick start、外部依赖、当前里程碑
2. **`.gitignore` 扩充**：
   - 忽略：`*.pt`/`*.ckpt`、tensorboard、`data/processed/*.jsonl`、LMDB、results 产物、工具缓存
   - 保留：`MP-100samples-benchmark/`、`processed/*.json` 小统计文件
3. **`data/README.md`** 更新：明确哪些入库、哪些本地生成、外部路径

## 目的

为新建 git 仓库（`pxrd-cell-indexer`）做准备：README 可直接作为仓库首页；大文件与实验产物不入库。
