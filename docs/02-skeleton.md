# Step 3 — 项目骨架

> **状态**：🟡 D1–D20 已确认（见 `01-design.md`）；下方目录树是**当前占位骨架**，真实模块划分见 `01-design.md §3`，**待 PM 批准后再落地新增文件**（跨 3+ 文件改动，需先说明拆分计划）
> **最后更新**：2026-07-07

---

## 目录树

```
PRXD-Cell-indexing-model-0706/
├── AGENT.md
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── 00-requirements.md
│   ├── 01-design.md
│   ├── 02-skeleton.md          ← 本文件
│   ├── 03-guardrails.md
│   ├── 04-progress.md
│   └── 05-retrospective.md
├── results/
│   └── README.md
├── scripts/
│   └── README.md
├── src/
│   └── pxrd_cell_indexing/
│       ├── __init__.py
│       ├── types.py            ← Pydantic 数据契约
│       └── pipeline.py         ← 空接口 + TODO
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

## 模块职责

| 模块 | 路径 | 职责 |
|---|---|---|
| **types** | `src/pxrd_cell_indexing/types.py` | 输入/输出 Schema、实验条件、评测结果的数据契约 |
| **pipeline** | `src/pxrd_cell_indexing/pipeline.py` | 预处理 → 推理 → 后处理 的编排接口（当前仅 stub） |
| **tests** | `tests/` | 冒烟测试 + 后续单元/集成测试 |
| **scripts** | `scripts/` | CLI 入口（训练、评测、批量推理） |
| **data** | `data/` | 运行时数据路径约定 |
| **results** | `results/` | 实验日志、checkpoint、评测输出 |

## 接口预览

核心类型与函数签名见 `src/pxrd_cell_indexing/types.py` 与 `pipeline.py`。

⛔ **Step 3 约束**：当前不写业务逻辑，仅保留接口与 docstring。
