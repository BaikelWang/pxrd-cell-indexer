# PXRD Cell Indexer

从粉末 XRD **峰表**预测 **Niggli 原胞** `(a, b, c, α, β, γ)` 的神经 cell indexing，并经 **seeded McMaille** 扩库、可选 **rerank** 出 Top-1 / Top-20。

本仓库只做 **indexing（定胞）**，不做全结构生成。

```text
peaks + λ
  → NN Flow seed (K)
  → symmetrize + LSQ
  → seeded McMaille (.allcells)
  → rank: McM20 或 linear rerank
  → Top-1 / Top-20  (primitive L4-strict)
```

---

## 1. 当前产品口径（2026-08）

### 1.1 端到端流程

| 步 | 模块 | 说明 |
|----|------|------|
| ① | 峰输入 | CNRS：多阈值并集 `I≥5 ∪ I≥1`；MP100：模拟峰 |
| ② | NN seed | Peak Transformer + Flow，采样 K=100 / 1000 个 primitive seed |
| ③ | 对称化 + LSQ | 投到常见晶系并精修 |
| ④ | Seeded McMaille | `policy=1`，局部搜索扩候选 |
| ⑤ | 排序 | `none`：McM20 降序；`linear`：V0 等权重重排 |
| ⑥ | 评测 | **primitive L4-strict**（统一主指标） |

Reranker **只置换** `.allcells` 顺序，不改 seed / LSQ / Mc，也不扩库（`lib` 不变）。挂接点：`scripts/run_cnrs_e2e_compare.py:seeded_ordered_cells`（`--rerank {none,linear}`）。

### 1.2 主 checkpoint

| 版本 | 路径 | 训练要点 |
|------|------|----------|
| **V4（产品默认）** | `results/flow_seedgen/pxrd_indexer_full6m_v4_wide_lr2e3/best.pt` | full6m · peaktf wide · 34 ep · 固定 I>5 · select `valid_macro@0.2%` |
| V5（研究） | `results/flow_seedgen/pxrd_indexer_full6m_v5_imin/best.pt` | 同上 + 60 ep + 训练时 `peak_imin_choices=5,2,1` |

启动脚本：`scripts/launch_pxrd_indexer_full6m_v4_wide.sh`、`scripts/launch_pxrd_indexer_full6m_v5_imin.sh`。

### 1.3 看板数字（primitive L4-strict）

**CNRS**（123 条真实谱，产品优先）：

| 配置 | Top-1 | lib |
|------|------:|----:|
| V4 · K=100 · McM20 | 39.0% | 64.2% |
| **V4 · K=100 · + rerank** | **43.1%** | 64.2% |
| V4 · K=1000 · McM20 | 31.7% | 68.3% |
| **V4 · K=1000 · + rerank** | **41.5%** | 68.3% |
| V5 · K=100 · McM20 | 40.7% | 64.2% |
| V5 · K=100 · + rerank | 39.8% | 64.2% |

**MP100**（100 条模拟）：

| 配置 | Top-1 | lib |
|------|------:|----:|
| V4 · K=100 · McM20 | 49% | 59% |
| V4 · K=1000 · + rerank | 57% | 69% |
| **V5 · K=1000 · McM20** | **59%** | **74%** |
| V5 · K=1000 · + rerank | 60% | 75% |

**产品默认（盯 CNRS）**：**V4 + K=1000（或 K=100）+ equal-weight rerank**。  
V5 在 valid / MP100 更好，但 CNRS 库召回未跟涨，且 V4 上锁定的 rerank **不能直接套到 V5**。

多引擎对标详见实验记录（CNRS / MP100 + JADE9 等）。

---

## 2. 关键结论（读这几条即可）

1. **K↑ 抬 lib、压 McM20 Top-1**；要用 K=1000，必须配能迁移的排序（当前是 V0 rerank）。
2. **Rerank V0**（冻结）：`score = McM20_pct + nn_dist_pct − 0.25·vol_dev`。在 V4 四池上均不降；合成优选 / XGB **跨不过 CNRS**，故默认用手写等权重。
3. **Rerank 局限**：只换序；会 help/hurt；与 seed 分布强耦合；峰匹配类 FOM 往往伤分。
4. **V5 vs V4**：合成指标↑、MP100↑；CNRS 正交 / raw lib 未兑现 imin 设计目标——**合成上学得更好 ≠ 实验谱更好**（域间隙，非简单负相关）。
5. **下一步**不宜只堆同分布数据重训；优先实验域对齐的训练/选模，以及跟新 seed 重标定排序。

---

## 3. Reranker（V0）

```text
NN seed → sym+LSQ → seeded Mc → .allcells
                                    │
                              ┌─────┴─────┐
                              │  RERANKER │  ← 唯一改动点
                              └─────┬─────┘
                                    ↓
                              Top-1 / Top-20
```

| 项 | 内容 |
|----|------|
| 实现 | `src/pxrd_cell_indexing/rerank/linear_v0.py` |
| 默认权重 | `(1, 1, 0.25)`，见 `results/rerank_v0/current/`（本地） |
| 训练数据 | 扰动合成库 `data/rerank/`（gitignore，可重建） |
| 设计 / 冻结评测 | [`docs/开发日志/20260731-Reranker设计与实验方案.md`](docs/开发日志/20260731-Reranker设计与实验方案.md) · [`docs/实验记录/20260731-Reranker-V0训练与冻结评测.md`](docs/实验记录/20260731-Reranker-V0训练与冻结评测.md) |

```bash
# 冻结四池评测（不对评测集调参）
python scripts/eval_rerank_freeze.py --model results/rerank_v0/current/model.json
```

---

## 4. 指标与数据

| | |
|---|---|
| **输入** | 变长峰表 `(2θ, I)` + `λ`（默认 Cu Kα 1.54184 Å） |
| **输出** | Niggli 原胞六参数；e2e 经 Mc 后取 Top-k |
| **推理契约** | peaks-only（无 formula / 无真实 CS） |
| **主 KPI（产品）** | CNRS / MP100 **primitive L4-strict** Top-1 · Top-20 · lib |
| **训练选模** | `valid_macro@0.2%`（分晶系宏平均；valid 峰阈值固定 I>5） |
| **训练数据** | full6m LMDB（外部）；合成谱 + 增强 |

旧栈「valid1400 strict raw / A3-G1 + q-search」仍保留在代码与历史文档中，**不再是产品主线**。

---

## 5. 模型（Flow seed 栈）

生产 seed 模型：`Peak Transformer (wide)` + **conditional Flow**，在 full6m 上 DDP 训练。

| 项 | V4 / V5 共用设定 |
|----|------------------|
| Backbone | peaktf wide：`d=512` · `L=8` · FFN 2048 · out 1024 |
| Flow | 8×1024 |
| Batch / lr / amp | global 2048 · `2e-3` · bf16 |
| Select | `valid_macro` @ 0.2% · K=100 探针含 MP100 |

V5 仅多：`epochs=60`、`peak_imin_choices=5,2,1`（训练时随机峰强阈值；valid/select 仍 I>5）。

训练入口：

```bash
# 推荐：launch 包装（含 torchrun / 日志）
bash scripts/launch_pxrd_indexer_full6m_v4_wide.sh
bash scripts/launch_pxrd_indexer_full6m_v5_imin.sh

# 或底层脚本
bash scripts/train_pxrd_indexer_full6m.sh   # 见脚本内参数
```

核心代码：`scripts/train_flow_seedgen.py`、`src/pxrd_cell_indexing/data/dataset.py`。

---

## 6. 端到端评测（常用）

```bash
# CNRS 多阈值并集 e2e（需已编译的 McMaille 与 run_lab 布局）
python scripts/run_cnrs_e2e_multithresh.py ...   # 见脚本 --help
python scripts/run_cnrs_e2e_compare.py --rerank none    # McM20
python scripts/run_cnrs_e2e_compare.py --rerank linear  # V0

# MP100：dump seed → reseed Mc → 评分
python scripts/dump_flow_seed_pool.py ...
# reseed 须在 third_party/McMaille/run_lab/ 下按既有 run 脚本执行
python scripts/summarize_mp100_benchmark.py ...
```

经典引擎对标：`scripts/run_cnrs_classic_engines.py`、`scripts/run_mp100_classic_engines.py`、`scripts/summarize_*_benchmark.py`。

本地依赖（不进 git）：

| 资源 | 说明 |
|------|------|
| 训练 LMDB | 外部 `pxrd_241113_*.lmdb` |
| Checkpoint | `results/flow_seedgen/**/best.pt`（gitignore） |
| McMaille / TREOR / ITO / DICVOL | `third_party/`（本地，gitignore） |
| CNRS / RealPXRD 数据 | 按实验记录路径挂载 |

---

## 7. 仓库结构

```text
pxrd-cell-indexer/
├── README.md · AGENT.md
├── configs/                      # 含 100k A3-G1 历史配置、6M 相关 yaml
├── docs/
│   ├── 开发日志/                 # 方案、决策、周报
│   ├── 实验记录/                 # 单次实验摘要
│   └── references/               # 论文摘要（大文件 gitignore）
├── src/pxrd_cell_indexing/
│   ├── data/ · model/ · training/
│   ├── search/                   # 历史 q-search 轨
│   ├── rerank/                   # V0 linear reranker
│   └── geometry.py · eval.py …
├── scripts/                      # 训练 · e2e · rerank · 对标 · 诊断
├── tests/
├── data/MP-100samples-benchmark/
├── results/                      # gitignore：ckpt / 跑分
└── third_party/                  # 本地经典引擎（gitignore）
```

---

## 8. Quick start

```bash
pip install -e ".[dev]"
make test
```

历史 **A3-G1 raw 回归**（非当前产品主线）仍可用：

```bash
python scripts/train.py --config configs/scale_100k_a3_g1_gstar6.yaml
python scripts/eval_mp100.py --checkpoint results/experiments/.../best.pt
```

---

## 9. 文档索引（近期优先）

| 文档 | 内容 |
|------|------|
| [`docs/实验记录/20260802-V5加Reranker端到端对比V4.md`](docs/实验记录/20260802-V5加Reranker端到端对比V4.md) | **V5 vs V4 + rerank** 看板 |
| [`docs/实验记录/20260731-Reranker-V0训练与冻结评测.md`](docs/实验记录/20260731-Reranker-V0训练与冻结评测.md) | Rerank V0 冻结数字 |
| [`docs/开发日志/20260731-Reranker设计与实验方案.md`](docs/开发日志/20260731-Reranker设计与实验方案.md) | Rerank 设计与 Gate |
| [`docs/实验记录/20260731-K1000消融-CNRS与MP100.md`](docs/实验记录/20260731-K1000消融-CNRS与MP100.md) | K=100 vs 1000 |
| [`docs/实验记录/20260729-CNRS多引擎L4对标.md`](docs/实验记录/20260729-CNRS多引擎L4对标.md) | CNRS 多引擎 |
| [`docs/实验记录/20260731-MP100多引擎L4对标含JADE9.md`](docs/实验记录/20260731-MP100多引擎L4对标含JADE9.md) | MP100 含 JADE9 |
| [`docs/开发日志/20260720-CellIndexing-后续优化方案v4.md`](docs/开发日志/20260720-CellIndexing-后续优化方案v4.md) | 早期 A3/q-search 方案（历史） |
| [`docs/开发日志/起点.md`](docs/开发日志/起点.md) | 项目背景 |
| [`AGENT.md`](AGENT.md) | Agent 协作约定 |

---

## License

TBD.
