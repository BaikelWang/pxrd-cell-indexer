# 2026-07-13 — RealPXRD Without-L MP100 评测与 Indexing 对标综合报告

> **性质**：实测 + 口径澄清 + 架构启示（综合报告）  
> **背景**：在 indexing 模型优化同期，对 RealPXRD-Solver Without-L 在 Benchmark-MP100 上做 lattice-only 复测，并回答「为何看似 67% 而我们只有 ~15%」「indexing 能否从 RealPXRD 学到什么」。  
> **结论先行**：Without-L 的 67% **不能**作为 peaks-only indexing 基线；正统对标仍是 **McMaille / JADE**；RealPXRD 的价值在数据/表示/任务拆分，不在 indexing 算法本身。

---

## 1. 评测设置（本次实测）

### 1.1 目标

对 RealPXRD-Solver **Without-L**（`pxrd-all`）在 MP-100 上报告：

- 输入：MP100 **ideal 模拟峰** + **primitive formula**（Without-L 推理契约要求化学式）
- 输出：仅比 **lattice match**（不做 StructureMatcher 全结构）
- 口径：`ltol=0.05` / `atol=3°`
- 指标：Top-1、Top-20（oracle，任一命中即算）

### 1.2 实现与产物

| 项 | 路径 |
|---|---|
| 评测脚本 | `archive/RealPXRD-Solver/scripts/eval_mp100_without_l_lattice.py` |
| 权重 | `archive/RealPXRD-Solver/pretrained/weight/2501/pxrd-all/last_one.ckpt` |
| MP100 CIF | `Task/PRXD-Cell-indexing-model-0706/data/MP-100samples-benchmark/` |
| 结果 JSON | `archive/RealPXRD-Solver/实验/mp100_without_l_lattice/mp100_without_l_ltol0.05_atol3.json` |
| 运行日志 | 同目录 `run.log`（全量约 200 s / 100 样本，K=20） |

### 1.3 关键口径细节

| 项 | 约定 |
|---|---|
| 峰 | 与 indexing/`mp100.py` 一致：conventional → reduced → XRDCalculator（Cu Kα），`y>5` |
| 真值晶胞 | `SpacegroupAnalyzer.find_primitive()` → `(a,b,c,α,β,γ)` |
| 化学式 | **primitive composition**（如 `Al2F6`，禁止约化式 `AlF3`）——对齐 `起点.md` P0 |
| Top-K | `num_evals=20` 次**独立** flow 采样；无 FOM/Rp 排序；oracle 命中 |
| Match | 同时报 `find_mapping` 与 **elementwise** |

> 说明：Without-L 的推理接口**必须**有 formula。这是结构生成契约，**不是** indexing 允许的输入。见 §4。

---

## 2. 实测结果

### 2.1 RealPXRD Without-L @ MP100（严口径）

| 指标 | Match rate |
|---|---:|
| **Top-1** `find_mapping` | **34%** |
| **Top-20** `find_mapping`（oracle） | **67%** |
| Top-1 elementwise | 32% |
| Top-20 elementwise | 62% |

- mapping 相对 elementwise 约 **+5pp**（轴置换/等价描述）
- Top-1→Top-20 跳约 **+33pp**：主要来自 **20 次独立采样的 oracle 召回**，不是单次点估计变强

### 2.2 与 indexing NN / 引擎对照（勿混口径）

| 系统 | 数据集 | 输入 | 指标 | 数字 |
|---|---|---|---|---:|
| RealPXRD Without-L | MP100 | 峰 + **formula** | Top-20 mapping @ 0.05/3° | **67%** |
| RealPXRD Without-L | MP100 | 峰 + formula | Top-1 elementwise | **32%** |
| Indexing NN（matrix6 旧 ckpt） | MP100 | **仅峰** | Top-20 mapping | ~26% |
| Indexing NN（同上） | MP100 | 仅峰 | Top-20 elementwise | ~8% |
| Indexing NN 冠军 `cubic_split_clf` | **valid1400** | 仅峰 | raw Top-1 elementwise | **15.43%** |
| Indexing NN 冠军 | valid1400 | 仅峰 | Top-K elementwise | **16.50%** |
| **McMaille**（历史） | MP100 | **仅峰 + λ** | Top-1 @ 0.05/3° | **~65.9%** |
| **JADE9**（历史） | MP100 | 仅峰 | Top-1 @ 0.05/3° | **~68.1%** |
| RealPXRD Without-L（历史 indexing 误用） | MP100 | 峰（+常错 formula） | Top-1 lattice | **~5%** |

**~15% vs ~67% 不是同题比较**：一边是 peaks-only、单点/窄邻域；一边是 composition-conditioned、20 次扩散 oracle。

---

## 3. 为何 Without-L 能到 67%？（机制拆解）

按影响大致排序：

### 3.1 Oracle Top-K（+33pp 量级）

同一峰 + formula，换随机噪声跑 20 次；评测事后挑「有没有一次对」。  
产品可用的是可排序 Top-1；67% 是**召回上界**，不是引擎式 Top-1。

### 3.2 Formula / atom_types 强先验（indexing 不允许）

`chemparse` → 完整 `atom_types` 列表 → GNN 图规模与化学计量固定。  
体积/Z/组成空间被大幅压缩。这是结构生成优势，**违反 cell indexing 输入契约**。

### 3.3 同分布 ideal 峰

训练 LMDB 与 MP100 评测共用 pymatgen stick 峰管线，分布对齐友好。

### 3.4 表示与后处理（次要）

- 训练/采样：原始 **3×3 矩阵** flow matching；**无** Niggli、无 6 参数规范化、无显式 volume 约束  
- 评测：`lattices_to_params_shape` → `find_mapping`  
- **结论：不是「对 lattice 做了特殊标准化才高」**；你们反而有 `MatrixLatticeNormalizer`

### 3.5 抽查真实性

Top-1 elementwise hit 样本边长相对误差多在 1–3%、角偏差 ≪ 3°，非假阳性。  
大结构（高 `n_atoms`）失败更多——组成先验帮小胞更明显。

---

## 4. 正统 Indexing 口径：只能喂 PXRD（McMaille）

### 4.1 任务定义（产品 / `起点.md`）

```
输入：主峰表 [[2θ, I], …] + 波长 λ
禁止：Formula / atom_num 参与搜索
输出：Top-N 晶胞（+ 晶系提示 / Grade）
```

### 4.2 McMaille 算法本质（Le Bail 2004）

```
峰表 + λ
  → 六大晶系各自独立 Monte Carlo 搜参数
  → Bragg 算理论峰 ↔ 观测伪谱 → Rp 型 R 因子
  → 局部 MC 精修（可接受少量上坡跳出假极小）
  → 全局合并，按 R ↓、对称性 ↑、体积 ↓、指标峰数 ↑ 排序
  → Top-N
```

要点：

1. **Indexing 是 Bragg 几何搜索问题**，化学式不是必要条件  
2. 晶系是**枚举的搜索维度**，不是先分类再回归  
3. 打分是 **峰位拟合（R / M20\*）**，不是噪声重建 MSE  
4. 故能在 **无 formula** 下达到 MP100 ~66%（严）/ ~76%（宽）

### 4.3 任务拓扑（勿混）

```
峰 + λ ──► Mc / JADE / NN-indexer ──► 晶胞
                                      │
峰 + formula + 晶胞 ──► RealPXRD With-L ──► 结构 CIF
峰 + formula ──► RealPXRD Without-L ──► 结构 CIF（非 indexing）
```

| | McMaille / JADE | RealPXRD Without-L | RealPXRD With-L | 本仓库 NN |
|---|---|---|---|---|
| 任务 | **Cell indexing** | 结构生成 | 已知晶胞的结构生成 | **Cell indexing** |
| 输入 | 峰 + λ | 峰 + **formula** | 峰 + formula + **cell** | **仅峰** |
| 机制 | 多晶系搜索 + R/M20 | Flow 联合 lattice+coords | 固定 lattice，采坐标 | 回归 + Bravais 邻域 + FOM |
| 北极星角色 | **对标对象** | 退出 indexing 对照表 | indexing **下游** | 要对齐 Mc/JADE |

**裁决**：Without-L @ 67% **退出** indexing 基线表。历史 ~5% Top-1 才是「误当 indexing 用」时的量级。产品路径仍是 **Mc/JADE（或 NN）indexing → With-L**。

---

## 5. 「只做 lattice 为何还不如他？」——问题本身不成立

在正确口径下：

- 该问的是：相对 **Mc ~66% / JADE ~68%**，我们为何只有 ~15% elementwise  
- 不该问：相对 **带 formula 的 Without-L oracle 67%** 为何更差  

「只回归 lattice、不做原子扩散」**并不更简单**：在 **缺组成、缺全局搜索** 时，盲 indexing 比条件生成更难。RealPXRD 的 lattice 头吃到了完整原子图；你们没有。

相对 Mc，NN 落后的主因（与既往 R0/R5 一致）：

1. **单点回归 + 窄邻域 Top-K**（Bravais snap / 尺度变体），不是多晶系参数空间搜索  
2. **raw 几何不足**：angle MAE ~8.7°，非立方 elementwise ~3%，单斜 ~0%  
3. 严口径下 FOM 偏爱半胞，**elementwise Top-1 可到 0%**（R5）  
4. Loss（SmoothL1 / Huber）未直接对齐峰位 R 或 0.05/3° gate  

---

## 6. 能从 RealPXRD 学什么 / 不学什么

### 6.1 应该学

| 启示 | 落地含义 |
|---|---|
| **任务拆分** | Indexing ≠ 结构生成；Without-L 不做 indexing 引擎 |
| **峰表契约** | 变长 `(2θ, I)`；与 241113 / MP100 仿真口径对齐 |
| **Encoder 资产** | Bert XRD encoder 可作预训练起点；indexing 仍需更强峰几何表征（histogram 线已证明 peak-token 不够） |
| **矩阵表示** | 3×3 / 度量张量思路 → 已演化为 matrix6；晶系可作后验/搜索维，非硬前置 |
| **多样本 ≠ 单点** | 单次点估计不够；indexing 侧应翻译为 **更宽的晶胞假设搜索 + 峰拟合打分**（Mc 哲学），而非带 formula 的扩散 |

### 6.2 明确不要学

| 不要学 | 原因 |
|---|---|
| 推理喂 formula / atom_num | 违反 indexing；Mc 不用 |
| 用 Without-L lattice% 当 indexing 基线 | 任务错配 |
| 联合 atom 扩散当 indexing 主路径 | 结构生成 |
| 无峰打分的 oracle Top-K 当产品指标 | 要可排序 Top-1 |
| 照搬 Flow Matching 噪声 MSE 当 indexing loss | 与 Bragg 峰拟合不对齐 |

### 6.3 一句话

> RealPXRD 教的是 **数据 / 峰表 / 矩阵表示 / 任务边界 /「单点不够」**；  
> Indexing 本体的老师是 **McMaille（多晶系搜索 + R/M20）**，不是 Without-L。

---

## 7. 对当前攻关的含义

1. **对照表清洗**：indexing 严口径对照只保留 Mc / JADE / 本 NN；RealPXRD Without-L 标注为「结构探索，非 indexing」。  
2. **Gate 不变**：产品 gate 继续用 **elementwise @ 0.05/3°**（见 R5）；mapping 仅作诊断。  
3. **主矛盾不变**：抬 **raw 非立方几何** + 把 Top-K 从「单锚点尺度邻域」升级为更接近 Mc 的 **多假设搜索 + 峰几何打分**；不要用喂 formula 走捷径。  
4. **With-L 定位**：indexing 达标后的下游结构生成，与本次 Without-L 复测脱钩。

---

## 8. 相关文档与脚本索引

| 类型 | 路径 |
|---|---|
| 本次评测脚本 | `archive/RealPXRD-Solver/scripts/eval_mp100_without_l_lattice.py` |
| 本次结果 | `archive/RealPXRD-Solver/实验/mp100_without_l_lattice/mp100_without_l_ltol0.05_atol3.json` |
| RealPXRD 深度调研 | `docs/开发日志/20260707-RealPXRD-Solver深度调研.md` |
| RealPXRD vs Mc 架构讨论 | `docs/开发日志/20260708-RealPXRD晶格回归与McMaille算法调研-架构讨论.md` |
| 历史三引擎与 formula 口径 | `docs/开发日志/起点.md` |
| 冠军严口径现状 | `docs/实验记录/20260713-R5-Diag-严口径TopK与FOM.md` |
| MP100 严口径护栏 | `docs/实验记录/20260709-A1-严口径评测护栏.md` |

---

## 9. 摘要

| 问题 | 答案 |
|---|---|
| Without-L @ MP100 Top-20 lattice？ | **67% mapping / 62% elementwise**（K=20 oracle，峰+primitive formula） |
| Top-1？ | **34% / 32%** |
| 是否靠 lattice 标准化？ | **否**；原始 3×3 flow，无 Niggli |
| 能否当 indexing 基线？ | **否**；泄漏 formula，任务是结构生成 |
| Indexing 正统老师？ | **McMaille / JADE**（仅峰+λ，~66–68% 严口径 Top-1） |
| 我们 ~15% 该对谁焦虑？ | 对 Mc/JADE，不对 Without-L 67% |
| 从 RealPXRD 学什么？ | 任务拆分、峰表/数据口径、矩阵表示、「单点不够→要搜索」；不学 formula 条件生成当 indexing |
