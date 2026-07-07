# Step 1 — 需求澄清

> **状态**：🟡 核心目标与输入/输出口径已确认；损失函数与 Top-K 实现待设计  
> **最后更新**：2026-07-07

---

## PM 已确认（2026-07-06）

| # | 项 | 结论 |
|---|---|---|
| 1 | **任务目标** | 训练一个模型：**输入 PXRD → 输出晶系 + lattice** |
| 2 | **训练数据源** | [`alex_aflow_oqmd_mp`](../../../alex_aflow_oqmd_mp/)（LMDB，见 `datasets/`） |
| 3 | **Benchmark** | [`data/MP-100samples-benchmark/`](../data/MP-100samples-benchmark/)（100 条分层 CIF） |
| 4 | **指标与历史背景** | 详见 [`docs/开发日志/起点.md`](开发日志/起点.md) |

---

## 业务目标（一句话）

训练一个 **PXRD → 晶系 + 晶胞参数 (lattice)** 的神经网络模型，在 MP100 benchmark 上与 McMaille / JADE9 等专用 indexing 引擎对照。

## 用户故事 / 使用场景

给定一条 PXRD 谱（或预处理后的峰表），模型预测：

- **晶系**（cubic / tetragonal / orthorhombic / …）
- **晶胞六参数** `(a, b, c, α, β, γ)`

用于 Cell Indexing 环节，为下游 With L 结构推理 / Refine 提供晶格先验。

## 输入 → 输出

| 维度 | 说明 |
|---|---|
| **输入** | RealPXRD 风格变长峰表：`pxrd_x` / `pxrd_y` / `peak_num`；前端可提供峰表 + λ（PM D4） |
| **输出** | **Top-K** 候选：晶系（分类）+ **primitive** lattice 六参数（回归）+ confidence（PM D12） |
| **训练标签** | `p_lattice_matrix` → 六参数；晶系由 **primitive** 结构推导（PM D1/D7） |

## 数据源

### 训练集 — `alex_aflow_oqmd_mp/datasets/`

| 文件 | 规模 | 说明 |
|---|---|---|
| `pxrd_241113_train.lmdb` | ~608 万 | **主训练集**（PM D2 确认） |
| `pxrd_241113_valid.lmdb` | 25,551 | 调参 / 早停（PM D5） |
| `pxrd_241113_test.lmdb` | 25,552 | 可选 hold-out；**最终对照用 MP100** |
| `pxrd_241119_*` | atom_num ≥ 25 | 子集 |
| `pxrd_241120_*` | atom_num ≥ 50 | 子集 |

LMDB 单条字段（参考 `codes/241113_save_pxrd_data.py`）：

- `pxrd_x`, `pxrd_y` — PXRD 2θ / 强度
- `p_lattice_matrix` — primitive lattice matrix（**训练标签来源**，PM 2026-07-07 确认）
- `p_atom_type`, `p_atom_pos` — 原胞结构（本任务不直接预测）

详见 [`alex_aflow_oqmd_mp/datasets/数据集说明.txt`](../../../alex_aflow_oqmd_mp/datasets/数据集说明.txt)。

### Benchmark — `data/MP-100samples-benchmark/`

- 100 条分层 CIF（`mp-*.cif`）
- 对应历史 MP100 Tier-0 评测集
- **评测**：CIF truth **转为 primitive** 后与模型预测比较（PM D6）
- 主指标与口径见 [`docs/开发日志/起点.md`](开发日志/起点.md)

## 验收标准

### 功能（benchmark 侧，摘自起点.md）

| 指标 | 容差 / 说明 |
|---|---|
| **lattice match rate** | lt ol=0.3, atol=10°（与 Mc/JADE 三引擎对照主指标） |
| **strict Top-1 recall** | 真解是否为排名第一候选 |
| **晶系准确率** | 预测晶系 vs CIF truth |
| **recall（候选池）** | 真胞是否进入 Top-N 候选池 |

> ⚠️ 不同历史报告容差/分母不同，评测脚本须统一口径后再横比。

### 对照基线（ideal 峰，起点.md §4）

| 引擎 | lattice match |
|---|---:|
| McMaille | ~76.4% |
| JADE9 | ~72.5% |
| RealPXRD Without L | ~5%（**不能**替代 indexing） |

本任务 NN 模型的合理期望：在 MP100 上 **接近或超越** 专用 indexing 引擎的 lattice match；RealPXRD Without L 不作为 indexing 基线。

### 非功能

_（待后续讨论：推理延迟、GPU 显存、部署形态）_

## Out-of-Scope

摘自 [`docs/开发日志/起点.md`](开发日志/起点.md) 与 PM 当前方向：

- ❌ 不做 **全结构 de novo 生成**（RealPXRD Without L 路线）
- ❌ 不做多相混合 QPA
- ❌ 不做 Rietveld 精修
- ⚠️ 端到端 NN 替代 Mc 的历史研判为「证据不足」（186）；本任务仍尝试训练，但需 realistic 预期

## 约束

| 类型 | 说明 |
|---|---|
| 训练数据 | `pxrd_241113_*.lmdb`（正式全量）；**初实验从 train 抽样**（PM D2） |
| Benchmark | `data/MP-100samples-benchmark/`（100 CIF）；与 valid **分工**（PM D5） |
| 模型 baseline | **RealPXRD encoder** + 晶系 / lattice Top-K 头；砍掉原子结构生成（PM D8/D13） |
| 算力 | 当前 1× RTX 4090 D（24 GB） |
| 历史参考 | [`docs/开发日志/起点.md`](开发日志/起点.md) — Cell Indexing 全历程复盘 |

## PM 已确认（2026-07-07）

| # | 项 | 结论 |
|---|---|---|
| D1 | lattice 训练标签 | **primitive 六参数** |
| D2 | 训练子集 | **`pxrd_241113_*`**；初实验 train **抽样 10k** |
| D3 | PXRD 输入 | **直接用 LMDB `pxrd_x/y`** |
| D4 | 输入形态 | **RealPXRD 风格变长峰表**；前端提供峰表 + λ |
| D5 | valid vs MP100 | valid 调参；**MP100 最终对照** |
| D6 | benchmark 评测 | **CIF truth → primitive** 再比 |
| D7 | 晶系标签 | **primitive 推导** |
| D8 | 模型 baseline | **RealPXRD `BertModel` encoder**（512-d，变长峰表输入） |
| D10 | 少样本规模 | **10,000 条**；atom_num < 25；各晶系尽量均匀；固定 seed |
| D11 | 训练输入 / 仪器泛化 | 训练输入 = **Cu Kα 理想模拟峰表**；**仪器泛化后续做** |
| D12 | 输出形态 | **Top-K lattice + 晶系候选** |
| D13 | RealPXRD 裁剪 | 保留 encoder/lattice 思路；砍掉原子结构生成 |
| D14 | Top-K / 峰数 | 参考 RealPXRD `num_evals`；峰表只做 `y>5` 过滤，无 180 峰硬上限 |

详见 [`docs/开发日志/20260707-PM决策与待确认清单.md`](开发日志/20260707-PM决策与待确认清单.md)。

## 待后续讨论

1. **损失函数**（D9）：多任务设计与权重
2. **Top-K 实现方式**：多头 / set prediction / 采样式候选
3. **峰数计算成本**：RealPXRD 无硬截断；是否设置 `max_peaks` 需设计
4. **仪器泛化**（D11，后续）：多 λ 增强 / λ 条件 / d 间距等
