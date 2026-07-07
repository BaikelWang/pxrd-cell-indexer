# 2026-07-06 — 训练数据调研：`alex_aflow_oqmd_mp`

> **当前步骤**：Step 2 前置 — 数据源摸底  
> **调研方式**：只读扫描 LMDB（`gzip` + `pickle`），统计条数、字段结构、PXRD 特征；**未修改任何数据文件**  
> **数据路径**：[`../../../alex_aflow_oqmd_mp/datasets/`](../../../alex_aflow_oqmd_mp/datasets/)

---

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| 有多少条？ | 主集 `pxrd_241113_*` 共 **6,139,286** 条（train 6,088,183 + valid 25,551 + test 25,552）；另有 atom_num 过滤子集与 1 万条分层 test |
| 每条长什么样？ | 5 字段：`p_atom_type`、`p_atom_pos`、`p_lattice_matrix`、`pxrd_x`、`pxrd_y` |
| 有没有 PXRD？ | **有**，但是 **pymatgen 模拟 PXRD**（2θ 5°–80°，强度归一化至 max=100），**不是实验谱** |
| 能否直接用于本任务？ | **可用作训练输入 + lattice 标签来源**，但须先对齐 **PXRD 与 lattice 的晶胞口径**（见 §4） |

---

## 2. 数据集规模（实测，2026-07-06）

| 文件 | 条数 | 大小 |
|---|---:|---:|
| `pxrd_241113_train.lmdb` | 6,088,183 | ~14.0 GB |
| `pxrd_241113_valid.lmdb` | 25,551 | ~57 MB |
| `pxrd_241113_test.lmdb` | 25,552 | ~56 MB |
| `pxrd_241119_train.lmdb` | 132,581 | ~873 MB |
| `pxrd_241119_valid.lmdb` | 482 | ~3.4 MB |
| `pxrd_241119_test.lmdb` | 514 | ~3.4 MB |
| `pxrd_241120_train.lmdb` | 42,701 | ~355 MB |
| `pxrd_241120_valid.lmdb` | 126 | ~1.2 MB |
| `pxrd_241120_test.lmdb` | 116 | ~1.1 MB |
| `pxrd_250112_test.lmdb` | 10,000 | ~22 MB |
| `temp_valid_origin.lmdb` | 51,103 | ~36 MB |
| `temp_valid.lmdb` / `temp_test.lmdb` | 25,551 / 25,552 | ~15 MB |

说明（摘自 `datasets/数据集说明.txt`）：

- **241113**：去重 + 删除无效 valid/test 后的主集
- **241119**：241113 基础上 **atom_num ≥ 25**
- **241120**：241119 基础上 **atom_num ≥ 50**
- **250112_test**：从 241113 test 按 atom_num 区间（25/50/75/100）分层等比采样 **1 万条**

当前目录 **未包含** 文档中提到的 `hklf_241123/124/126` 系列 LMDB。

---

## 3. 单条数据结构

### 3.1 PXRD 版（`pxrd_241113_*.lmdb`）

每条为 pickle 字典，字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `p_atom_type` | `list[str]` | 原子种类（元素符号） |
| `p_atom_pos` | `ndarray (N, 3)` | 分数坐标 |
| `p_lattice_matrix` | `ndarray (3, 3)` | 晶格矩阵（标签候选） |
| `pxrd_x` | `ndarray` | 2θ 角度（°） |
| `pxrd_y` | `ndarray` | 强度，**最强峰归一化为 100** |

**样例（`pxrd_241113_test.lmdb` index=0，Ni-P 化合物）**：

- 原子数：4
- PXRD 峰数：24；2θ 范围 11.3°–78.1°
- `pxrd_y` 最大值：100.0

**抽样统计（test 集随机 500 条）**：

| 统计量 | 值 |
|---|---|
| 原子数 | min=3, max=80, median=5, mean≈7.9 |
| PXRD 峰数 | min=7, max=480, median=35, mean≈64.6 |
| 2θ 范围 | min≈5.4°, max=80.0° |

train 集分布（来自 `datasets/说明.txt`，历史分析脚本输出）：

- 常见原子数：4、5、8、12、6（百万级频次）
- 常见晶系：tetragonal、cubic、monoclinic、orthorhombic、trigonal

### 3.2 预处理前原始结构（`temp_valid.lmdb`）

| 字段 | 含义 |
|---|---|
| `atom_type`, `atom_pos`, `lattice_matrix` | 结构 |
| `abc`, `angles` | 晶胞参数 |
| `source` | 数据来源标签 |

### 3.3 数据来源（`temp_valid_origin.lmdb`，51,103 条）

| source | 占比 |
|---|---:|
| aflow | 34.4% |
| oqmd | 10.4% |
| alexandria_*（多 shard 合计） | ~54% |
| matbench_mp_* 等 MP 相关 | <1% |

命名含义：**alex** = Alexandria，**aflow** = AFLOW，**oqmd** = OQMD，**mp** = Materials Project（matbench 子集）。

---

## 4. PXRD 如何生成（关键口径）

生成脚本：[`alex_aflow_oqmd_mp/codes/241113_save_pxrd_data.py`](../../../alex_aflow_oqmd_mp/codes/241113_save_pxrd_data.py)

```python
# 1. 从原始 atom_type / atom_pos / lattice_matrix 建 Structure
# 2. conventional standard → reduced structure
# 3. pymatgen XRDCalculator，2θ ∈ [5°, 80°]，scaled=True
pxrd_x, pxrd_y = cal_pxrd(conventional_structure, 5, 80)
```

写入 LMDB 时：

- **`pxrd_x` / `pxrd_y`**：来自 **conventional → reduced** 结构的模拟谱
- **`p_lattice_matrix` / `p_atom_*`**：来自 **原始 LMDB 的 primitive 字段**（`data_dict['lattice_matrix']` 等），**未替换为 conventional 晶胞**

⚠️ **口径风险（须 Step 2 明确）**：

> 输入 PXRD 与输出 lattice 标签可能对应 **不同晶胞表示**（conventional reduced vs primitive）。  
> 历史上 conventional/primitive 混用曾导致指标失真（见 [`起点.md`](起点.md) §5）。  
> 训练前须 PM 拍板：**统一标签为 conventional 六参数，或在 dataloader 中做晶胞转换后再监督。**

---

## 5. 与本任务的关系

| 本任务需求 | LMDB 对应 | 备注 |
|---|---|---|
| 输入 PXRD | `pxrd_x`, `pxrd_y` | 变长峰表，非固定长度向量；需 Step 2 定预处理（重采样 / padding / 峰表） |
| 输出 lattice | 由 `p_lattice_matrix` 推导 `(a,b,c,α,β,γ)` | 须确认 primitive vs conventional |
| 输出晶系 | 由 lattice 经 `SpacegroupAnalyzer` 推导 | 训练时可在线算或离线缓存 |
| Benchmark | `data/MP-100samples-benchmark/`（100 CIF） | 与训练集 **分布/口径均不同**，评测须单独生成 ideal/deploy 峰 |

**本任务不直接使用**：`p_atom_type`、`p_atom_pos`（结构生成用，Cell Indexing 不预测）。

---

## 6. 子集选用 — 候选方案（待 PM 拍板）

| 方案 | 数据 | 优势 | 劣势 |
|---|---|---|---|
| **A. 主集 241113** | ~608 万 train | 覆盖最全；与历史 pxrd 管线一致 | 多数结构原子数很少（median≈5）；全量训练耗时长 |
| **B. 241119（atom≥25）** | ~13.3 万 train | 结构更大、更接近 MP100 复杂度 | 样本量降两个数量级 |
| **C. 241120（atom≥50）** | ~4.3 万 train | 大胞结构，峰更密 | 样本更少；valid/test 仅百条级 |
| **D. 开发阶段** | 241113 valid/test 或 250112_test（1 万） | 快速 smoke / 调管线 | 不能代表全量分布 |

建议路径（Agent 意见，非最终决定）：

1. **管线开发**：先用 `pxrd_241113_valid.lmdb`（2.5 万）或 `pxrd_250112_test.lmdb`（1 万）跑通 smoke  
2. **正式训练**：在 A vs B 之间由 PM 根据算力（1×4090D 24GB）与目标精度选择  
3. **口径对齐**：无论选哪套，先解决 §4 的 lattice 标签定义

---

## 7. 待确认问题（给 PM）

1. ~~**训练标签 lattice**~~ → **✅ PM 2026-07-07：监督 primitive 六参数**（`p_lattice_matrix`）
2. **默认训练子集**：241113 全量 vs 241119/241120 过滤集？
3. **输入形态**：变长 `(pxrd_x, pxrd_y)` 峰表 vs 固定网格重采样（如 0.02° 步长）？
4. **valid 用途**：241113 valid 与 benchmark MP100 是否分工（valid=开发调参，MP100=最终对照）？
5. **PXRD 与标签对齐**：沿用 LMDB 现成谱（conventional reduced 模拟）vs 用 primitive 重算 PXRD？
6. **Benchmark 口径**：primitive 预测如何与 MP100 CIF（多为 conventional）做 lattice match？

完整清单见 [`20260707-PM决策与待确认清单.md`](20260707-PM决策与待确认清单.md)。

---

## 8. 下一步

- [ ] PM 确认 §7 口径与子集  
- [ ] 更新 `docs/01-design.md`（数据管线 + 模型候选）  
- [ ] 实现只读 LMDB dataloader smoke（≤3 文件，先小样本）

---

## 附录：验证命令

```bash
# 在 alex_aflow_oqmd_mp/codes 下用 LMDBDataset 或等价脚本统计 len(dataset)
# 本次调研于 2026-07-06 在 /nanolab/users/wyx 环境实测，数字与 数据集说明.txt 一致
```
