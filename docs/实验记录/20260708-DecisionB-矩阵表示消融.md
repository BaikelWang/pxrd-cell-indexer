# 2026-07-08 — Decision B：3×3 矩阵表示消融

> **目的**：在方案 A（去晶系分类头 + Bravais snap Top-K）基础上，独立验证 lattice 回归目标从 (a,b,c,α,β,γ) 改为「规范 3×3 矩阵 6 自由分量」是否提升精度。
> **配置**：[`configs/scale_100k_no_cs_matrix6.yaml`](../../configs/scale_100k_no_cs_matrix6.yaml)
> **Checkpoint**：`results/experiments/scale_100k_no_cs_matrix6_seed42/checkpoints/best.pt`（epoch 11）
> **对照基线**：[`20260708-DecisionA-去晶系分类头.md`](20260708-DecisionA-去晶系分类头.md)（`scale_100k_no_cs_seed42`）

---

## 1. 架构变更摘要

| 组件 | 方案 A（`scale_100k_no_cs_seed42`） | 方案 B（`scale_100k_no_cs_matrix6_seed42`） |
|---|---|---|
| 模型头 | 6 参数回归头 | **不变**（仍输出 6 个数） |
| 训练 loss | SmoothL1 | **不变** |
| 标签归一化 | `LatticeNormalizer`（log+zscore 长度，zscore 角度） | `MatrixLatticeNormalizer`（规范矩阵 6 分量独立 z-score） |
| 反归一化 | 直接还原 (a,b,c,α,β,γ) | 6 分量 → 补零成 3×3 → `lattice_lengths_angles` → (a,b,c,α,β,γ) |
| Top-K / Bravais / 评测 | Bravais snap + pymatgen | **完全不变**（下游仍吃标准 6 参数） |

新增模块：
- [`src/pxrd_cell_indexing/data/normalization.py`](../../src/pxrd_cell_indexing/data/normalization.py)：`MatrixLatticeNormalizer`、`build_lattice_normalizer`
- [`scripts/compute_matrix6_stats.py`](../../scripts/compute_matrix6_stats.py)
- 统计文件：`data/processed/lattice_matrix6_stats_100k_seed42.json`

规范矩阵 6 自由分量（与 `geometry.py::lattice_params_to_matrix` 一致）：
`(a_x, a_z, b_x, b_y, b_z, c_z)`，对应 `vector_a=[a_x,0,a_z]`、`vector_b=[b_x,b_y,b_z]`、`vector_c=[0,0,c_z]`。

---

## 2. 实验设置（严格 ablation）

| 项 | 值 |
|---|---|
| 训练数据 | `train100k_seed42.jsonl`（100,000 条） |
| 验证集 | `valid1400_seed42.jsonl`（1,400 条） |
| seed | 42 |
| 超参 | 与 `scale_100k_no_cs.yaml` 完全一致 |
| **唯一变量** | `data.representation: matrix6` + matrix6 统计文件 |
| 训练耗时 | ~26.4 min（11 epoch best，early stop patience=5） |
| 单测 | **57/57** pytest 通过 |

---

## 3. valid1400 核心对比

| 指标 | 方案 A | 方案 B | Δ |
|---|---:|---:|---:|
| **top1_lattice_match_rate**（主指标） | 39.1% | **40.3%** | **+1.2pp** |
| topk_lattice_match_rate | 99.3% | 99.1% | −0.2pp |
| lattice_mae | 5.55 | 5.43 | −0.12 |
| length_mape | 16.0% | 16.0% | ≈0 |
| crystal_system_accuracy（事后推断，诊断） | 14.2% | 15.4% | +1.2pp |
| top1_joint_match_rate（诊断） | 4.1% | 4.4% | +0.3pp |

结果文件：
- 方案 A：[`results/valid1400_scale_100k_no_cs_seed42.json`](../../results/valid1400_scale_100k_no_cs_seed42.json)
- 方案 B：[`results/valid1400_scale_100k_no_cs_matrix6_seed42.json`](../../results/valid1400_scale_100k_no_cs_matrix6_seed42.json)

### 解读

1. **Top-1 有小幅提升**（+1.2pp）：矩阵表示在相同架构/数据/seed 下略优于角度直接回归，说明规范矩阵参数化对回归精度有一定帮助。
2. **Top-20 基本持平**（99.1% vs 99.3%）：候选池覆盖率不受表示方式影响，与 D31 结论一致——瓶颈仍在排序而非候选生成。
3. **回归 MAE 略降**（5.55→5.43）：与 Top-1 提升方向一致。

---

## 4. 纯 Lattice 候选排名分布（条件于真解在池内）

| 池内排名 | 方案 A | 方案 B |
|---|---:|---:|
| rank 1 | 39.4% | **40.6%** |
| rank 3 | 22.0% | 21.8% |
| rank 4 | 29.7% | 23.6% |
| rank 5 | 6.4% | 9.6% |

结果文件：
- 方案 A：[`results/decompose_joint_scale_100k_no_cs_seed42.json`](../../results/decompose_joint_scale_100k_no_cs_seed42.json)
- 方案 B：[`results/decompose_joint_scale_100k_no_cs_matrix6_seed42.json`](../../results/decompose_joint_scale_100k_no_cs_matrix6_seed42.json)

矩阵表示使真解排到 rank 1 的条件概率从 39.4% 升至 40.6%（+1.2pp），与 Top-1 提升一致；rank 3/4 的集中模式仍存在，排序优化仍是独立遗留课题。

---

## 5. 训练曲线摘要

| epoch | valid top1_lattice_match_rate | valid loss |
|---:|---:|---:|
| 5 | 38.6% | 0.219 |
| 11（best） | **40.9%** | 0.216 |
| 最终 | 40.3%（eval_valid 全量） | — |

---

## 6. 结论与决策建议

| 判定项 | 结论 |
|---|---|
| 矩阵表示是否可行 | ✅ **可行**，实现成本低（仅 normalizer 切换，下游不变） |
| Top-1 是否提升 | ✅ **小幅提升**（+1.2pp，39.1%→40.3%） |
| Top-20 是否变化 | ≈ **不变**（99.3%→99.1%） |
| 是否优于方案 A | ✅ **略优**，可作为后续 scaling 的默认表示 |
| 与 RealPXRD 9 维无约束矩阵的关系 | 本次为规范 6 分量等价重参数化；无约束 9 维 + Gram loss 留作可选后续 |

**Decision B 采纳（温和版）**：后续 500k/全量训练默认使用 `representation: matrix6` + `MatrixLatticeNormalizer`。方案 A（angles 表示）保留为历史基线，不做迁移。

---

## 7. 遗留与后续

- [ ] 500k/全量训练用 `scale_*_no_cs_matrix6.yaml` 系列
- [ ] 候选排序优化（D31 遗留，与表示方式无关）
- [ ] 可选：无约束 9 维矩阵 + 度量张量 `L·Lᵀ` loss（更激进版 Decision B）
