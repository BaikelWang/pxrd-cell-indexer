# 2026-07-08 — Decision A：去晶系分类头 + Bravais 几何 Snap Top-K

> **目的**：移除训练/Top-K 中的晶系分类头，改为纯 lattice 回归 + McMaille 式 Bravais 多假设 snap 排序；晶系仅作事后观测指标。与 `scale_100k_seed42`（旧架构）做严格 ablation 对比。
> **配置**：[`configs/scale_100k_no_cs.yaml`](../../configs/scale_100k_no_cs.yaml)
> **Checkpoint**：`results/experiments/scale_100k_no_cs_seed42/checkpoints/best.pt`（epoch 5）
> **前置依据**：[`20260708-Bravais原胞约束验证.md`](20260708-Bravais原胞约束验证.md)

---

## 1. 架构变更摘要

| 组件 | 旧架构 (`scale_100k_seed42`) | 新架构 (`scale_100k_no_cs_seed42`) |
|---|---|---|
| 模型头 | 晶系分类头 + 6 参数回归头 | **仅** 6 参数回归头 |
| 训练 loss | CE + SmoothL1（1:1） | **仅** SmoothL1 |
| Top-K 生成 | 分类 argmax + 次晶系标签 + 倍胞变体 | **8 条 Bravais snap 假设**按几何偏差排序 + 倍胞变体 |
| 晶系信息 | 参与训练、Top-K 排序、joint 指标 | **不参与**训练/排序；事后 `infer_crystal_system_idx_from_lattice` 推断 |

新增模块：[`src/pxrd_cell_indexing/model/bravais.py`](../../src/pxrd_cell_indexing/model/bravais.py)（8 假设：`cubic_P/F/I`、`tetragonal_P`、`orthorhombic_P`、`hex_trig_P`、`trigonal_R`、`identity`）

---

## 2. 实验设置（严格 ablation）

| 项 | 值 |
|---|---|
| 训练数据 | `train100k_seed42.jsonl`（100,000 条，与基线相同） |
| 验证集 | `valid1400_seed42.jsonl`（1,400 条，7 晶系各 200） |
| seed | 42 |
| 超参 | 与 `scale_100k.yaml` 完全一致（lr、batch、epoch、augment 等） |
| 唯一变量 | 架构（去分类头 + Bravais Top-K） |
| 训练耗时 | ~11.7 min（10 epoch，early stop patience=5，best@epoch 5） |
| 单测 | **51/51** pytest 通过 |

> **Checkpoint 兼容性**：旧 `scale_100k_seed42` checkpoint 含 `crystal_system_head` 权重，无法加载到新 `IndexingModel`；属预期行为，不做迁移。

---

## 3. valid1400 核心对比

| 指标 | 旧 `scale_100k_seed42` | 新 `scale_100k_no_cs_seed42` | Δ |
|---|---:|---:|---:|
| **top1_lattice_match_rate**（主指标） | **39.7%** | **39.1%** | **−0.6pp** |
| topk_lattice_match_rate | 99.3% | 99.3% | 0.0pp |
| lattice_mae | 5.54 | 5.55 | +0.01 |
| length_mape | 16.2% | 16.0% | −0.2pp |
| crystal_system_accuracy | 65.1% | 14.2% | −50.9pp |
| top1_joint_match_rate | 27.4% | 4.1% | −23.3pp |

结果文件：
- 基线：[`results/valid1400_scale_100k_seed42.json`](../../results/valid1400_scale_100k_seed42.json)
- 新架构：[`results/valid1400_scale_100k_no_cs_seed42.json`](../../results/valid1400_scale_100k_no_cs_seed42.json)

### 解读

1. **主指标基本持平**：`top1_lattice_match_rate` 39.7% → 39.1%（−0.6pp），在噪声范围内可视为**无显著退化**。去掉分类头并未损害 lattice 回归能力。
2. **Top-K 召回不变**：`topk_lattice_match_rate` 维持 99.3%，说明 Bravais snap 候选池与旧分类驱动候选池在几何召回上等效。
3. **晶系/joint 指标大幅下降是预期行为**：新架构不再训练分类头，事后 Bravais 几何推断的 `crystal_system_accuracy`（14.2%）和 `top1_joint_match_rate`（4.1%）**仅作诊断**，不参与训练或 checkpoint 选择。旧架构的 65.1%/27.4% 含分类头直接监督，不可横向比较。

---

## 4. 真解漏斗分解（decompose_joint）

| 漏斗阶段 | 旧架构 | 新架构 | Δ |
|---|---:|---:|---:|
| lattice_top1 | 39.7% | 39.1% | −0.6pp |
| lattice_in_pool（真解在 Top-20） | 99.3% | 99.3% | 0.0pp |
| crystal_system_top1（事后推断） | 64.9% | 8.9% | −56.0pp |
| joint_top1 | 27.4% | 4.1% | −23.3pp |
| joint_in_pool | 76.9% | 31.4% | −45.5pp |
| joint_in_pool_not_top1 | 49.4% | 27.3% | −22.1pp |
| fail_cs_wrong_lattice_ok_top1 | 12.3% | 35.1% | +22.8pp |
| fail_both_bad_top1 | 22.8% | 56.1% | +33.3pp |

结果文件：
- 基线：[`results/decompose_joint_scale_100k_seed42.json`](../../results/decompose_joint_scale_100k_seed42.json)
- 新架构：[`results/decompose_joint_scale_100k_no_cs_seed42.json`](../../results/decompose_joint_scale_100k_no_cs_seed42.json)

### 关键观察

- **真解在 Top-20 池的比例未下降**（99.3% 持平），Decision A 的核心担忧（去掉分类头驱动候选生成会降低池内召回）**未发生**。
- `fail_cs_wrong_lattice_ok_top1` 从 12.3% 升至 35.1%：大量样本 lattice Top-1 正确但事后晶系推断错误——符合「晶系从 lattice 形状难以可靠判定」的预期（尤其 hex/trig/monoclinic/triclinic 组）。
- `joint_in_pool` 下降主要因事后晶系推断不准，而非 lattice 候选池质量下降。

---

## 4b. 纯 Lattice 候选排名分布（新增，2026-07-08 补充分析）

为区分「候选生成覆盖率不足」vs「排序打分不准」两种可能，给 `decompose_joint.py` 增加了**纯 lattice（不受晶系诊断门控）候选排名分布**统计（`lattice_rank_when_in_pool`，条件于真解已在 Top-20 池内）：

| 池内排名 | 占比 | 累计 |
|---|---:|---:|
| rank 1 | 39.4% | 39.4% |
| rank 2 | 0.3% | 39.7% |
| rank 3 | 22.0% | 61.7% |
| rank 4 | 29.7% | 91.4% |
| rank 5 | 6.4% | 97.8% |
| rank 6+ | 2.2% | 100% |

结果文件：[`results/decompose_joint_scale_100k_no_cs_seed42.json`](../../results/decompose_joint_scale_100k_no_cs_seed42.json)（字段 `lattice_rank_when_in_pool`）

**判定**：
- **候选生成没问题**：真解入池率 99.3%，与旧架构持平，Bravais snap 策略覆盖率充分。
- **排序打分有问题**：真解入池后仅 39.4% 排到 rank 1，91.4% 集中在 rank 1/3/4——大概率是当前 `confidence = 1/(1+score)` 打分公式或倍胞/子胞变体插入顺序把真解挤到固定的次优槽位，而非随机分散（否则不会这样集中在 3-4 个 rank）。
- **Decision A 的有效性判断应聚焦 Top-20（`topk_lattice_match_rate`）而非 Top-1**：Top-1 短板是独立于「是否用晶系分类头」的排序问题，可留到后续单独优化（不阻塞 Decision A 落地判断）。

> **PM 决策（2026-07-08）**：checkpoint 选择的 `best_metric` **暂不改动**，继续用 `top1_lattice_match_rate`（Top-20 已接近饱和 99.3%，区分度不足，无法用于 early-stop/best 选择）；仅在**实验评估/决策口径**上以 Top-20 为 Decision A 有效性的主要判据。排序优化留作后续独立课题（不在本次 Decision A 范围内）。

## 5. 训练曲线摘要

| epoch | valid top1_lattice_match_rate | valid loss |
|---:|---:|---:|
| 1 | 36.4% | 0.244 |
| 5（best） | **39.2%** | 0.218 |
| 10 | 39.1% | 0.217 |

best checkpoint 选择依据：`top1_lattice_match_rate`（pymatgen real match）。

---

## 6. 结论与决策建议

| 判定项 | 结论 |
|---|---|
| 主指标（Top-1 lattice match） | ✅ **持平**（−0.6pp，可忽略） |
| Top-K 召回 | ✅ **持平**（99.3%） |
| 架构简化收益 | ✅ 去掉分类头、CE loss、不确定性加权；Top-K 逻辑更可解释（McMaille 式几何穷举） |
| 晶系诊断能力 | ⚠️ 事后推断仅 14.2%，**不作为产品 KPI** |
| Checkpoint 选择 | ✅ 继续以 `top1_lattice_match_rate` 为 best_metric |

**Decision A 采纳**：新架构作为后续 scaling（500k/全量）的默认架构。旧 `scale_100k_seed42` checkpoint 保留为历史基线数值，不做迁移。

---

## 7. 遗留与后续

- [ ] 500k/全量训练用 `scale_*_no_cs.yaml` 系列配置
- [ ] `identity_penalty_score` 敏感性检查（当前默认 1.0）
- [ ] MP100 benchmark 评测（新架构 checkpoint）
- [ ] Decision B（3×3 矩阵回归）— 显式 deferred
- [ ] **候选排序优化**（D31 发现的独立课题）：真解入池后 91.4% 集中在 rank 1/3/4，说明打分公式/变体插入顺序有系统性偏差；后续可尝试用真实 lattice match 作为排序目标重新校准 confidence 公式或变体插入策略
