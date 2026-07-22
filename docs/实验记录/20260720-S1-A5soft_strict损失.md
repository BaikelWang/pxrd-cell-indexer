# S1 实验报告：A5 metric-aligned soft-strict loss（2026-07-20）

> 承接 `docs/开发日志/20260720-CellIndexing-后续优化方案v4.md` §5。  
> 基线：A3-G1 生产栈（`configs/scale_100k_a3_g1_gstar6.yaml`，valid1400 strict 三 seed mean 43.07%）。  
> 目的：验证「惩罚最差自由维」的 metric-aligned 辅助损失能否低成本抬升 strict。

---

## 1. 一句话结论

**A5 soft_strict 在 P0-700 三组超参网格上均未过 95% Gate（最高 56.0%），已淘汰，未进 100k。** 结合已淘汰的 G2（decoded_cell 均值损失，best 78.6%），进一步证明 decode 空间的辅助损失（无论均值还是最差维形式）在当前 SmoothL1(gstar6) 主干 + 700 样本满秩 overfit 场景下都会显著拖慢甚至阻断收敛；metric 对齐 loss 这条杠杆基本走到头。**后续主线转向 S3(A4) 鲁棒课程 + S2 独立候选生成。**

---

## 2. 实现

- `src/pxrd_cell_indexing/losses.py`：新增 `mode="soft_strict"` 与 `soft_strict_loss()`。
  - 归一化误差：长度 `|Δa|/(0.05·a_truth)`，角度 `|Δα|/3°`（与 strict Gate 边界对齐，1.0 = 恰好在容差线上）。
  - 仅在 `CS_PHYS_PARAM_MASK` 标记的自由维参与 log-sum-exp；固定（对称锁定）维通过有限负哨兵值排除，不进入 softmax（已单测验证不受污染）。
  - `loss_total = loss_reg(SmoothL1 on gstar6) + physical_weight · soft_strict_loss(τ)`。
- `LossWeights.soft_strict_tau`（默认 0.5）+ `training/config.py` 解析 `loss.soft_strict_tau`。
- 单测（`tests/test_losses.py`）：
  - `test_soft_strict_loss_finite_gradient`：端到端 finite grad。
  - `test_soft_strict_loss_requires_crystal_system_idx`：缺 CS 时报错。
  - `test_soft_strict_loss_penalizes_worst_dim_direction`：恶化「已是最差维」比恶化「已经很好的维」惩罚更重（方向正确）。
  - `test_soft_strict_loss_ignores_masked_dims`：固定维即使预测离谱也不影响 loss（mask 生效）。
  - 4 个新测试 + 原 24 个全部通过。

---

## 3. P0-700 Gate 结果

| 配置 | τ | λ (`physical_weight`) | best strict（≤1200 ep） | 收敛形态 |
|------|:--:|:--:|:--:|------|
| 原始网格点 | 0.5 | 0.1 | **23.1%**（ep1180） | 一直卡在 20% 出头，无平台突破迹象 |
| 降权 | 0.5 | 0.02 | **55.7%**（ep1187） | ep1000 后收敛到 ~56% 平台，不再上升 |
| 软化 τ | 2.0 | 0.1 | **56.0%**（ep1167） | 同上，几乎一致的天花板 |
| 对照：纯 SmoothL1(gstar6) 基线 | — | — | **96%**（ep935，A3-G1 P0） | ep700 后陡峭爬升到收敛 |

三组末期（ep~1200）诊断：

| 配置 | `loss_reg` | `loss_phys` | per-dim pass rate（a/b/c/α/β/γ） |
|------|:--:|:--:|------|
| λ=0.1,τ=0.5 | 0.092 | 3.12 | 48/47/45/48/57/59% |
| λ=0.02,τ=0.5 | 0.019 | 1.59 | 80/77/77/74/83/85% |
| λ=0.1,τ=2.0 | 0.023 | 3.81 | 80/79/78/76/83/85% |

**梯度冲突诊断**（取 λ=0.1,τ=0.5 训练中 checkpoint，手动对同一 batch 分别 backward `loss_reg` 与 `loss_phys`）：

```
g_reg norm  = 0.0856
g_phys norm = 1.2505
cosine(g_reg, g_phys) = +0.29   # 不对抗，非负相关
g_total(实际反传) norm ≈ ‖g_reg + 0.1·g_phys‖，数值吻合
```

结论：**不是梯度互相抵消的实现 bug**——两个损失方向基本一致（弱正相关）。真实瓶颈是：per-dim pass rate 已到 74–85%（比 A3-G1 基线在同等训练量下同期更高），但 700 样本 × 6 维的 **AND 合取**仍卡在 ~56%，说明「惩罚最差维」这一目标函数形态，相比均值 SmoothL1，在小样本满秩 overfit 场景下天然更难让**所有维、所有样本同时**压进容差带（每步梯度集中在各样本各自的"当前最差维"，样本间目标不一致，减慢联合收敛）。

---

## 4. 与 G2 的对比

| | G2（`decoded_cell`，均值 SmoothL1） | S1/A5（`soft_strict`，最差维 log-sum-exp） |
|---|---|---|
| λ | 0.05 | 0.02 / 0.1 |
| P0 best strict | 78.6%（ep1183） | 56.0%（最优组） |
| Gate 结论 | 未过（<95%），已淘汰 | 未过（<95%），已淘汰 |

A5 的最差维聚焦目标比 G2 的均值目标**更难通过纯 overfit 检验**，即"惩罚最差维"这一更贴近 strict 指标本身的目标函数，代价是训练动力学更陡峭、更不利于小样本满秩收敛。两次独立失败共同指向：**在当前 SmoothL1(gstar6) 主干上叠加 decode 空间辅助项这条路线（表 2.3 杠杆 A）收益已耗尽**。

---

## 5. 决策与对 v4 规划的影响

- **A5 soft_strict 淘汰**，100k seed42 训练跳过（P0 Gate 未过，按 §5.3 规则不进入 100k）。
- 保留三份 P0 配置/日志供复盘：
  - `configs/overfit700_a5_s1_soft_strict.yaml`（τ=0.5, λ=0.1）
  - `configs/overfit700_a5_s1_soft_strict_lam002.yaml`（τ=0.5, λ=0.02）
  - `configs/overfit700_a5_s1_soft_strict_tau2.yaml`（τ=2.0, λ=0.1）
  - 已备好但未使用：`configs/scale_100k_a5_s1_soft_strict_seed42.yaml`（P0 未过，未运行）
- **v4 已更新**：`docs/开发日志/20260720-CellIndexing-后续优化方案v4.md` §5.4 / §3 / §10 / §11 / §12 / §9 已按本结果重排——**S3(A4) 鲁棒课程升级为唯一主线**，S2 仍待独立 q-search 原型证明头寸后才投入，S1 方向（含 S5b 支线的其他 decode-space loss 变体）不再继续尝试。
