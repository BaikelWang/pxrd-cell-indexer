# 2026-07-08 — Decision B（修正版）：真正的 9 维无约束矩阵回归

> **背景**：[`20260708-DecisionB-矩阵表示消融.md`](20260708-DecisionB-矩阵表示消融.md) 中的"matrix6"实验实际上只是**目标重参数化**（head 仍输出 6 个数，只是归一化坐标系换成了矩阵的 6 个非零分量），与 RealPXRD 的真实做法（**head 直接输出 9 个自由数，MSE 直接在矩阵元素上算**）不同。本实验补做与 RealPXRD 设计一致的"真矩阵回归"版本。
> **配置**：[`configs/scale_100k_no_cs_matrix9.yaml`](../../configs/scale_100k_no_cs_matrix9.yaml)
> **Checkpoint**：`results/experiments/scale_100k_no_cs_matrix9_seed42/checkpoints/best.pt`（epoch 7）

---

## 1. 与 RealPXRD 的对齐核实

调研 [`archive/RealPXRD-Solver`](../../../../archive/RealPXRD-Solver) 源码确认：

| 环节 | RealPXRD 代码依据 | 结论 |
|---|---|---|
| head 输出维度 | `app/model/cspnet_xrd.py:127`：`self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)` | **9 维无约束**，不经过任何"先构造 6 参数矩阵"的步骤 |
| 输出语义 | `cspnet_xrd.py:166-169`：`lattice_out.view(-1,3,3)` 后 `torch.einsum('bij,bjk->bik', lattice_out, lattices)` | 预测的是作用在当前（噪声）晶格上的**变换矩阵**，不是绝对晶格 |
| loss | `app/model/flow.py:138,165`：`rand_l = torch.randn_like(lattices)`；`loss_lattice = F.mse_loss(pred_l, rand_l)` | **直接在 9 个矩阵元素上算 MSE**（flow matching 去噪损失），全程不出现 (a,b,c,α,β,γ) |
| `L·Lᵀ` 用途 | `cspnet_xrd.py:64-68`：`lattice_ips = lattices @ lattices.transpose(-1,-2)` | 仅作**消息传递的条件输入特征**（旋转不变度量张量），不是回归目标 |

**结论**：RealPXRD 是"矩阵原生"端到端设计（无约束 9 维 head + 矩阵空间 MSE），且用于 flow matching（迭代去噪），不是普通监督回归。我们保留监督回归范式（与既有 ablation 方法论一致，一次只变一个量），但把 **head 输出维度改为 9、loss 直接在 9 维归一化空间算**，这才是与 RealPXRD 设计原则对齐的版本。

---

## 2. 实现方式

### Head：真正输出 9 个数

```27:37:src/pxrd_cell_indexing/model/heads.py
@dataclass(frozen=True)
class HeadConfig:
    embedding_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    output_dim: int = 6
```

`LatticeRegressionHead` 最后一层 `Linear(hidden_dim, config.output_dim)`，`output_dim=9` 时头部真正多出 3 个自由输出通道。

### 目标构造：全 9 元素矩阵（含 3 个结构性恒零位）

标签仍来自 `(a,b,c,α,β,γ)`，用 `lattice_params_to_matrix` 转成完整 3×3 矩阵后**直接 flatten 成 9 维**（不再只挑 6 个非零位）：

```
component[0..8] = matrix.flatten()  # 位置[0,1]=a_y、[2,0]=c_x、[2,1]=c_y 恒为0
```

`Matrix9Normalizer` 对 9 个分量各自独立 z-score；3 个恒零位的方差为 0，`from_stats` 用 `max(std, 1e-8)` 保护，避免除零——这 3 个位置的归一化目标恒为 0，head 可以自由输出噪声在这里，不受硬约束。

### Loss：不变，仍是 SmoothL1，只是维度变成 9

```python
loss_reg = F.smooth_l1_loss(lattice_norm_pred, lattice_norm_target)  # now both shape [B, 9]
```

### 解码：任意 9 维 → 补形成 3×3 → `lattice_lengths_angles`（旋转不变，鲁棒）

```204:215:src/pxrd_cell_indexing/data/normalization.py（Matrix9Normalizer.denormalize 内联版本）
components = (lattice_norm * std + mean).reshape(-1, 3, 3)
lengths, angles = lattice_lengths_angles(components)  # 基于向量范数/点积，非精确零假设
return torch.cat([lengths, angles], dim=-1)
```

因为 `lattice_lengths_angles` 是从三个基向量的范数和两两点积算长度/夹角，**即使 head 在"应恒零"的 3 个位置输出了噪声，解码依然给出合法（若略有偏差）的长度和角度**，不会崩溃——这正好复现了 RealPXRD 式"无约束但可解码"的鲁棒性。

---

## 3. 三方对比（valid1400，100k 严格 ablation，同数据同 seed）

| 指标 | angles（方案 A 基线） | matrix6（重参数化） | **matrix9（真矩阵回归）** |
|---|---:|---:|---:|
| head 输出维度 | 6 | 6 | **9** |
| **top1_lattice_match_rate** | 39.1% | 40.3% | **40.4%** |
| topk_lattice_match_rate | 99.3% | 99.1% | 99.0% |
| lattice_mae | 5.55 | 5.43 | 5.60 |
| length_mape | 16.0% | 16.0% | 16.1% |
| crystal_system_accuracy（诊断） | 14.2% | 15.4% | 15.2% |
| best epoch | 5 | 11 | 7 |

结果文件：
- angles：[`results/valid1400_scale_100k_no_cs_seed42.json`](../../results/valid1400_scale_100k_no_cs_seed42.json)
- matrix6：[`results/valid1400_scale_100k_no_cs_matrix6_seed42.json`](../../results/valid1400_scale_100k_no_cs_matrix6_seed42.json)
- matrix9：[`results/valid1400_scale_100k_no_cs_matrix9_seed42.json`](../../results/valid1400_scale_100k_no_cs_matrix9_seed42.json)

### 纯 Lattice 候选排名分布（条件于真解在池内）

| 池内排名 | angles | matrix6 | matrix9 |
|---|---:|---:|---:|
| rank 1 | 39.4% | 40.6% | **40.8%** |
| rank 3 | 22.0% | 21.8% | 23.0% |
| rank 4 | 29.7% | 23.6% | 22.8% |

结果文件：[`results/decompose_joint_scale_100k_no_cs_matrix9_seed42.json`](../../results/decompose_joint_scale_100k_no_cs_matrix9_seed42.json)

---

## 4. 解读

1. **matrix9（真矩阵回归）与 matrix6（重参数化）几乎打平**（40.4% vs 40.3%，差距在噪声范围内），两者都比 angles 直接回归好（+1.2~1.3pp）。
2. **额外 3 个自由输出通道没有带来进一步收益**：给 head 3 个"应该学会输出 0"的额外通道，模型确实学到了（否则解码会崩，但实际 lattice_mae/topk 都正常），但没有比 6 分量重参数化更好——说明**提升的主要来源是"目标坐标系的非线性重参数化"本身**（把长度/角度换成 sin/cos 混合的矩阵分量），而不是"给模型更多自由度"。
3. **与 RealPXRD 的关键差异仍然存在**：本实验是监督回归 + SmoothL1，RealPXRD 是 flow matching + MSE 去噪损失，且其 9 维输出被解释为"作用在噪声晶格上的变换"而非"绝对晶格"。这两种范式差异较大，本实验只对齐了"head 输出维度/矩阵原生 loss"这一个变量，不是完整复刻 RealPXRD 训练范式。

---

## 5. 结论与决策建议

| 判定项 | 结论 |
|---|---|
| 真矩阵回归 vs 重参数化 | **基本等价**（40.4% vs 40.3%，无统计显著差异） |
| 是否需要真的 9 维 head | **不需要**——matrix6（6 维 head + 矩阵坐标系归一化）已能拿到几乎全部收益，且实现更简单、无需处理零方差位 |
| 推荐方案 | **matrix6**（简单、等效、无冗余自由度） |
| 是否需要复刻 RealPXRD 的 flow matching 范式 | 超出本次 ablation 范围；如需验证，需要单独设计（迭代去噪 + 残差变换头），成本显著更高，暂不建议 |

**结论**：给模型"真正的"无约束矩阵自由度（9 维）并不比"6 维重参数化"更好——两者性能几乎相同。之前 matrix6 实验的 +1.2pp 提升，其机制是**目标坐标系变换让 loss landscape 更均匀**，与是否"看起来像 RealPXRD 的矩阵回归"关系不大。**推荐继续使用更简单的 matrix6 方案**，无需额外维护 matrix9 的零方差位保护逻辑。

---

## 6. 遗留与后续

- [ ] 500k/全量训练继续用 `configs/scale_100k_no_cs_matrix6.yaml` 系列（不采纳 matrix9）
- [ ] 若未来想真正复刻 RealPXRD 的训练范式（flow matching + 残差变换头），需要单独立项，工作量远超本次 ablation
- [ ] 候选排序优化（D31 遗留）仍是当前最大的 Top-1 提升空间，与表示方式无关
