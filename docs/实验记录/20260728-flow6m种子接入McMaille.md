# flow6m 种子接入 seeded-McMaille（替换 A2）

> ⚠️ **架构勘误**：本文所用种子模型 `full6m_equiv_off` **未使用 RealPXRD 预训练权重**，
> 是自研 PeakTransformer + 自研 flow 从头训练的产物，属未报备的路线偏离。
> PM 指定的主线是方案 A（RealPXRD 预训练迁移）。详见
> [`20260728-路线偏离说明与纠正.md`](../开发日志/20260728-路线偏离说明与纠正.md)。
> 本文的 **McMaille 后处理结论与种子来源无关，仍然有效**。


> 种子：`results/flow_seedgen/full6m_equiv_off/best.pt`（ep16）  
> 池：`results/flow_seedgen/pool_k100_mp100.json`  
> 汇总：`results/reseed_flow6m_ab_mp100.json`、`results/rank_policies_flow6m_k100_mp100.json`

## 口径（L4）

- **L4-loose**：`find_mapping(ltol=0.05, atol=3°)`
- **L4-strict**：loose 且 `|det(scale)−1| < 0.25`
- 真值：primitive standard
- Match rate：样本比例

## NN-only（未进 McMaille）

| | Top-1 | Top-20 | library@100 |
|---|---:|---:|---:|
| L4-strict | 39% | 61% | **67%** |
| L4-loose | — | — | 79% |

（checkpoint 训练时评过 69%；重采样方差约 ±2pp。）

## Seeded-McMaille（McM20 排序后 Top-K / library）

MP100 · **prim · L4-strict**：

| 种子源 | 池大小 | Top-1 | Top-20 | **library** | \|det\| 中位 |
|---|---:|---:|---:|---:|---:|
| **flow6m K20 → Mc** | 70 | **30%** | **55%** | **62%** | 1.0 |
| **flow6m K100 → Mc** | 352 | **31%** | **53%** | **67%** | 1.0 |
| 点回归 6M K20 → Mc | 67 | 16% | 31% | 39% | 1.0 |
| A2 phase5 → Mc | 3864 | 1% | 7% | 10% | 3.0 |

L4-loose library：flow6m K100 → Mc = **92%**（A2 = 100%，但 A2 几乎全是错体积）。

## 排序策略（flow6m K100 池）

prim L4-strict：

| policy | Top-1 | Top-20 | library |
|---|---:|---:|---:|
| **nn_only**（截前 20 采样） | **39%** | **61%** | 61% |
| mcm20 | 31% | 53% | **67%** |
| nn_refined | 23% | 57% | 65% |

McMaille 把 library 从 nn_only@20 的 61% 扩到 67%，但 **McM20 排序压低了 Top-1**（39%→31%）。当前最优部署读法仍是：**种子用 flow 采样；若要 Top-1，优先 NN 顺序；Mc 阶段作扩库/精修。**
