# 原生 McMaille 单候选全流程照搬与 A/B

> ⚠️ **架构勘误**：本文所用种子模型 `full6m_equiv_off` **未使用 RealPXRD 预训练权重**，
> 是自研 PeakTransformer + 自研 flow 从头训练的产物，属未报备的路线偏离。
> PM 指定的主线是方案 A（RealPXRD 预训练迁移）。详见
> [`20260728-路线偏离说明与纠正.md`](../开发日志/20260728-路线偏离说明与纠正.md)。
> 本文的 **McMaille 后处理结论与种子来源无关，仍然有效**。


> 二进制：`third_party/McMaille/run_lab/mcmaille_seeded`（`McMaille_seeded.for` 新增 policy 分支）
> 池：`results/flow_seedgen/pool_k100_mp100.json`（ep16，K=100）
> 汇总：`results/flow_seedgen/value_pol_native.json`、`value_pol_multistage.json`、`native_gate_autopsy.json`

## 动机

此前一直用自研的「非破坏多阶段」接法：一颗种子在库里留下 RAW / SUPCEL / LOCAL_MC / CELREF 多行，
最大化 library。但从没有把原生 McMaille 拿到候选之后的那条链路**原样跑过一遍**，
属于没照搬就先改。本次先做忠实复刻，再谈优化。

## 照搬了什么

`.seed` 首行改为 `NSEEDS NSEEDPOL`，`NSEEDPOL=1` 走原生单候选路径：把蒙特卡洛提议换成外部种子，
其余一字不改。

```
局部 MC 精修（只动该晶系自由参数）
        │
        ▼  三重门
  RMAX  < Rmax(0.15)
  RMAX2 < 0.15
  NDAT-LLHKL <= NIND(3)
        │
        ▼  入库 1 行（破坏性）
   Rmax 收紧 → SUPCEL 覆盖该行 → BRAV
        │
        ▼  CELREF2（按晶系 INDIC/AFI）
        │
        ▼  同晶系破坏性去重 → McM20 排序
```

对照源码时发现三处此前没照搬的细节：

1. **门是三重的**，不只是 Rp。还有 `RMAX2` 和「未索引峰数 ≤ NIND」。
2. **CELREF2 的结果不写回库行**，只用来取 DDT/DDQ 算 M20/F20。库行保持 MC 精修后的胞。
3. **去重是破坏性的**（`DO 118`），同晶系体积相近的行会被塌缩。

行为验证：复刻后每样本中位 4 行、mean 5 行；原生 McMaille 自己是中位 2 行；非破坏版 551 行。

## A/B 结果（MP100 · prim · L4-strict，分母 100）

| 流程 | 有库样本 | library@100 | Top-1@100 | Top-1/library |
|---|---:|---:|---:|---:|
| 原生复刻 | 71 | **34.0%** | **24.0%** | **70.6%** |
| 非破坏多阶段 | 100 | 68.0% | 35.0% | 51.5% |

原生版有 **29 个样本被门控清空**，一条候选都不剩，所以要按全 100 归一才公平。

## 关键读数：排序比我们的好，输在门控

原生的 **Top-1/library = 70.6%**，明显高于多阶段的 51.5%——进了库的答案多数能顶到第一。
所以 McM20 那套打分排序应该整套接过来，不该先动它。

门控尸检（`diagnose_native_gate.py`，对进门前的 RAW 种子）：

| 分母 = 池中含答案的 67 个样本 | 数值 |
|---|---:|
| 答案过 Rp 门 | 40 |
| 答案过未索引峰数门 | 43 |
| **答案过完整三重门** | **32** |
| 被门控杀掉的正确答案 | **52%** |
| 这些正确答案的几何误差中位 | **1.06%** |
| 这些正确答案的 Rp 中位 | 0.101 |

配合此前 `rp_vs_accuracy.json` 的标定（0.2% 误差 → Rp 中位 0.22，仅 37% 过门；
1% 误差 → Rp 中位 0.54），结论是：

**门控杀的不是错答案，是精度不够的对答案。Rp<0.15 实际要求约 0.2% 几何精度，
而种子只有约 1%。** 原生自带的局部 MC 有帮助（进门前 44 个样本有候选能过门，
精修后 71 个），但拉不动 1% → 0.2%。

## 结论

- 忠实照搬跑通，指标可复现。
- 原生排序是强项，不要改。
- 单点缺陷是**种子进门前的精度**，这是有证据支撑的结论，不是猜测。
- 下一步：在对称化与原生链路之间插入一个能达到 0.2% 的局部优化器。

## 产物

- `McMaille_seeded.for`：`NSEEDPOL` 开关 + 8300 原生复刻块
- `run_mp100_reseed_nn.py`：`--seed-policy {multistage,native}`
- `scripts/diagnose_native_gate.py`（新增）
