# 2026-07-14 — R9 规范胞标签与小 benchmark 对照

## 目标

统一训练/评测到 Niggli 规范胞，并在 **3500/700 无重叠小 benchmark** 上做 primitive vs niggli 公平 A/B（同 H0 头、同预算）。

## 已完成

| 项 | 结果 |
|---|---|
| physical length/angle loss 分量 | `loss_length_phys` / `loss_angle_phys` / `loss_matrix6` |
| 双 checkpoint | `best.pt`（elementwise）+ `best_valid_loss.pt` |
| canonical 模块 | `data/canonical.py`：`primitive` / `reduced` / `niggli` |
| 审计（140 样本） | Niggli 幂等 **100%**；Niggli==reduced **100%**；峰 Chamfer 三口径一致 → **采用 niggli** |
| 小 benchmark | `benchmark_train3500_seed42.jsonl` + `benchmark_valid700_seed42.jsonl`（0 key overlap） |
| Niggli 重标 | overfit700 / train3500 / valid700 + `lattice_matrix6_stats_benchmark3500_niggli_seed42.json` |
| MP100 真值 | `mp100.py` / `eval_mp100.py --convention`（默认 primitive 保旧数；R9+ 用 niggli） |

审计产物：`results/beat_engine/r9/canonical_audit.json`

## P0 结果（full-batch 1200ep，正确协议）

| 口径 | best elem | @ep | final elem | ang MAE | Gate |
|---|---:|---:|---:|---:|:---:|
| primitive | **99.7%** | 800 | 99.7% | 0.16° | **PASS** |
| niggli | **99.7%** | 750 | 98.7% | 0.42° | **PASS** |

产物：`results/beat_engine/r9/p0_700_fullbatch.json`

## 3500 A/B 结果（R9 Gate，同 H0）

| 口径 | best elem | @ep | ang MAE | length MAE |
|---|---:|---:|---:|---:|
| primitive | 6.57% | 73 | 12.90° | 1.364 Å |
| niggli | 6.29% | 51 | **12.59°** | **1.346 Å** |

裁决：**接受 Niggli**（物理误差略优，elem −0.28pp 在噪声内）。小样上绝对值都偏低，只作口径 Gate，不作能力上限。

## 计划顺序（2026-07-14 再订）

```
A-算法：R9 → R10 → R11b（先拉满初值 / 压低有效 loss）
A-训练：R11 → 扩数据 / 全量
Phase B：多假设 / 搜索 …
```

3500/Niggli 上 H0 vs H1 仅为探路；正式裁决仍在 100k。

## R10 探路结果（3500/Niggli，best elem checkpoint）

| 臂 | 结构 | best elem | @ep | ang | length |
|---|---|---:|---:|---:|---:|
| H0 | 7 CS + setting | **6.29%** | 51 | 12.59° | 1.346 Å |
| H1 | 7 CS，无 setting | 4.57% | 37 | 13.13° | 1.316 Å |
| H2 | 共享加深（4×512） | 4.43% | 71 | 13.66° | 1.313 Å |
| H3 | FiLM 加深（4×512） | 4.71% | 61 | 13.25° | 1.323 Å |

**裁决（2026-07-14）**：采纳简化头。小样上 H0 略高，但回退可接受；在 H1/H2/H3 中取 **H3 FiLM** 为默认 `R10-slim`（无 setting；共享回归权重 + CS FiLM + 轻量 CS CE）。后续 100k / Encoder 加深均架在此头上，**不再加回 setting**。

## Configs

- `configs/overfit700_r9_primitive_h0.yaml` / `overfit700_r9_niggli_h0.yaml`
- `configs/benchmark3500_r9_primitive_h0.yaml` / `benchmark3500_r9_niggli_h0.yaml`
- `configs/benchmark3500_r10_niggli_h1.yaml`
- `scripts/r9_p0_700_overfit.py`
