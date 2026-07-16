# 2026-07-14 — 100k R10-slim（Niggli + FiLM）vs 旧冠军

## 设置

| 项 | 旧冠军 `r3_cubic_split_clf` | 本次 `r10_slim_film_niggli` |
|---|---|---|
| 数据 | train100k / valid1400 | 同 split |
| 标签 | primitive | **Niggli** |
| 头 | 7 CS + cubic setting | **FiLM 共享加深**（无 setting） |
| Encoder | histogram 浅 MLP | 同结构（未加深） |
| 优化 | bs64 / lr1e-3（历史） | bs256 / lr2e-3（R7 公平配方） |
| 预算 | best @ep13 | max 40ep，best @ep32 |

产物：`results/experiments/scale_100k_r10_slim_film_niggli_seed42/`  
对比 JSON：`results/beat_engine/r10/compare_vs_champion.json`

> 口径说明：标签从 primitive→Niggli 后，**同一物理胞的真值六参数不同**；主比 **strict elementwise** 与 **angle/length MAE**（物理量）。normalized loss 不可直接跨口径比。

## valid1400 主指标（各自 best-elem checkpoint）

| 指标 | 旧冠军 | R7 bs256（primitive 对照） | **R10-slim Niggli** | vs 冠军 |
|---|---:|---:|---:|---:|
| **strict elem** | 15.43% | 15.36% | **21.29%** | **+5.86 pp** |
| strict mapping | 15.93% | — | — | — |
| angle MAE | 8.71° | 8.68° | **7.80°** | **−0.91°** |
| length MAE | 0.897 Å | 0.866 Å | **0.786 Å** | **−0.11 Å** |
| CS acc | 22.5% | — | 23.6% | +1.1 pp |
| best epoch | 13 | 20 | 32 | — |

## 结论

1. **有明显进步**：strict elem 15.4% → **21.3%**；角/长误差同步下降。  
2. 进步来自组合：**Niggli 标签 + FiLM 简化加深头 + bs256/sqrt-LR 配方**（本 run 未加深 encoder）。  
3. 相对「仅换大 batch」的 R7（仍 ~15.4%），本次抬升不是单纯优化器效应。  
4. 下一步按计划：在此 `R10-slim` 上做 **R11b Encoder 加深**，再抠训练策略/放量。

## 配置

- `configs/scale_100k_r10_slim_film_niggli.yaml`
- 数据：`train100k_niggli_seed42.jsonl` / `valid1400_niggli_seed42.jsonl`
- stats：`lattice_matrix6_stats_100k_niggli_seed42.json`
