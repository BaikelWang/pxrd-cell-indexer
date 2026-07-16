# R5-Diag：严口径 Top-K / FOM 现状（冠军 `cubic_split_clf`）

> 2026-07-13 · valid1400 · ltol=0.05 / atol=3°  
> Checkpoint: `scale_100k_r3_cubic_split_clf_seed42`

## 主结果

| 口径 | raw Top-1 | Top-K 池召回 | FOM Top-1 |
|---|---:|---:|---:|
| **elementwise（Gate）** | **15.43%** | **16.50%** | **0.00%** |
| pymatgen mapping | 15.93% | **25.36%** | 18.64% |
| volume-guarded | 15.86% | 17.14% | 0.21% |

分晶系（elementwise）：

| CS | raw | pool |
|---|---:|---:|
| cubic | 89.0% | 90.5% |
| tet | 1.5% | 4.0% |
| orth | 1.5% | 1.5% |
| hex | 7.5% | 8.0% |
| trig | 6.5% | 8.5% |
| mono | 0.0% | 0.0% |
| tric | 2.0% | 3.0% |
| **非立方均值** | **3.17%** | **4.17%** |

## 裁决

1. **单锚点 Bravais 扩展几乎抬不动 elementwise**（+1.1pp overall，非立方 +1pp）→ 池召回接近 raw，**raw 回归误差仍是主因**。
2. **FOM 在 elementwise 口径下失效（0%）**：探针显示 FOM 偏好半胞/超胞尺度变体（如 a=2.65 vs 真值 5.29），mapping 命中但 elementwise 失败。宽口径 M2.5 的 85.9% 不可直接搬到严口径 elementwise Gate。
3. **进入 R5-A**：用 `lattice_norm_all` 多锚点扩池；同时对 elementwise 评测关闭尺度变体 / 加体积护栏，避免 FOM 再选半胞。

产物：
- `scripts/diag_strict_topk_fom.py`
- `results/beat_engine/raw_diag/r5/strict_topk_fom_clf.json`
- `results/beat_engine/raw_diag/r5/eval_valid_strict_clf.json`
