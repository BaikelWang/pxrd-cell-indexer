# B1-S2 续：ortho/mono pass-1 调度 + offdiag 多样性（2026-07-23）

> 承接 P0.2（kwargs 无效）/ P0.3（种子走不到）。本轮改生产 `qsearch` 顺序求解路径。  
> 不改 NN；6M 训练仍等中低对称搜索配方更稳。

---

## 1. 代码改动（`src/.../search/qsearch.py`）

| 项 | 内容 |
|----|------|
| **pass-1 对角种子** | ortho/mono：低 q 峰三重组合 × 小 Miller(≤4)，按 `(max_peak, sum)` 排序；轴按 Gii 降序标记 |
| **pass-2** | 原 dedupe product；无 perfect 且有剩余时间则继续（不再被 pass-1 耗尽预算） |
| **pass-1 时间盒** | 非 triclinic：约 55% 预算后强制进入 pass-2 |
| **offdiag 多样性** | `per_off` 半配额留给 miller_sum≥3，避免 (011) 挤掉 (022) |
| **mono 轴向窗** | 12 → 16 |
| **opts 排序** | ortho/mono pass-2：peak-first（非 miller-first） |
| **预算** | ortho 15s；mono 60s（≈20s/unique-axis） |

试过又撤回：逐种子 approx 排序 offdiag（预算被吃光，mono 最好子集 20%/空池 80%）。

---

## 2. 最好子集（consistent ∧ axial3）

| 轮次 | overall@20 | empty | ortho@20 | mono@20 |
|------|----------:|------:|---------:|--------:|
| P0.2 近生产 | 43.3% | 46.7% | ~60% | ~27% |
| pass-1 初版 | **57.5%** | 37.5% | **85%** | 30% |
| +时间盒/窗16 | **60%** | 33% | 80% | **40%** |

深挖空池 mono 样（3.684, 5.010, 5.029, β≈104°）：由空池 → Top-20 hit（`monoclinic_a` ~9s）。

---

## 3. B1-S2 全协议复测（n=40/系）

产物：`results/beat_engine/b1_search/b1_s2_valid1400_ortho_mono_pass1.json`

| 子集 | q@20 | nn@20 | merged@20 | merged@100 | Δ(q−nn)@20 |
|------|-----:|------:|----------:|-----------:|-----------:|
| **overall** | **18.8%** | 18.8% | 18.8% | **31.2%** | **0pp**（旧 −3.8pp） |
| orthorhombic | 22.5% | 25.0% | 25.0% | 35.0% | −2.5pp（旧 −7.5） |
| monoclinic | **15.0%** | 12.5% | 12.5% | 27.5% | **+2.5pp**（旧 0） |

相对旧 B1-S2（q@20=15%）：**+3.8pp**；与 NN 打平，**仍未过**「相对 NN +8pp / Top-20≥30%」Gate。  
瓶颈仍在全协议空池 + label≠geom（~60%），不是 B2 排序。

---

## 4. 与 6M 的关系

- 搜索主路径已比旧基线明显更好，但 **配方未冻结**（mono 最好子集空池仍高，全协议未过 Gate）。  
- 若优先看规模效应：可先开一版 A3-G1×6M（搜索侧用当前顺序求解），与 100k 对照；不期待单靠 6M 消掉空池。
