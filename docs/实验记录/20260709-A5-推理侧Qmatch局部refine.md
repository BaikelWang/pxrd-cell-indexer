# 实验 A5 — 推理侧 Q-match 局部晶胞 refine

> **日期**：2026-07-09  
> **状态**：✅ 完成（**未达成功标准**）  
> **总纲**：[`20260709-完胜引擎攻关方案v1.md`](../开发日志/20260709-完胜引擎攻关方案v1.md) §7 A5  
> **单因素**：同一 baseline ckpt / 默认 Top-K；只叠加局部 refine  
> **前置**：A1 评测护栏、A2 扩池无效

---

## 1. 假设

回归点落在「宽松等价类」附近，但不在 `0.05/3°` 内；对池内候选做有限步 Q-match 坐标下降，可把真解拉进严口径。

**成功标准**：严口径 `fom_top1` 相对 A2/A1 基线 **≥+5pp**，且 `topk_elementwise` 不下降。

---

## 2. Code review 要点

| 发现 | 处理 |
|---|---|
| FOM 硬匹配 `q_tol=1e-6` 对优化不可微/梯度几乎为零 | 用 **soft nearest-Q** 损失（`min|ΔQ|` capped） |
| 全池 refine 太慢 | 默认只 refine top-N（10） |
| 体积漂移风险 | seed 相对 `|log V|≤log(2)` 硬护栏 + 目标函数惩罚 |
| 仅当 loss 改善才替换候选 | 避免无意义扰动 |

**实现**：`src/pxrd_cell_indexing/model/refine.py`  
**接入**：`eval_mp100.py` / `eval_valid.py` 的 `--refine-steps` 等  
**测试**：`tests/test_refine.py`（扰动 cubic 可拉回；20 相关测试全绿）

---

## 3. 实验矩阵（MP100，`ltol=0.05` / `atol=3°`）

Checkpoint：`scale_100k_no_cs_matrix6_seed42/best.pt`

| ID | 池 | refine |
|---|---|---|
| A1/A2a 基线 | default K20 | 关 |
| **A5a** | default K20 | steps=40, top_n=10, len±25%, ang±15° |
| **A5b** | default K20 | steps=80, top_n=20, len±40%, ang±25° |
| **A5c** | extended + pool log(2) | steps=40, top_n=10 |

```bash
python scripts/eval_mp100.py ... --ltol 0.05 --atol-deg 3 \
  --refine-steps 40 --refine-top-n 10 \
  --output-path results/beat_engine/a5a_mp100_strict.json
```

---

## 4. 结果（MP100）

| ID | fom mapping | **fom elementwise** | topk mapping | **topk elementwise** | Δ fom vs A1 |
|---|---:|---:|---:|---:|---:|
| A1 基线 | 12% | 0% | 26% | **8%** | — |
| A2a | 13% | 0% | 24% | 7% | +1pp |
| **A5a** | 12% | 1% | 28% | **7%** | **0pp** |
| **A5b** | 13% | 1% | 30% | **7%** | **+1pp** |
| **A5c** | 4% | 1% | 15% | **8%** | **−8pp** |

单元测试中：对真解附近扰动的 cubic，refine **能**降低 soft Q loss 并拉近边长 → 算法本身工作；端到端无效说明 **seed 离真解太远，不在局部盆地内**。

---

## 5. 结论

1. **假设否定（端到端）**：局部 refine **不能**把严口径 fom 抬 +5pp；最佳仅 **+1pp**（A5b）。  
2. **真池不涨**：`topk_elementwise` 仍 **7–8%**，与 A2 一致。  
3. **根因**：median 角误差 ~18°、边长相对误差 ~32%，远超 refine 的合理搜索半径（±15–25° / ±25–40% 仍不够，再放大会过拟合假解）。  
4. **A5c 变差**：在已过滤的 extended 池上 refine，FOM 更易选错 → 推理侧微调无法弥补几何缺口。  
5. **段 1 推理侧上限**：A1 尺子 + A2 扩池 + A5 refine 均已证伪「不重训就能大幅抬严口径」。下一步必须动 **表示/编码（A3）** 或 **低对称假设（A4）**，再考虑训练对齐（段 2）。

---

## 6. 下一步

| 优先级 | 实验 | 理由 |
|---|---|---|
| **高** | **A3** Encoder 2θ 连续化 / 更细分箱 | 修峰位截断；短训验证趋势 |
| 中 | **A4** 低对称 Bravais | 补 mono/tric 等假设 |
| 低 | A5 更强全局搜索（差分进化等） | 成本高，且 A5 已暗示局部盆地不对 |

**Gate-1 状态**：`topk_elementwise` 仍 ~8%，距 40% 很远；**禁止**因此启动全量训练。
