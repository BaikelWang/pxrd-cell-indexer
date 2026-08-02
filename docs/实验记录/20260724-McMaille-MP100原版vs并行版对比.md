# 2026-07-24 — McMaille MP100：原版 vs OpenMP 修复版

> **目的**：在 MP100（n=100）上对比 `mcmaille_anchored`（原版单线程）与 `mcmaille_omp_fixed`（方案 B 修复后的并行版），看**精度有没有掉**、**耗时有没有收益**。  
> **脚本**：`third_party/McMaille/run_lab/run_mp100_mcmaille_compare.py`  
> **产物**：`third_party/McMaille/run_lab/mp100_compare/{original,omp_fixed}/*/result.json`、`summary.json`

---

## 口径

| 项 | 设置 |
|---|---|
| 输入 | MP100 CIF → pymatgen 模拟峰（与 `load_mp100_sample` 一致）→ 前 20 峰 |
| λ / zero / NGRID | `1.54184` / `0` / `-3`（黑盒，跳过三斜搜索） |
| 命中 | Top-1 suggested cell vs **conventional-standard** 真值，`Lattice.find_mapping(ltol=0.05, atol=3°)` |
| 原版 | `mcmaille_anchored`，`OMP_NUM_THREADS=1`，20 样本进程并行 |
| 并行版 | `mcmaille_omp_fixed`，每样本 `OMP_NUM_THREADS=16`，6 样本进程并行 |
| 超时 | 原版 3600s / 并行 1200s；**两边均 0 timeout** |

说明：本文 Top-1 mapping@conventional ≈ **52%**，低于文档里历史 McMaille **65.9%** 对标数字——本次重点是**两引擎相对差**，不是复现历史绝对分；绝对分差可能来自峰表/λ/conventional 标签/黑盒参数差异。

---

## 总表

| 指标 | 原版 | 并行版 | 差额 |
|---|---:|---:|---:|
| Top-1 mapping（conventional） | **52%** | **50%** | **−2 pp** |
| Top-20 mapping（conventional） | 53% | **55%** | +2 pp |
| Top-1 mapping（primitive） | 36% | 36% | 0 |
| FoM 损坏（Infinity/NaN 嫌疑） | 1 | 2 | +1 |
| 超时 | 0 | 0 | 0 |
| 单样本墙钟合计 | 24550 s（6.82 h） | **6353 s（1.76 h）** | **3.86× 加速** |
| 单样本墙钟均值 | 245.5 s | **63.5 s** | — |
| 单样本墙钟中位数 | 17.1 s | **2.4 s** | — |

成对翻转（同一批 100 条）：
- 两边都命中：44  
- 两边都未命中：42  
- **仅原版命中：8**  
- **仅并行版命中：6**  
- 净差 = −2，与总表 −2 pp 一致  

→ **整体精度基本持平**；差值落在 MC 随机 + Top-1 摇摆量级，不是「并行版系统性崩坏」。

---

## 分晶系（Top-1 mapping / 墙钟均值）

| 晶系 | n | 原版命中 | 并行命中 | 原版墙钟均值 | 并行墙钟均值 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| cubic | 15 | 6.7% | 13.3% | 0.1 s | 0.3 s | 0.5×（太快，开销主导） |
| hexagonal | 15 | 46.7% | **53.3%** | 61.8 s | 14.1 s | 4.4× |
| tetragonal | 15 | 53.3% | **60.0%** | 77.7 s | 12.9 s | 6.0× |
| trigonal | 15 | 60.0% | 53.3% | 14.1 s | 1.7 s | 8.2× |
| orthorhombic | 20 | 75.0% | 70.0% | 445 s | 163 s | 2.7× |
| **monoclinic** | 15 | **80.0%** | **60.0%** | 405 s | 46.6 s | **8.7×** |
| triclinic | 5 | 0% | 0% | 1454 s | 392 s | 3.7× |

**值得盯的一点**：单斜 Top-1 **−20 pp**（12/15 → 9/15），是总表 −2 pp 的主要来源。  
Top-20 并行版反而略高，说明部分单斜真解仍在候选池里，但 suggested Top-1 排序/MC 采样路径在并行下更易偏到同族非最优胞。

---

## 耗时结论

- **有明显加速，不是损耗**：按单样本墙钟合计，并行版约 **3.9×**；中位数从 17 s → 2.4 s。  
- 单斜/三斜等重样本加速更明显（单斜均值 405→47 s）。  
- 极短样本（立方 ~0.1 s）并行版可能略慢（线程启动/调度开销），不影响总量。  
- `totCPU` 并行版合计更高（约 1.5×），符合「多核摊墙钟、总 CPU 不少做」；墙钟仍大幅下降。

---

## 精度结论（诚实版）

1. **总体：基本不影响**（52% → 50%，净翻转 −2）。  
2. **未再现坏 OpenMP 的 NaN 崩盘**（损坏计数 1 vs 2，可忽略）。  
3. **单斜 Top-1 有实质掉点（−20 pp）**，需要单独跟进（候选池还在，偏 Top-1 选择/随机种子）。  
4. 本次绝对命中率低于历史 65.9% 对标，不宜用本次数字直接改写文档基线；相对对比成立。

---

## 复跑

```bash
cd third_party/McMaille/run_lab
PYTHONPATH=../../../src python3 run_mp100_mcmaille_compare.py --engine original --workers 20
PYTHONPATH=../../../src python3 run_mp100_mcmaille_compare.py --engine omp_fixed --workers 6 --omp-threads 16
PYTHONPATH=../../../src python3 run_mp100_mcmaille_compare.py --summarize
```
