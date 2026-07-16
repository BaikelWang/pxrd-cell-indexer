# Shu et al., JCIM 2026 (AIdex-R2 / Transformer indexing)

- **Title**: Insights into the Indexing of Powder X-ray Diffraction from a Robust Transformer Deep Learning
- **Authors**: Ke Shu, Wei-Xin Yan, Huai-Hai Li, Dong-Yun Gui, Chun-Hai Wang\*
- **DOI**: https://doi.org/10.1021/acs.jcim.6c01362
- **Published**: 2026-07-13（JCIM ASAP / XXXX）
- **前作**：AIdex (JCIM 2025) → [`../../AIdex_ShuWang2025/`](../../AIdex_ShuWang2025/)
- **代码/权重**：同 AIdex-R2 Zenodo（Windows + ONNX，无训练源码）→ `../code/AIdex-R2-core`（软链）
- **主文 PDF**：ACS 需订阅；本环境直连 SI 也 403。浏览器可另存：
  - 主文：https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.6c01362
  - SI（通常免费）：https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6c01362/suppl_file/ci6c01362_si_001.pdf → 建议存为 `Shu2026_JCIM_SI.pdf`

## Abstract（摘录）

针对低对称 / 大晶胞 / 非理想 PXRD，提出 **AIdex-R2**：Transformer 端到端从**低角峰序列**联合推断 extinction group (EG) 与晶胞参数。报告：EG Top-5 ~**98.5%**；晶胞 MAPE ~**1.44%**；在零漂 ±0.30°、噪声 ±0.15°、缺峰 n=4、杂质峰 n=2 等扰动下 indexing 成功率仍 >~90%。大规模对比优于 TREOR / ITO / DICVOL。可解释性分析称决策由**第一衍射峰（低 2θ）主导**并配合动态权重分配。

## 与 2025 AIdex 的关系

| | AIdex 2025 (`5c01506`) | AIdex-R2 2026（本文） |
|---|---|---|
| 任务 | EG + lattice 端到端 indexing | 同任务，强调 **Transformer + 可解释性** |
| 指标（论文口径） | EG Top-5 ~97%；MAPE <5% | EG Top-5 ~98.5%；MAPE ~1.44% |
| 扰动 | 零漂 ±0.6° / 噪声 ±0.15° | 零漂 ±0.30° + 缺峰/杂质等更细压力测试 |
| 发布物 | Zenodo `AIdex-R2.zip`（ONNX） | 同一 R2 软件包；本文补 attention / 决策机制分析 |

本仓库已下载的 ONNX 即 R2：`Transformer_EG_classifier.onnx` + 分 EG 的 `Finetuned_transformer_EG_*_suffix.onnx`。

## 对本项目（Phase A）的 takeaways

1. **Encoder**：峰序列 Transformer + 低角峰权重高 → 支持我们「特征要强」、保留精确峰位；勿把子度级信息过早 bin 掉。
2. **Head**：分层 **EG 分类 → 条件 lattice 回归**（多 ONNX 专家头）→ 与我们「头要简单」有张力：产品锁定不做 SG/EG 主 KPI，但 **CS 正确子集上的 lattice** 可借鉴其「先对称再参数」的条件化思路（FiLM/shared 比 100+ 专家更贴合我们）。
3. **训练**：抗零漂/缺峰/杂质的增强是其成功率来源；与我们 peak consistency 试验可对照，但不宜为对标 MAPE 口径而泄漏化学式。
4. **评估口径**：论文 MAPE / Top-5 EG ≠ 我们 MP100 elementwise；只作方法参考，不作数字对赌。
