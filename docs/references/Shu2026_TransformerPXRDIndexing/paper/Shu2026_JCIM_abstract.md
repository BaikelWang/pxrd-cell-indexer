# Insights into the Indexing of Powder X-ray Diffraction from a Robust Transformer Deep Learning

- **Cite**: J. Chem. Inf. Model. 2026, XXXX, XXX, XXX–XXX
- **DOI**: https://doi.org/10.1021/acs.jcim.6c01362
- **Published**: July 13, 2026
- **Authors**: Ke Shu, Wei-Xin Yan, Huai-Hai Li, Dong-Yun Gui, Chun-Hai Wang
- **Code/weights**: Zenodo AIdex-R2（见 `../../AIdex_ShuWang2025/code/`）

## Abstract

The indexing of powder X-ray diffraction (PXRD) for ab initio determination of unknown crystal structures remains challenging for systems with low-symmetry (e.g., triclinic and monoclinic), large unit cell (V > 1000 Å³), and nonideal data due to reliance on heuristic methods with limited robustness. Here, we propose **AIdex-R2**, a transformer-based end-to-end framework that performs joint inference of the extinction groups (EGs) and unit cell parameters from sequences of low-angle diffraction reflections. The model achieves ∼98.5% top-5 accuracy for EG identification and ∼1.44% mean absolute percentage error (MAPE) for cell parameter prediction (indexing), with an indexing success rate exceeding ∼90% under extremely realistic perturbations (e.g., zero-shift errors ±0.30°, uncertainty noise ±0.15°, reflection absence n = 4, and impurity reflections n = 2). Large-scale benchmarking shows superior performance over classical algorithms (TREOR, ITO, DICVOL) in both speed/efficiency and accuracy. Through interpretability studies of the internal decision-making mechanism, we confirm that the model adopts a discriminant logic dominated by the first diffraction peak (low 2θ) with dynamic weight allocation, providing a novel perspective and evidence for understanding the “black box” of PXRD indexing.

## Supporting Information（目录提示）

ACS SI 通常免费：`ci6c01362_si_001.pdf`（~15.5 MB）。含超参表、预训练曲线、各晶系 EG 混淆矩阵、额外扰动成功率、外推极限、实验 Cu-Kα 例、attention 权重统计、M20 对比等。本机若 403，请浏览器另存到本目录 `Shu2026_JCIM_SI.pdf`。
