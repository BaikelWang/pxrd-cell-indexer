# Machine Learning Tackles the Challenge of Powder X-ray Diffraction Indexing for All Crystal Systems

- **Authors**: Ke Shu, Dong-Yun Gui, Wei-Xin Yan, Chun-Hai Wang*
- **Journal**: J. Chem. Inf. Model. 2025, 65, 19, 10025–10036
- **DOI**: https://doi.org/10.1021/acs.jcim.5c01506
- **Published**: September 22, 2025
- **Code**: Zenodo https://doi.org/10.5281/zenodo.18798696 (`AIdex-R2.zip`, ~926 MB)

## Abstract

The indexing of powder X-ray diffraction (PXRD) in ab initio unknown structure determinations is a critical yet challenging step in crystallography, particularly for low-symmetry systems (e.g., monoclinic, triclinic) and/or large unit cell systems (V > 1000 Å³). In this work, a machine learning-based indexing method is presented, which achieves high-precision, end-to-end prediction of crystal symmetry and unit cell parameters from powder diffraction peaks for all crystal systems. The trained models (denoted as AIdex) achieve a top-5 accuracy of ∼97% in extinction group (symmetry class) identification, and a mean absolute percentage error (MAPE) <5% for indexing, demonstrating significant improvements in both accuracy and time consumed compared to traditional algorithms (TREOR/ITO/DICVOL). AIdex also shows high capacity for experimental applications, maintaining a success rate of ∼90% even under extreme conditions involving zero-shift error (±0.6°) and uncertainty noise (±0.15°). Applied to practical PXRD data, AIdex gives predicted unit cell parameters close to the experimentally refined ones (MAPE < 5%), serving as ideal initial inputs for further Pawley refinements.

## Access notes

本环境对 ACS 主文 PDF / SI 与 Zenodo 大文件可能返回 403。请用浏览器或机构网下载后放入本目录：

- 主文 PDF → `AIdex_JCIM2025.pdf`
- SI PDF → `AIdex_JCIM2025_SI.pdf`（ACS SI 通常免费）
- 代码包 → 见 `../code/download_aidex.sh`
