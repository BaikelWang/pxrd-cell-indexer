# 外部论文与代码参考（本地镜像）

目录：`docs/references/`。供 Encoder / 输出头 / 训练策略对照学习，**不进入训练主依赖**。

| 项目 | 论文 | 代码 | 本机状态 |
|---|---|---|---|
| **OpenAlphaDiffract 2026** | arXiv:2603.23367 PDF ✅ | GitHub clone ✅ | `OpenAlphaDiffract2026/` |
| **Chitturi 2021 / DeepLPnet** | PMC 全文 md ✅（IUCr PDF 本机 403） | GitHub clone ✅（含 models） | `Chitturi2021_DeepLPnet/` |
| **AIdex Shu/Wang 2025** | 摘要 md ✅（ACS 主文需订阅） | Zenodo 包 ✅（Windows/ONNX，非训练源码） | `AIdex_ShuWang2025/` |
| **Shu 2026 AIdex-R2 Transformer** | 摘要 md ✅；DOI `acs.jcim.6c01362`；SI 本机 403 | 软链 → AIdex-R2 ONNX | `Shu2026_TransformerPXRDIndexing/` |
| **Attention Is Not All You Need 2026** | arXiv:2604.23811 PDF+md ✅ | GitHub clone ✅（checkpoint 在 Zenodo） | `AttentionNotAllYouNeed_Diffraction2026/` |

## 建议优先阅读的代码文件

### OpenAlphaDiffract
- `code/OpenAlphaDiffract/src/trainer/model/model.py`
- `code/OpenAlphaDiffract/src/simulator/diffraction_generator.py`
- `code/OpenAlphaDiffract/src/trainer/train.py`

### DeepLPnet
- `code/DeepLPnet/src/model.py` — 1D-CNN 回归
- `code/DeepLPnet/src/data_generation.py` — 增强/非理想性
- `code/DeepLPnet/src/helper_functions.py` / `helper_functions_topas.py`

### AIdex / AIdex-R2（Shu 2025 + 2026）
- `AIdex_ShuWang2025/code/AIdex-R2-core/model_r2/*.onnx` — EG 分类 + 分 EG lattice 回归（Netron）
- `Shu2026_TransformerPXRDIndexing/paper/` — R2 可解释性 / 扰动结论摘要
- 注意：无开源训练 `.py`，只有推理二进制与 ONNX

### Attention Is Not All You Need（对称 / 99 EG）
- `AttentionNotAllYouNeed_Diffraction2026/code/paper-ai-diffraction/src/paper_ai_diffraction/core/model.py`
- `.../core/rt_model.py` · `.../core/inference.py`（融合 / T-scaling）
- `.../utils/extinction_multilabel.py` · `docs/TRAINING.md`

## 与 Phase A 三支柱的快速对照

| 支柱 | Shu 2026 (AIdex-R2) | Attention…Diffraction |
|---|---|---|
| 特征要强 | 低角峰序列 Transformer；首峰主导 attention | \(\sin^2\theta\) 通道 + physics PE；峰间关系 |
| 头要简单且深 | EG→分 EG 专家回归（偏复杂；我们取条件化思想即可） | 双头规则/EG（对称任务专用；勿照搬到 lattice KPI） |
| 训练要满 | 强扰动增强；对标 TREOR 等 | 均匀预训练→真实细调→推理标定；PO/杂质课程 |

## 补下命令

```bash
# AIdex / AIdex-R2（~926MB，遇 403 隔一段时间再试）
bash docs/references/AIdex_ShuWang2025/code/download_aidex.sh

# Attention 仓库更新
cd docs/references/AttentionNotAllYouNeed_Diffraction2026/code/paper-ai-diffraction && git pull --ff-only

# 浏览器另存 PDF（本环境常 403）
# Chitturi: https://journals.iucr.org/j/issues/2021/06/00/vb5020/vb5020.pdf
# AIdex 2025: https://doi.org/10.1021/acs.jcim.5c01506
# Shu 2026:   https://doi.org/10.1021/acs.jcim.6c01362
# Shu 2026 SI: .../suppl_file/ci6c01362_si_001.pdf
```
