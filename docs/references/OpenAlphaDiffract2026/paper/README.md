# AlphaDiffract / OpenAlphaDiffract (2026)

- **Paper**: Andrejevic et al., *AlphaDiffract: Automated Crystallographic Analysis of Powder X-ray Diffraction Data*, arXiv:2603.23367
- **PDF（已下载）**: `AlphaDiffract_arXiv2603.23367.pdf`
- **Code**: https://github.com/AdvancedPhotonSource/OpenAlphaDiffract
- **本机代码**: `../code/OpenAlphaDiffract/`
- **核心算法入口**:
  - `src/trainer/model/model.py` — 1D ConvNeXt + CS/SG/lattice 头
  - `src/trainer/train.py` / `run_train_with_manifests.py`
  - `src/simulator/diffraction_generator.py` — 谱仿真
  - `src/ui/app/model_inference.py` — 推理
