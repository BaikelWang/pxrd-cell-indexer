# Baggett / Ratcliff et al., arXiv:2604.23811 (2026)

- **Title**: Attention Is Not All You Need for Diffraction
- **arXiv**: https://arxiv.org/abs/2604.23811 · PDF 本机：`Attention_Is_Not_All_You_Need_arXiv2604.23811.pdf`（29 页）
- **全文抽取（节选）**：`Attention_Is_Not_All_You_Need_arXiv2604.23811.md`
- **Code**：https://github.com/scattering/paper-ai-diffraction → `../code/paper-ai-diffraction`
- **Zenodo checkpoints**：https://doi.org/10.5281/zenodo.21048093（仓库 README；另有 repro 包记录可查 `reproducibility/`）

## Abstract（要点）

粉末衍射对称性分类：attention 优于 CNN，但**不够**——必须把晶体学知识编进**架构 + 课程**。提出 physics-informed transformer，分到 **99 extinction groups**（粉末可辨的最细对称类）：显式 \(\sin^2\theta\) 通道、物理位置编码、双头 decoder（规则 bit / 直接 EG）+ 融合。三阶段课程（均匀合成预训练 → 真实域细调含择优取向 → 推理时 Bayes prior）+ **温度标定** 是 sim→real 关键。错误在 t-subgroup DAG 上局部、偏向下对称后代。

## 方法骨架

| 模块 | 内容 |
|---|---|
| 目标 | **99 EG**（非 230 SG）；匹配实验证明直接 EG 训练 ≫ SG 再 collapse |
| 输入 | 连续谱强度 + \(\sin^2\theta\) 坐标通道（非 peaks-only） |
| Encoder | Regular Transformer（合成更强）/ ViT（真实更稳）；physics-aware PE |
| Decoder | Split head（37-bit 规则 → lookup EG）+ Aux EG softmax；融合 \(p=\alpha p_{\mathrm{split}}+(1-\alpha)p_{\mathrm{aux}}\) |
| 训练 | Stage1 均匀合成 → Stage2 RRUFF 条件真实噪声/杂质/PO → 推理 log-prior + **T-scaling** |
| 真实 KPI | RRUFF-473 / 325：校准后 Top-1 约 **10–17%** 量级（任务难、标签空间大） |

## 对本项目（Phase A）的 takeaways

1. **Encoder（优先可借鉴）**  
   - 峰位几何：\(\sin^2\theta\) / \(Q^2\) 式「物理尺」比纯 learned PE 更贴衍射；peaks-only 路径可对每峰附带 \(\sin^2\theta\) 或 \(1/d^2\) 特征，而不是只喂强度 token。  
   - Attention 擅长峰间关系；但 CNN 在尖峰合成上会 plateau——与我们 deepen encoder（R11b）方向一致，**容量+几何先验**比盲目堆 head 重要。

2. **Head（对齐「头要简单」）**  
   - 他们用复杂双头是因为任务是 **99 EG 分类**；我们产品锁定 **不做 SG/EG 主 KPI**，勿照搬 37-bit split。  
   - 可吸收的是：**轻量 CS / 规则正则当辅助**，主回归头保持 slim+deep（FiLM），规则细节留给 Phase B 搜索。

3. **训练（优先可借鉴）**  
   - **均匀预训练防先验塌缩** → 再真实域细调；与「训练要满」一致。  
   - **温度标定 / 先验注入是推理期操作**，不改权重就能抬真实 Top-k——Phase B 多假设排序时很有用。  
   - 显式 preferred orientation / 缺峰噪声进入课程，比单纯 peak-consistency λ 更贴近真实失败模式。

4. **任务边界**  
   - 本文是 **对称分类**，不做晶胞回归；AIdex-R2 才是 indexing 端到端标杆。两篇互补：一篇教「对称可辨什么 + 怎么训」，一篇教「峰序列 → 胞参」。

## 建议先读代码

- `code/paper-ai-diffraction/src/paper_ai_diffraction/core/model.py` — ViT / dual-head
- `code/paper-ai-diffraction/src/paper_ai_diffraction/core/rt_model.py` — regular transformer
- `code/paper-ai-diffraction/src/paper_ai_diffraction/utils/extinction_multilabel.py` — flat-37 / EG 映射
- `code/paper-ai-diffraction/docs/TRAINING.md` · `docs/BENCHMARKS.md`
