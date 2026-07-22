# 2026-07-15 — Cell Indexing 可执行优化方案 v3

> **目标**：在严格 peaks-only 契约下，把当前单点网络做成可靠的 PXRD cell-indexing 初值器，并在此基础上增加独立峰搜索与候选排序，逐步提高最终 Top-1 晶胞命中率。  
> **主指标**：strict elementwise，长度 `ltol=0.05`、角度 `atol=3°`。  
> **输入边界**：推理只允许 `(2θ, I)` 峰表与波长 `λ`；禁止 formula、atom types、atom count、真 CS、真 SG 进入模型或搜索。  
> **标签边界**：训练、验证、MP100 全链路使用同一 Niggli-reduced 约定；禁止预测后直接做 naïve Niggli reduction。  
> **执行纪律**：先补评测护栏，再逐个验证表示、输出坐标、训练课程，最后扩数据和做全局搜索；每一阶段只改变一个核心因素。

本文是 v2 的执行版修订。v2 中已经完成或被实验否决的方向不再重复展开，后续以本文的阶段编号、实验矩阵和 Gate 为准。

---

## 1. 当前真实起点

### 1.1 当前生产基线

`scale_100k_r10_slim_film_niggli_seed42`

```text
PXRD peaks (2θ, I) + λ
  → I > 5 过滤
  → 256-bin inverse-d² intensity histogram
     + 24 个排序后的低角 inverse-d²
     + peak_count / 50
  → 281-d 固定向量
  → shallow histogram MLP
  → 512-d embedding
  → 轻量 7-CS classifier
  → oracle-CS 训练 / predicted-CS 评测的 FiLM lattice head
  → normalized matrix6
  → Niggli lattice (a,b,c,α,β,γ)
```

固定配置：

- 100k Niggli train + valid1400 Niggli；
- batch 256；
- encoder/head LR 均为 `2e-3`；
- warmup 2 epoch，最多 40 epoch；
- SmoothL1(matrix6) + `0.1 × CE(CS)`；
- augmentation 关闭；
- checkpoint 按 strict raw Top-1 elementwise 选择。

### 1.2 当前已确认数字

| 指标 | 当前值 |
|---|---:|
| valid1400 strict raw elementwise | **21.29%** |
| angle MAE | **7.80°** |
| length MAE | **0.786 Å** |
| classifier CS accuracy | **80.4%** |
| oracle-CS elementwise | **约 21.79%** |
| predicted-CS elementwise | **21.29%** |
| CS 正确子集 lattice elementwise | **约 26.0%** |

CS 路由只造成约 0.5pp gap，因此当前主瓶颈不是分类器，而是峰几何到低对称晶胞的映射。

按晶系看，模型主要依赖高对称容易样本：

| 晶系 | strict elementwise |
|---|---:|
| cubic | 86.0% |
| tetragonal | 8.5% |
| orthorhombic | 3.0% |
| hexagonal | 24.0% |
| trigonal | 25.5% |
| monoclinic | 1.0% |
| triclinic | 1.0% |

按峰数看，峰越多、组合越复杂，当前 bag 表示越难解析：

| 峰数 | strict elementwise |
|---|---:|
| ≤10 | 54.2% |
| 11–20 | 20.0% |
| 21–40 | 8.9% |
| >40 | 2.3% |

### 1.3 已完成的 Encoder 消融

在 3500/Niggli + 同一 FiLM 头上：

| Encoder | strict elementwise | 裁决 |
|---|---:|---|
| shallow histogram | 4.71% | 基线 |
| 4×1024 ResMLP | 6.86% | 有效 |
| 8×1024 ResMLP | 6.43% | 有效但不优 |
| **8×2048 ResMLP (E1c)** | **7.14%** | 当前候选 |
| bins512 + peaks48 | 5.71% | 不采纳 |
| E1c + spectrum CNN | 7.00–7.14% | 不优于 E1c，关闭 |
| spectrum-only | 1.43% | 否决 |

E1c 证明增加容量可能有用，但不能证明继续扩到 200M–600M Dense 模型有效。当前输入只有 281 维，过度堆参数无法恢复已经丢失的峰间结构。

---

## 2. 核心诊断与优化假设

### H1：当前主要是输入表示瓶颈

Histogram 保留了总体强度分布，但弱化或丢失：

- 峰序关系；
- 相邻 `1/d²` 间隔；
- 多峰之间的组合约束；
- 同一 bin 内多个峰的信息；
- 对高峰数复杂谱的显式交互。

AIdex-R2 的本地 ONNX 显示，其 EG classifier 直接接收 20 个低角标量，经约 256 维的小型 Transformer 建模；这说明应优先验证“低角峰序列 + 峰间 attention”，而不是继续扩大 histogram Dense tower。

### H2：输出坐标应与衍射方程更直接对齐

衍射峰位置满足：

\[
\frac{1}{d_{hkl}^2}=\mathbf h^\mathsf T G^* \mathbf h
\]

其中 \(G^*\) 是 reciprocal metric。当前 matrix6 回归的是实空间基矩阵自由分量，网络必须间接学会矩阵求逆和二次型关系。将输出改为正定 reciprocal metric，有可能降低低对称和角度回归难度。

### H3：增强应采用 clean → perturbed 课程

当前训练完全关闭增强。直接从 epoch 0 对所有样本强增强可能破坏 ideal PXRD 精度。更稳妥的方案是：

1. 先在 clean ideal peaks 上学会晶胞映射；
2. 再用少量 zero-shift、峰位 jitter、缺峰、杂质峰和强度畸变微调；
3. 同时保留 clean 样本，避免遗忘。

### H4：单点网络不会独自解决全部 indexing 歧义

monoclinic / triclinic 近零命中和旧局部 refine 失败说明，低对称问题具有多解与错误盆地。网络应先成为可靠的体积、晶系和 metric 初值器；要逼近传统 indexing 工具，最终仍需要从观测峰独立构造候选的全局搜索。

---

## 3. 全阶段统一实验协议

### 3.1 数据分层

| 层级 | 用途 | 是否用于裁决 |
|---|---|---|
| P0-700 | round-trip、梯度、过拟合探针 | 只裁决实现正确性 |
| 3500/700 | 快速排除明显失败架构 | 不外推最终收益 |
| 100k/valid1400 | 主算法消融、选模 | **主要裁决集** |
| 500k → 1M → 6M | 放大已通过算法 | 只改数据规模 |
| MP100 | 阶段里程碑验收 | 禁止连续调参 |

### 3.2 固定报告指标

每个 100k 以上实验必须同时输出：

1. `strict_raw_top1_elementwise_rate`；
2. classifier `crystal_system_accuracy`；
3. oracle-CS 与 predicted-CS strict elementwise；
4. `cs_correct_subset_lattice_elementwise`；
5. per-CS strict elementwise；
6. 按峰数 `≤10 / 11–20 / 21–40 / >40` 分层；
7. angle MAE、length MAE、length MAPE；
8. 六个晶胞参数分别的达标率；
9. 非法/非正定晶胞比例；
10. 参数量、峰值显存、samples/s、best epoch、总墙钟；
11. raw prediction 的逐样本 JSON，供同一 evaluator 复算。

Top-K、mapping、FOM 只作诊断，不得替代 raw strict elementwise。

### 3.3 可比性纪律

- 每轮只改变一个核心因素；
- 探索阶段 seed=42；
- 通过 Gate 的结构再做 seeds 42/43/44；
- 同轮使用相同 train/valid keys、Niggli 标签、stats、batch、epoch 预算和 checkpoint Gate；
- 新输出表示必须重新计算对应 train-only stats；
- 100k Gate 未通过，不上 500k/1M；
- MP100 每个阶段最多评一次通过 valid Gate 的候选。

### 3.4 通用进步 Gate

单项实验只有满足以下任一主条件，且无严重副作用，才算通过：

- overall strict elementwise 相对同预算对照 **≥ +2pp**；或
- non-cubic strict elementwise **≥ +2pp**；或
- `peak_count > 40` 子集 **≥ +3pp**；或
- angle MAE **≥ -0.5°** 且 strict elementwise 不下降。

共同约束：

- cubic 相对对照回退不超过 1pp；
- CS-correct lattice 不下降超过 1pp；
- 不靠 loose mapping 或 oracle 信息涨分；
- 三 seed 复核后提升方向一致。

---

## 4. 阶段 A0：评测与协议护栏

### 要做什么

1. 将 classifier 自身输出的 CS accuracy 与“由预测晶胞反推的 CS”彻底分开；
2. 增加 CS-correct subset、oracle/predicted route gap；
3. 增加 per-CS、峰数分层、六参数独立达标率；
4. checkpoint 中保存 `representation`、`canonical convention`、stats 路径；
5. `eval_valid.py`、`eval_mp100.py` 自动继承 checkpoint 的 Niggli/matrix6 口径；
6. 对 R10-slim 重跑一次 valid1400 raw，并只对通过协议检查的 checkpoint 跑一次 MP100 raw。

### 原理

如果分类 accuracy、晶胞反推 CS、宽松 mapping 和 strict elementwise 混在一起，后续无法判断提升来自哪里。A0 不提高模型分数，但它决定所有后续结论是否可信。

### 实验设计

- checkpoint：R10-slim Niggli seed42；
- rerank：`none`；
- candidate：仅 raw Top-1；
- valid1400 复算两次，确认同一 checkpoint 指标完全一致；
- 用同一逐样本 JSON 离线复算指标，确认脚本间无偏差。

### 预期与 Gate

- R10 strict raw 应复现到 21.29% 附近；
- classifier CS accuracy 应复现到约 80.4%；
- oracle/predicted gap 应在约 1pp 内；
- 所有后续 run 产出统一 metrics schema。

若复现偏差 >0.5pp，暂停训练，先定位标签、stats、路由或 evaluator 差异。

### 交付物

- 统一 metrics JSON schema；
- R10 valid1400 canonical baseline JSON；
- R10 MP100 raw milestone JSON；
- 评测协议说明和可复现命令。

---

## 5. 阶段 A1：E1c × 100k 单点集成

### 要做什么

只把 R10-slim 的 shallow histogram encoder 替换为：

```yaml
histogram_hidden_dim: 2048
histogram_num_blocks: 8
histogram_dropout: 0.15
```

FiLM head、Niggli matrix6、数据、loss、CS 路由全部保持不变。

### 原理

E1c 在 3500 上比 shallow encoder 高 2.43pp，但小数据结果可能来自容量和训练时长差异。必须在 100k 同口径上验证，才能决定 Dense histogram 容量是否仍值得保留。

### 实验设计

#### A1-M0：基线复现

- 直接复用 R10-slim 配置；
- 40 epoch；
- seed42；
- 目标：确认 21.29% 可复现。

#### A1-M1：E1c 100k

- 8×2048 encoder；
- head 仍为 FiLM 4×512；
- LR 从 `1e-3` 起；
- warmup 3 epoch；
- max 80–100 epoch；
- min 40 epoch，patience ≥20；
- 不加增强、不改 loss。

#### A1-M2：仅在 M1 最佳点仍持续上升时长训

- 从 M1 最佳 recipe 延长，不改结构；
- 不单独把“训练更久”算作架构收益。

### 预期

- 合理区间：相对 R10 **0–3pp**；
- 若收益存在，应优先出现在复杂峰或 non-cubic，而不是只涨 cubic；
- 预计训练成本显著高于 R10，因此需同时报告收益/墙钟。

### 裁决

- 通过通用 Gate：保留 E1c，作为 Peak Transformer 的 histogram 对照/融合分支；
- 未通过：停止 U1/U2/U3 和更大 Dense tower，默认回到 shallow histogram；
- 即使通过，也不直接继续扩到 12×3072/16×3072，先进入 A2。

---

## 6. 阶段 A2：AIdex 式 Peak Geometry Transformer

这是 Phase A 的最高优先级算法实验。

### 6.1 要做什么

新增独立 encoder，不复用旧 discrete-2θ Bert：

```text
按 2θ 从低到高排序的前 N 个峰
  → 每峰物理 token
  → CLS + rank embedding
  → 4–6 层 pre-LN Transformer
  → CLS / attention pooling
  → 256-d peak-sequence embedding
```

候选 token：

\[
x_i=[
\tilde g_i,\;
\Delta\tilde g_i,\;
\tilde I_i,\;
i/N
]
\]

其中：

- \(g_i=1/d_i^2\)；
- \(\tilde g_i=g_i/g_{\max}\)；
- \(\Delta\tilde g_i=(g_i-g_{i-1})/g_{\max}\)；
- \(\tilde I_i\) 为每谱归一化强度，可选 sqrt/log；
- padding 使用显式 mask。

初始规模：

- `d_model=256`；
- 8 heads；
- 4 layers；
- FFN 1024；
- dropout 0.1；
- learned CLS + rank embedding；
- 参数量目标约 5M–20M，而不是百 M。

### 6.2 原理

低角峰决定大尺度晶胞几何，峰间距包含 reciprocal lattice 的组合结构。Attention 能直接比较任意峰对；histogram bag 做不到这一点。

AIdex-R2 已验证“20 个低角标量 + 256 维 Transformer + 对称条件回归”在同类任务中有效。当前旧 Bert 的失败不能否定该假设，因为旧模型的峰位编码、深度和宽度都不等价。

### 6.3 实验矩阵

所有臂固定：

- 100k Niggli；
- R10 FiLM head；
- baseline SmoothL1 + 0.1 CS CE；
- 无增强；
- 与 A1 winner 同训练预算；
- seed42 探索。

#### A2-T0：Histogram 对照

- 取 A1 中胜出的 shallow 或 E1c；
- 不改任何其他设置。

#### A2-T1：T20-pos

- 前 20 个低角峰；
- token 只含 `g` 和 mask；
- 最大程度检验 AIdex 的低角位置假设；
- 不输入强度，不与 histogram 融合。

#### A2-T2：T20-posI

- 在 T1 上加入 normalized intensity；
- 回答“强度是否提供净增益”。

#### A2-T3：T48-geom

- 前 48 峰；
- token 使用 `[g, Δg, I, rank]`；
- 回答更多峰及显式间隔是否改善复杂谱。

#### A2-T4：T48-geom + histogram fusion

```text
Peak Transformer 256-d ─┐
                         ├→ concat → projection 512-d → FiLM head
Histogram encoder 256-d ─┘
```

- fusion 初版只使用 concat + Linear + LayerNorm；
- 暂不使用复杂 gate/cross-attention；
- 回答局部峰关系和全局峰分布是否互补。

### 6.4 P0 与 100k 流程

1. 单元测试：padding mask、排序、N=0/N<20/N>48、CPU/GPU shape；
2. P0-700：关闭 dropout，确认 train elementwise ≥95%；
3. 3500：只淘汰完全不收敛的臂，不按 0.xpp 排名；
4. 100k：T0/T1/T2/T3；
5. 只有 T3 优于 T0，才跑 T4 fusion；
6. winner 用 seeds 42/43/44 复核。

### 6.5 预期

- T1/T2 用于验证论文式低角输入，可能打平或小幅超过 histogram；
- T3 预期主要改善 `peak_count>20`、orthorhombic/monoclinic；
- T4 若互补成立，合理目标是相对 T0 **+2–6pp**；
- 收益不要求全部来自 overall，低对称和复杂峰改善同样可通过 Gate。

### 6.6 失败处理

- T1 失败、T3 成功：说明 20 峰不足，不否定 sequence encoder；
- T1/T2/T3 均失败：检查物理归一化、mask、容量和优化后，停止 Transformer 主线；
- T3 成功、T4 失败：保留纯 Transformer，关闭 fusion；
- 不通过时不得通过增加到 100M+ 参数挽救。

---

## 7. 阶段 A3：Reciprocal Metric `gstar6` 输出表示

### 7.1 要做什么

为 Niggli 晶胞构造 reciprocal metric：

\[
G^*=(AA^\mathsf T)^{-1}
\]

使用 Cholesky 保证正定：

\[
G^*=LL^\mathsf T
\]

网络输出六个自由参数：

```text
[log L11, log L22, log L33, L21, L31, L32]
```

每个分量使用 train-only mean/std 标准化。解码时：

1. 恢复 Cholesky；
2. 构造正定 \(G^*\)；
3. 求逆得到 direct metric \(G\)；
4. 解码为 `(a,b,c,α,β,γ)`；
5. 使用现有 strict evaluator。

### 7.2 原理

理论峰的 `1/d²` 是 \(G^*\) 的线性二次型。直接预测 reciprocal metric：

- 与峰位置物理关系更近；
- 自动保证 metric 正定；
- 避免 matrix6 输出非物理或退化基矩阵；
- 可能减轻 angle pull-to-90。

它不等于预测后做 Niggli reduction。标签在训练前已经固定为 Niggli basis，模型学习的仍是唯一标签。

### 7.3 实验矩阵

固定架构基线为 **A2-ctrl**（Peak Transformer T48-geom + `cs_conditional` L=6；A2.5 已证明 `cs_conditional ≫ film/shared`，不再使用 FiLM）：

| 实验 | 输出 | Loss | 结果（valid1400 strict） |
|---|---|---|---|
| A3-G0 | matrix6 | SmoothL1 | **40.57%**（seed42，既有 A2-ctrl） |
| A3-G1 | gstar6 | normalized gstar6 SmoothL1 | seed42/43/44 = **42.50 / 43.71 / 43.00%**；mean **43.07%**（+2.50pp）→ **锁定生产默认** |
| A3-G2 | gstar6 | G1 + decoded-cell λ=0.05 | P0-700 best **78.6%≪95%** → **淘汰**，不跑 100k |

G2 的 decoded loss：

- length 使用 relative SmoothL1；
- angle 使用 degree/3 标准化 SmoothL1；
- 初始权重 `0.05`，只在 G1 稳定后测试。

### 7.3.1 实测结论（2026-07-17）

- G1 P0-700：ep927 best strict **96%**，SPD/`min_eig(G*)>0`/finite **100%**。
- G1@100k vs G0：三 seed mean **43.07% vs 40.57%**（**+2.50pp**；95% CI ≈[41.56, 44.59]%）→ **锁定 `gstar6` 为生产默认输出表示**。
- angMAE mean 6.65° vs G0 6.26°（略升，未达 −0.3° 条件门）；ortho/tet 有改善，mono/tric 仅轻度变化。
- G2：decoded-cell 辅助损失阻碍 overfit（P0 卡在 ~78%），淘汰。
- 详细表见 `docs/实验记录/20260717-A3-gstar6输出表示.md`。
- 生产配置：`configs/scale_100k_a3_g1_gstar6.yaml`；对照保留 `configs/scale_100k_a2p5_a2ctrl_cscond.yaml`。

### 7.4 实验流程

1. 随机 Niggli cells 做 encode→decode round-trip，误差接近数值精度；
2. P0-700 train elementwise ≥95%；
3. 检查所有输出 `G*` 的最小特征值 >0；
4. 100k 做 G0/G1；
5. G1 不劣于 G0 后才跑 G2；
6. winner 三 seed 复核。

### 7.5 预期

- 合理收益：overall **0–3pp**；
- 更重要的预期是 angle MAE 下降、monoclinic/triclinic 或复杂峰子集改善；
- 非正定/无效晶胞率应降为 0。

### 7.6 失败处理

- P0 不过：先检查 row/column convention、求逆和 Cholesky stats；
- P0 通过、100k 不涨：保留 matrix6，不继续叠复杂 metric loss；
- G1 angle 改善但 strict 不涨：允许保留为 Phase B 搜索坐标候选，但不替换生产 Top-1。

---

## 8. 阶段 A4：Clean → Perturbed 鲁棒训练课程

### 8.1 要做什么

不要只用一个 `train_augment` 开关。把增强拆成可审计参数：

- `global_zero_shift_deg`：全谱统一偏移；
- `per_peak_jitter_deg`：每峰独立测量误差；
- `peak_dropout_count/rate`：随机缺峰；
- `impurity_peak_count`：随机杂质峰；
- `intensity_noise`；
- `preferred_orientation_suppression`：随机压制部分峰强度；
- `clean_sample_probability`。

训练课程：

```text
Stage 1：100% clean，训练到 clean-valid 最佳
Stage 2：从最佳 ckpt 微调
         80% clean + 20% perturbed
         LR = 原 LR 的 0.1
         约 10–20 epoch
```

初始扰动范围：

- global zero shift：`[-0.3°, +0.3°]`；
- per-peak jitter：`σ=0.05°`，截断在 `±0.15°`；
- 缺峰：0–4；
- 杂质峰：0–2，强度低于主峰的 20%；
- 强度乘性噪声：约 5–10%；
- preferred orientation：随机压制 0–30% 峰的强度。

### 8.2 原理

真实实验误差不是单一随机 shift：

- 仪器 zero error 通常是全谱相关；
- peak fitting uncertainty 是逐峰误差；
- preferred orientation 主要改变强度甚至使弱峰消失；
- 杂相产生额外峰。

课程训练先学习理想几何，再学习不变性，能降低增强破坏 clean 精度的风险。

### 8.3 固定鲁棒验证集

从 valid1400 离线生成、固定 seed 并永久冻结：

| 集合 | 扰动 |
|---|---|
| V-clean | 无 |
| V-zero | global shift ±0.1/0.2/0.3° |
| V-jitter | per-peak σ=0.03/0.05/0.10° |
| V-drop | 缺 1/2/4 峰 |
| V-impurity | 加 1/2/4 弱杂质峰 |
| V-mixed | zero + jitter + drop + impurity |

### 8.4 实验矩阵

| 实验 | 训练 |
|---|---|
| A4-C0 | clean only |
| A4-C1 | 从 epoch 0 全程 20% perturb |
| A4-C2 | **clean → 80/20 perturb fine-tune** |

先比较 C0/C2；C1 只用于验证课程是否必要。

### 8.5 预期与 Gate

通过条件：

- V-clean 相对 C0 回退 ≤0.5pp；
- V-mixed strict elementwise 相对 C0 ≥+3pp；
- 各单项扰动均无灾难性下降；
- non-cubic 不因增强进一步恶化。

若 clean 回退 >0.5pp：

- perturbed 比例降至 10%；
- fine-tune LR 再降；
- 不通过时生产模型仍使用 clean recipe。

---

## 9. 阶段 A5：平滑 strict-match 辅助 Loss

### 9.1 要做什么

当前 SmoothL1 优化平均分量误差，而 strict elementwise 要求六个参数同时过线。对 decoded lattice 构造归一化误差：

\[
e=[
|\Delta a|/(0.05a),\;
|\Delta b|/(0.05b),\;
|\Delta c|/(0.05c),\;
|\Delta\alpha|/3^\circ,\;
|\Delta\beta|/3^\circ,\;
|\Delta\gamma|/3^\circ
]
\]

使用平滑最大值：

\[
\mathcal L_{\mathrm{strict-soft}}
=\tau\log\sum_i \exp(e_i/\tau)
\]

总 loss：

\[
\mathcal L=
\mathcal L_{\mathrm{base}}
+\lambda_s\mathcal L_{\mathrm{strict-soft}}
+0.1\mathcal L_{\mathrm{CS}}
\]

### 9.2 原理

该 loss 会更关注当前最差的一个晶胞参数，比逐元素 hinge 平滑，也比单纯加 angle 权重更直接对应 strict criterion。

但当前只有约 7.2% 样本属于“六维仅错一维”的近失配，因此本阶段预期有限，优先级低于 A2/A3。

### 9.3 实验设计

固定 A3/A4 winner：

- L0：base loss；
- L1：`λs=0.01`；
- L2：`λs=0.03`；
- L3：只在 L1/L2 有正信号时测试 `λs=0.10`；
- `τ=0.5` 起步。

先记录 base loss 和 strict-soft 的初始量纲，确保辅助项梯度不支配主 loss。

### 9.4 预期与 Gate

- 合理收益：**0–1pp**；
- 或六参数中最差参数达标率提高，angle MAE ≥-0.3°；
- overall 不下降，cubic 不回退。

若 seed42 无 ≥0.5pp 或明确 angle 改善，立即停止，不做三 seed。

`peak_consistency` 已在公平对照中失败，本阶段不重新扩大其权重。

---

## 10. 阶段 A6：训练效率与数据规模

### 10.1 要做什么

在 A2–A5 胜出算法完全冻结后，只优化吞吐：

1. BF16/AMP；
2. fused AdamW（环境支持时）；
3. DataLoader workers/pin/prefetch；
4. 稳定后再试 `torch.compile`；
5. batch/LR 按实测稳定性调整，不以“填满显存”为目标；
6. 记录 samples/s、达到相同 valid 指标的墙钟。

然后按固定结构扩数据：

```text
100k → 500k → 1M → 6M
```

### 10.2 原理

更多数据只能放大正确范式。先冻结 encoder、输出表示、loss、增强，再画 scaling curve，才能知道收益来自数据而不是混杂改动。

### 10.3 实验设计

- 100k：winner 三 seed；
- 500k：seed42；
- 500k 相对 100k 有正收益后才做 1M；
- 1M 仍有明显斜率后才做 6M；
- 每个规模重新计算 train-only normalizer stats；
- valid1400 始终不变；
- 最终规模才做三 seed。

### 10.4 预期与停止条件

工程预期不是线性相加：

- 500k 相对 100k：可能 +2–5pp；
- 1M/6M：收益递减，重点看低对称是否继续增长。

停止条件：

- 500k overall <+1pp 且 non-cubic 无改善；
- train loss 降而 valid 不涨，说明表示/目标仍是瓶颈；
- 数据阶梯不通过时，不用 6M 长训掩盖算法问题。

---

## 11. Phase B1：独立峰驱动的全局候选搜索

Phase B 可以并行开发原型，但只有 Phase A winner 确定后才正式调参。

### 11.1 要做什么

从观测 \(q_i=1/d_i^2\) 独立构造候选，不只在 NN 单点附近 perturb：

```text
observed q peaks
  → 按晶系自由度生成 metric seeds
  → 枚举低阶 hkl / Bravais constraints
  → coarse-to-fine 全局搜索
  → reciprocal metric 局部精修
  → 去重、正定和体积护栏
  → 每晶系保留固定预算候选
```

NN 只提供软先验：

- CS 搜索预算；
- 体积中心和范围；
- reciprocal metric 初值；
- 不允许硬裁掉低概率晶系。

### 11.2 原理

旧 L-BFGS 失败是因为它从错误 NN 点局部启动，不能跨越 hkl 赋值和半胞/超胞盆地。传统 indexing 工具的优势来自独立组合峰和全局搜索；这也是从 20% 向更高命中率推进的必要能力。

### 11.3 实验顺序

#### B1-S0：Synthetic 单元测试

每个晶系生成已知 lattice 和 ideal peaks，不使用真 lattice 初始化：

- cubic；
- tetragonal/hexagonal；
- orthorhombic；
- monoclinic；
- triclinic。

检查：

- 真胞是否进入候选；
- 轴置换和 Niggli 约定；
- 半胞/超胞；
- 缺峰和杂质峰；
- 候选去重；
- 搜索时间。

Synthetic Top-K recall 未过 95%，不得进入 valid1400。

#### B1-S1：高对称 valid

- cubic/tetragonal/hexagonal；
- 对比 q-search only、NN only、NN prior + q-search；
- 先看 Top-100、Top-20 recall，不做 learned rerank。

#### B1-S2：低对称 valid

- orthorhombic → monoclinic → triclinic 逐级开放；
- 固定每晶系时间预算；
- 不因计算量大一次性全开。

### 11.4 预期与 Gate

成功的搜索首先表现为 pool recall，而不是 Top-1：

- valid Top-20 strict elementwise ≥30%；
- non-cubic Top-20 ≥15%；
- 相对 NN 本地候选池至少 +8pp recall；
- mapping-elementwise gap 不扩大；
- 单样本耗时满足预先约定预算。

未过 Gate 时继续改 seed 和搜索边界，不进入 learned reranker。

---

## 12. Phase B2：候选排序

### 12.1 要做什么

先构造确定性 peaks-only score：

- indexed observed peak count；
- matched fraction；
- median / mean / max `|Δq|`；
- 未解释观测峰惩罚；
- 理论峰过密惩罚；
- `|log(V/V_nn)|`；
- CS probability；
- 半胞/超胞标记；
- reciprocal metric 距离 NN proposal。

先用确定性规则；只有排序效率不足时再训练小型 pairwise/listwise reranker。

### 12.2 原理

Top-K recall 是 Top-1 上限。FOM 只有在候选池包含真解时才有意义。排序必须惩罚“峰拟合看似好但体积错误”的半胞/超胞。

### 12.3 实验设计

固定 B1 候选池，对比：

1. NN confidence；
2. 当前 R6-C FOM；
3. 新 deterministic score；
4. learned reranker（仅在 3 不足时）。

定义：

\[
\mathrm{ranking\ efficiency}
=
\frac{\mathrm{ranked\ Top1\ strict}}
{\mathrm{TopK\ strict\ recall}}
\]

learned reranker：

- 训练候选只来自 train split；
- 正样本为 strict match 或连续 lattice error 最小候选；
- hard negatives 包括半胞、超胞、错 setting、峰拟合近似候选；
- valid1400 调超参；
- MP100 不参与训练或调参。

### 12.4 预期与 Gate

- ranking efficiency ≥75%；
- valid ranked Top-1 ≥25%；
- 半胞/超胞错误率显著低于 legacy FOM；
- 若 pool recall 已高但 Top-1 低，继续优化排序；不得返回扩大主 encoder。

---

## 13. Phase B3：可选迭代 Refiner

只有 B1 recall 和 B2 ranking 已通过才进入。

### 要做什么

对 Top-M 候选计算理论峰与观测峰的残差特征，迭代 1–3 步：

\[
G^*_{t+1}=G^*_t+\Delta_\theta(
E_{\mathrm{PXRD}},
G^*_t,
R(G^*_t,\mathrm{PXRD})
)
\]

每步保持 \(G^*\) 正定，并重新计算 peak residual。

### 原理

Refiner 适合微调已经在正确盆地内的候选，不适合从错误单点寻找真胞。因此它必须后置，不能替代 B1。

### 实验设计与预期

- 0/1/3 步对照；
- 只对 B1 Top-20 做；
- 记录每步 peak score 和 lattice error 是否同向下降；
- Gate：ranked Top-1 ≥+2pp，或 Top-20 recall 不变且 angle MAE 显著下降。

若 peak score 改善但 strict 变差，说明 residual objective 偏好等价/伪胞，停止 refiner。

---

## 14. 最小可执行实验清单

按优先级和依赖严格执行：

| 顺序 | ID | 核心问题 | 运行规模 |
|---:|---|---|---|
| 1 | A0 | 指标和协议是否可信？ | 无训练 |
| 2 | A1-M0 | R10 21.29% 能否复现？ | 100k |
| 3 | A1-M1 | E1c 能否迁移到 100k？ | 100k |
| 4 | A2-T1 | 20 个低角位置是否足够？ | P0→100k |
| 5 | A2-T2 | 强度是否有净增益？ | P0→100k |
| 6 | A2-T3 | 48 峰几何 attention 是否改善复杂谱？ | P0→100k |
| 7 | A2-T4 | Transformer 与 histogram 是否互补？ | 仅 T3 通过后 |
| 8 | A3-G1 | reciprocal metric 是否优于 matrix6？ | P0→100k |
| 9 | A4-C2 | clean→perturb 是否兼顾精度和鲁棒？ | winner 100k |
| 10 | A5-L1/L2 | smooth strict loss 是否有小增益？ | winner 100k |
| 11 | A6 | 正确算法能否随数据规模增长？ | 500k→1M→6M |
| 12 | B1 | 独立 q-search 能否扩大 pool recall？ | synthetic→valid |
| 13 | B2 | 能否把 recall 转成 Top-1？ | 固定 candidate pool |
| 14 | B3 | 正确盆地内精修是否有效？ | 可选 |

第一批真正需要实现和运行的只有：

```text
A0
→ A1-M0/M1
→ A2-T1/T2/T3
→ 通过后才做 T4 和 A3
```

---

## 15. 工程改动地图

### A0 评测

- `src/pxrd_cell_indexing/eval.py`
- `src/pxrd_cell_indexing/training/trainer.py`
- `scripts/eval_valid.py`
- `scripts/eval_mp100.py`

### A2 Peak Transformer

- 新增 `src/pxrd_cell_indexing/model/encoder/peak_transformer.py`
- 修改 encoder factory / `IndexingModel`
- `training/config.py` 新增 Transformer 与 fusion 参数
- `data/peak_features.py` 新增 `Δg`、rank、padding token builder
- 新增 encoder shape/mask/overfit tests

### A3 Reciprocal Metric

- `data/normalization.py` 新增 `gstar6` normalizer
- `geometry.py` 新增 direct↔reciprocal metric 与 Cholesky round-trip
- `training/config.py` 扩展 representation literal
- 训练数据 stats 脚本支持 `gstar6`
- evaluator 仍统一解码到 lattice6

### A4 课程增强

- `data/dataset.py` 将增强拆成显式参数
- `training/config.py` 增加 clean probability 和各扰动范围
- 新增固定 robust-valid 生成脚本

### A5 Loss

- `losses.py` 新增 smooth strict auxiliary loss
- `trainer.py` 记录各 loss 分量与梯度尺度

### Phase B

- 独立搜索模块，不塞入训练 forward；
- candidate schema 固定后再实现 reranker；
- 所有候选必须记录来源、CS、体积、peak score 和去重键。

---

## 16. 资源与并行安排

### 主训练资源

严格串行：

```text
A1 → A2 → A3 → A4 → A5 → A6
```

原因是每一阶段都依赖上一阶段 winner，提前组合会造成无法归因。

### 可并行工作

- A0 评测护栏与 A1 config 准备；
- A2 单元测试与 A1 训练；
- A4 固定鲁棒 valid 集构建与 A2/A3 训练；
- B1 synthetic search 原型与 Phase A 训练，但不提前用 valid/MP100 调参。

### 三 seed 使用

- 探路全部 seed42；
- 只有通过 Gate 的候选才运行 43/44；
- 不对失败方向做“多 seed 找最好值”。

---

## 17. 决策树

```text
A0 无法复现 R10
  → 修协议，不训练

A1 E1c@100k 未过 Gate
  → 停止所有更大 Dense/U 系列
  → shallow histogram 作为 A2 对照

A1 通过
  → E1c 作为 A2 对照和可选 fusion 分支

A2 所有 peak Transformer 未过 Gate
  → 保留 histogram winner
  → 直接验证 A3 reciprocal metric

A2 有 winner
  → 三 seed
  → 再做 A3

A3 不优于 matrix6
  → 生产输出保留 matrix6
  → Phase B 内部仍可使用 G* 作搜索坐标

A4 clean 精度回退 >0.5pp
  → 降扰动比例/LR
  → 仍失败则生产模型不启用增强

A5 无 ≥0.5pp 信号
  → 停止 loss sweep

A6 500k 无 ≥1pp 收益
  → 不上 6M
  → 转向 B1 搜索

B1 pool recall 未涨
  → 改独立 seed/search，不做 reranker

B1 recall 涨、B2 Top-1 不涨
  → 排序是瓶颈，做 hard-negative reranker
```

---

## 18. 合理预期

以下是工程预期，不是承诺，也不能简单相加：

| 阶段 | 相对同阶段对照的合理预期 |
|---|---|
| A1 E1c@100k | 0–3pp |
| A2 Peak Transformer/fusion | 2–6pp，重点看复杂峰与低对称 |
| A3 gstar6 | 0–3pp，重点看 angle/低对称 |
| A4 curriculum | clean 基本持平，扰动集 +3pp 以上 |
| A5 smooth strict loss | 0–1pp |
| A6 500k→1M | 2–8pp，取决于 scaling slope |
| B1 全局搜索 | Top-K recall 可能获得最大增量 |
| B2 排序 | 将已有 recall 的 ≥75% 转成 Top-1 |

直接神经网络从当前 21.29% 一步达到 McMaille/JADE 约 66–68% 不现实。更合理的产品路线是：

```text
Peak Transformer / metric model
  → 更可靠的 raw 初值与搜索先验
  → 独立 q-search 扩大真解召回
  → peaks-only reranker 转成最终 Top-1
```

---

## 19. 明确停止项

以下方向不再进入主实验矩阵：

- U2/U3 级别的 200M–600M Dense histogram 模型；
- 单纯增加 histogram bins / sorted peak count；
- spectrum-only 或已打平的 spectrum fusion；
- 旧 discrete-2θ Bert 复跑；
- SG、flat-37、99-EG 多任务头；
- formula / atom 条件；
- naïve prediction 后 Niggli reduction；
- 提高已失败 peak-consistency loss 权重；
- raw pool recall 未提高时继续调 FOM；
- 用 loose mapping 或 oracle Top-K 宣称产品提升；
- 在算法未通过 100k Gate 前使用 6M 全量长训。

---

## 20. 最终执行原则

1. **先确认测得对**：A0 固化 strict peaks-only 评测。  
2. **先验证现有容量结论**：只跑一次 E1c@100k，不继续盲目堆 Dense。  
3. **优先恢复峰间结构**：AIdex 式低角 Peak Transformer 是下一项核心算法。  
4. **让输出贴近物理方程**：验证正定 reciprocal metric，而非预测后硬做 Niggli。  
5. **鲁棒性后置微调**：clean→perturbed，不牺牲 ideal 精度换表面鲁棒。  
6. **loss 只做小修正**：表示和输出坐标优先于 strict surrogate。  
7. **通过后再放量**：100k→500k→1M→6M，只改数据规模。  
8. **最终必须补全局搜索**：低对称 indexing 不能只靠单点回归与局部 refine。  

下一步执行入口明确为：

```text
A0 评测护栏
  → A1 E1c@100k
  → A2 Peak Transformer 三臂
```

在这三步完成前，不启动 U 系列、peak-loss sweep、全量训练或 FOM 大扫参。
