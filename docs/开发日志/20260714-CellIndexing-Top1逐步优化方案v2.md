# 2026-07-14 — Cell Indexing Top-1 逐步优化方案 v2

> **状态（2026-07-15）**：本文已由 [`20260715-CellIndexing-可执行优化方案v3.md`](20260715-CellIndexing-可执行优化方案v3.md) 取代。v3 已纳入 R10/R11b 实验结果、AIdex-R2 ONNX 架构核验、Peak Transformer、reciprocal metric、鲁棒训练课程及逐阶段实验 Gate；后续执行以 v3 为准。  
> **目标**：在严格 peaks-only 契约下，逐步提升 MP100 的最终 Top-1 lattice match。  
> **主口径**：elementwise，`ltol=0.05` / `atol=3°`。  
> **原则**：不追求一次性重构；每一步只验证一个核心假设，通过 Gate 后再进入下一步。  
> **输入边界**：模型和搜索只允许使用 PXRD 峰表 `(2θ, I)` 与波长 λ；禁止 formula、atom types、atom count 进入推理。  
> **修订**（同日后续）：峰为主、规范胞必做、不做 SG、lattice 为主 KPI；**后续优化收成三条主线——① Encoder 加强特征 ② 输出头做减法并加深 ③ 训练策略拉满（loss + 显存利用率 + 速度）**；以抬高 raw 初值为先，多假设/搜索为初值过关后的第二阶段。  
> **修订**（同日再订）：执行顺序改为 **先算法、后训练/数据、最后 Phase B**——先把结构与目标函数拉满初值（R9→R10→**R11b**），再抠训练策略与数据量（**R11**→扩数据），最后多假设/搜索。编号仍保留 R11=训练、R11b=Encoder，但 **R11b 先于 R11 执行**。  
> 连续谱 CNN 归入 Encoder 旁路；全量约 600 万可用，瓶颈在算法与训练效率，而非“再凑数据”。

---

## 0. 产品决策锁定（2026-07-14）

以下边界后续实验默认遵守，不再作为开放选项反复讨论：

| 决策 | 含义 |
|---|---|
| **峰为主** | 主输入仍是峰表 / \(Q\)；主 encoder 继续以 `1/d²` histogram（及后续峰几何特征）为核心 |
| **连续谱 CNN 可选** | 可由峰重建或原始 `.xy` 得到连续谱，经 1D CNN/ConvNeXt 作 **fusion 旁路**；必须用消融证明 ≥ 纯峰，否则关闭 |
| **规范胞必做** | 训练标签与评测真值统一到同一规范表示（优先 Niggli-reduced，或书面固定的唯一约定）；换口径后旧数字不可直接对比 |
| **不做 SG 多任务** | 不上 230 空间群头；不为 GEMD/SG 冲榜。CS 分类只服务 lattice 路由，不是第二产品目标 |
| **lattice 为主 KPI** | 最关心 **CS 正确子集上的 lattice elementwise**；总分涨但 CS-correct lattice 不涨，不算有效进步 |
| **输出头做减法并加深** | 去掉 setting 专家丛；默认 **共享/FiLM 条件化 lattice 头**；深度与容量加在 encoder/骨干，不加在路由拓扑上 |
| **训练策略拉满** | 设计好 loss 与优化器；在 **显存打满** 前提下追求步速与收敛；主目标是压低有效 loss、抬高 raw / CS-correct lattice |
| **数据不缺** | 全量约 **600 万** 独立样本可用；未过算法 Gate 前不上全量，过 Gate 后可直接按阶梯扩到全量 |

**刻意不做 / 不再论证的方向**：

- AlphaDiffract 式 SG / 多任务联合头；
- 用 formula / atom 图换分；
- 用 continuous-spectrum **替换**峰表作为唯一输入；
- 在单点路由上继续堆 setting / 更多专家头；
- **初值未拉高前**把主资源砸进复杂搜索/reranker；
- 算法未过 Gate 时用“数据不够”解释失败。

---

## 0.1 后续优化总纲：先算法、后训练、再放量

总原则（2026-07-14 锁定）：

```
Phase A-算法：把「学什么 / 用什么结构」定对
  R9 规范胞 → R10 头减法/加深 → R11b Encoder 加深
  目标：raw 初值精度拉满；有效 loss 尽量贴近 elementwise match 标准
        ↓
Phase A-训练：在已定算法上抠收敛与规模
  R11 训练策略（schedule / batch / AMP / peakλ 细扫）→ 数据阶梯 → 全量
  目标：同结构下把指标与精度再拉到极致
        ↓
Phase B：多假设 / 搜索 / 排序 / refine
  仅当 Phase A 初值与可复现提升已过关
```

三条能力主线仍在，但 **执行顺序按上表，不再按①→②→③编号串行**：

| 主线 | 要解决什么 | 落在哪一段 | 成功长什么样 |
|---|---|---|---|
| **② 输出头** | 头过重、容量用在路由而非拟合 | **A-算法（R10）** | setting 删除；slim/FiLM 头 raw 不掉或更好 |
| **① Encoder** | 峰几何信息不够强、表示偏浅 | **A-算法（R11b）** | 同头、同最小训练底板上 raw / 角误差明显改善 |
| **③ 训练策略** | 吞吐、schedule、目标函数细调、数据量 | **A-训练（R11→扩数据）** | 同结构下 valid 再涨；墙钟与显存打满 |

**最小训练底板**（大 batch + sqrt LR 等）可在算法阶段沿用，避免深模型训不稳；**完整** R11（含 peak λ 细扫、长训打磨）放在结构定稿之后。

旧三条原则（多假设、闭环、峰对齐）**不废弃**：峰对齐 / match 导向的目标函数可在算法段做粗验、训练段做细扫；多假设与闭环进 Phase B。口号改为：

> **先把算法初值做准做深；再把训练与数据量拉满；最后才做搜索补召回。**

---
## 1. 当前起点

### 1.1 冠军模型

`scale_100k_r3_cubic_split_clf_seed42`

```
峰表
  → 1/d² histogram encoder
  → 512-d embedding
  → 晶系分类 + 晶系条件化回归头
  → cubic P/F/I setting 分头
  → 一个 matrix6 晶胞预测
  → Bravais / Top-K
  → FOM 排序
```

当前标签为 **primitive lattice → matrix6**，**尚未**统一到规范胞；立方 setting 分头是 primitive 口径下的有效补丁。**输出头目标是做减法**：规范胞后默认删除 setting 专家丛，避免用路由复杂度掩盖标签歧义。

### 1.2 当前严格指标

| 数据集 | 指标 | 当前值 |
|---|---|---:|
| MP100 | raw Top-1 elementwise | **13%** |
| MP100 | raw Top-1 mapping | 14% |
| MP100 | Top-20 elementwise（旧默认池） | **14%** |
| MP100 | Top-20 mapping | 38% |
| valid1400 | raw Top-1 elementwise | **15.43%** |
| valid1400 | Top-K elementwise（R6-C 后） | **17.93%** |
| valid1400 | FOM Top-1 elementwise（R6-C 后） | **14.79%** |
| valid1400 | 非立方 raw elementwise | **3.17%** |
| valid1400 | angle MAE | **8.69°** |

外部 peaks-only 参考：

- McMaille：MP100 strict Top-1 ≈ **65.9%**
- JADE9：MP100 strict Top-1 ≈ **68.1%**

RealPXRD Without-L 的 67% Top-20 使用 formula，不属于 indexing 对照；纯 XRD 消融在 elementwise 下为 0%，不能作为捷径。

### 1.3 当前瓶颈

```
峰 → raw 单点 13–15%
       ↓
窄邻域候选池 14–18%
       ↓
严格安全的排序 ≤15%
```

1. **标签 setting 歧义**（规范胞未统一）放大条件均值与角拉向 90°；  
2. **输出头过重**：7 CS 回归 + setting 分头 + 双分类，复杂度用在单点路由，而非多解表达；  
3. **raw 初值对低对称很差**；  
4. **候选池贴着 raw，缺少独立峰搜索**；  
5. 排序上限受池召回限制；模型只输出一个点，不表达多解性。

---

## 2. 已确认的经验与教训

### 2.1 已经确认有效，应冻结

1. **输入表示**：`1/d² histogram + I>5` 明显优于 RealPXRD peak-token Bert；峰表主线冻结。
2. **标签表示（几何）**：matrix6 足够；matrix9 没有显著收益。
3. **条件化回归（历史结论）**：在 primitive 口径下，7 晶系分头曾优于共享头；**规范胞后须重测**，不默认永久冻结为“越分越细”。
4. **立方 setting 分头（历史补丁）**：只在 **primitive** 口径下证明有用；规范胞后列为 **待删除默认项**，不是长期架构。
5. **严格选模**：checkpoint 必须按 strict elementwise 选。
6. **FOM 体积修复**：使用 NN 预测体积作参考，避免默认偏好半胞。
7. **评测护栏**：elementwise 是主指标；mapping 只作诊断。

### 2.2 已失败，不再原样重复

1. peak-token Bert、短训 continuous-2θ Bert（**不等于**禁止后续“峰为主 + 谱旁路 fusion”消融）。
2. 单纯增加 histogram bins、重新放入所有弱峰。
3. angle-heavy、strict hinge、angle prior、manifold consistency。
4. 盲目 hard-CS 过采样 / finetune。
5. 直接把已有 7 个晶系头当 multi-anchor；它们并未按候选集目标训练。
6. 从错误 NN 单点出发做局部 L-BFGS；R6-B 已证明容易困在错误盆地。
7. 无体积护栏的尺度扩池与旧 FOM；会产生大量半胞/超胞伪命中。
8. **未过算法 Gate 就扩规模**；数据有 600 万，但错误范式放大仍无效。
9. 用 loose 0.3/10° 或 mapping-only 宣称进步。
10. **SG / 多任务冲榜**换取表面上的“更完整晶体学监督”。
11. **在单点输出上继续堆专家头**（更多 setting / 更细路由）——复杂度用错地方。

### 2.3 优化原则（更新）

**Phase A 主原则（当前）**：

1. **特征要强**：Encoder 优先吸收可迁移的几何/谱表示，不做已证伪的 Bert 主路。
2. **头要简单且深**：删 setting / 专家丛；容量加在更深的共享或 FiLM 骨干。
3. **训练要满**：loss 对齐峰与规范胞；显存打满、步速要快；用墙钟效率而不是只看 epoch 数。

**Phase B 原则（初值过关后）**：

1. **多假设**：单点不够时再上可训练 K 候选。
2. **闭环 / 峰搜索**：PXRD 进入候选更新与精修。
3. **峰对齐排序**：Top-1 由峰打分决定。

### 2.4 从调研可迁移 / 不可迁移

**可迁移 → 主要进 ① Encoder / ③ 训练**：

- 规范胞 / Niggli 统一标签；
- 连续谱 1D CNN/ConvNeXt 作 **特征增强旁路**（AlphaDiffract 类）；
- 更深的 1D/MLP 骨干、更好的归一化与训练配方；
- 峰对齐 / ΔQ 类辅助 loss（进训练策略，不必等搜索）。

**可迁移 → Phase B**：

- 多假设、峰条件闭环、峰驱动搜索与排序。

**不可迁移**：

- formula / atom 图条件、联合原子坐标扩散；
- oracle Top-20 当产品指标；
- SG 多任务头；
- 用结构生成模型的 lattice 指标当 indexing 基线；
- peak-token Bert 作为主 encoder。

---

## 3. 总体任务与三条主线设计

### 3.0 当前 vs 目标（网络侧优先）

当前：

\[
\hat L=f_\theta(\mathrm{PXRD})
\quad\text{（复杂头 + 浅 histogram MLP + 未拉满训练）}
\]

Phase A 目标：把 \(f_\theta\) 本身做强——

\[
\hat L = \mathrm{Head}_{\mathrm{slim}}\big(\mathrm{Enc}^*(\mathrm{PXRD})\big)
\quad\text{规范胞口径，raw 初值尽量准}
\]

Phase B 目标（后置）：

\[
\mathcal C=\mathrm{Propose}_\theta(\mathrm{PXRD}), \qquad
\hat L=\arg\min_{L\in\mathcal C} S(L,\mathrm{PXRD})
\]

最终 pipeline：

```
峰表 (+ λ)  ──→  加强版 Encoder（主）  ──┐
                                         ├→ fusion → 简化且加深的 FiLM/共享头 → matrix6（或 K×）
连续谱（可选）→ 1D CNN 旁路 ─────────────┘
                                         ↓
                    （Phase B）搜索 / 排序 / refine → Top-1
```

### 3.1 主线①：优化 Encoder，加强特征

**定位**：在峰为主前提下，把表示做强；参考外部算法时只搬“特征怎么抽”，不搬产品定义。

**保留底座**：`1/d²` histogram + `I>5` 仍是主输入契约的一部分（可增强，不轻易换掉）。

**增强方向（按优先级）**：

1. **加深峰几何骨干**：histogram → 更深 MLP / ResMLP / 轻量 1D 卷积栈；加宽 embedding（如 512→768/1024），用 P0 确认仍可过拟合。
2. **显式 Q / 峰统计特征**：排序 Q、峰数、强度矩、低阶差分等与 hist 拼接（既往有部分；系统消融哪些有用）。
3. **连续谱旁路（可选）**：由峰重建或 `.xy` → 1D ConvNeXt/CNN → 与峰 embedding fusion；必须 `fusion ≥ peak-only`。
4. **不做什么**：不回到 peak-token Bert 主路；不靠“加 bins / 加回弱峰”单点扫参当主增量。

**Gate**：同 slim 头、同训练预算下，valid raw ew 或 CS-correct lattice / angle 优于旧 encoder。

### 3.2 主线②：优化输出头——简化 + 加深，抬初值

**推荐默认形态（S1）**：

```
embedding
  ├─ CS 头（轻量，小权重 CE）
  └─ 更深的共享 lattice MLP
        ← CS 做 FiLM / concat
        → matrix6（规范胞）；日后可扩成 K×6
  └─（可选）晶系流形投影
```

**减法**：删除 setting 分类与立方 P/F/I 三头。  
**加法**：把原专家丛的参数量 **挪到更深的共享/FiLM 骨干**，而不是挪到更多路由。  
**裁决**：H0→H1（去 setting）→H2/H3（共享/FiLM+加深）；冻结 `R10-slim`（可含 deepened backbone）。

详见原 R10 阶段；与 ① 的配合顺序建议：**先能跑通 slim 头，再换强 Encoder**，或 Encoder/头分开单因素，禁止一次改俩。

### 3.3 主线③：优化训练策略——loss、优化与显存打满

**Loss（规范胞上）**：

\[
\mathcal L=
\mathcal L_{\mathrm{SmoothL1}}
+\lambda_{\mathrm{peak}}\mathcal L_{\mathrm{peak}}
+0.1\mathcal L_{\mathrm{CS}}
\]

- 无 SG、无 setting CE；
- 峰一致性进训练（不必等搜索）；
- 盯 CS-correct lattice，而非 CS 准确率冲榜。

**优化与吞吐**：

| 手段 | 目的 |
|---|---|
| 尽量大的 batch（显存打满） | 稳定梯度、提高样本吞吐 |
| AMP / `torch.compile`（若稳定） | 提速 |
| 合理 LR schedule（warmup + cosine 等） | 压到更低 loss |
| 梯度累积（单卡不够时） | 等效大 batch |
| DataLoader：多 worker、pin memory、预取 | 避免 GPU 饿死 |
| 选模仍用 strict elementwise | 防止只报 train loss |

**速度 KPI**：samples/sec、step time、达到目标 valid 指标的墙钟；不只报 epoch。

**Gate**：同结构下，训练配方使 valid loss / raw 指标优于旧配方；显存利用率高且无异常 OOM 抖动。

### 3.4 目标输出头（与②一致）

**当前（过重）**：

```
embedding
  → CS 分类 + 7 个晶系回归专家
  → setting 分类 + 3 个立方 setting 专家
  → 路由出一个 matrix6
```

**目标**：

```
embedding（加强版 Encoder）
  → 轻量 CS + 更深 FiLM/共享 lattice → matrix6（规范胞）
```

| 删除 / 收敛 | 保留 / 后移 |
|---|---|
| setting 分头 + setting 分类 | 规范胞 |
| 7 专家硬路由 | FiLM / 共享加深 |
| SG 头 | — |
| 复杂搜索当第一优先级 | Phase B |

---
## 4. 统一实验纪律

### 4.1 数据使用

| 层级 | 用途 |
|---|---|
| P0-700 | 过拟合与梯度正确性；不判断泛化 |
| 10k | smoke / 排除明显失败；不做最终裁决 |
| 100k + valid1400 | 算法阶段训练、选模和超参裁决 |
| 分层 500k → 1M → **全量 ~600 万** | 仅算法 Gate 通过后；放大正确范式 |
| MP100 | 阶段性最终验收；禁止用于连续调参 |

MP100 每个阶段只运行一次“通过 Gate 的候选”，防止对 100 个样本反复调参导致 benchmark 泄漏。

**说明**：全量数据充足，不把“再扩数据”当作主叙事；但扩规模仍要求 **算法已过 Gate**，否则 600 万只会放大错误目标。

### 4.2 固定主指标

所有阶段必须同时报告：

1. `raw_top1_elementwise_rate`
2. `topk_elementwise_rate`
3. `ranked_top1_elementwise_rate`
4. **`cs_correct_subset_lattice_elementwise`**（主诊断：CS 预测正确时的 lattice ew）
5. `cs_accuracy`（仅诊断，不为冲榜）
6. `raw/topk mapping rate`（仅诊断）
7. `mapping_vs_elementwise_gap`
8. per-crystal-system elementwise
9. angle MAE、length MAPE、pull-to-90
10. 候选数、单样本推理时间

**有效进步判定**：产品 Top-1 ew 提升，且 **CS-correct lattice ew** 不同步变差；仅靠 CS 涨分而 lattice 几何不改善，不算过关。

### 4.3 单因素与复现

- 每轮只改一个核心因素。
- 探索阶段固定 seed=42。
- 通过 Gate 后用 seeds 42/43/44 复核均值和方差。
- 新 encoder / loss / **简化头** / 多假设头 / 规范胞标签必须先过 P0-700。
- 所有结果保存 config、checkpoint、JSON、日志和分样本结果。
- **换规范胞口径后**：必须重跑 baseline，旧 primitive 数字只作历史对照。

---

## 5. 分阶段实施方案

> **读法**：R7–R9 是底座；**Phase A-算法** = R10 头 → **R11b Encoder**；**Phase A-训练** = **R11** 策略/参数 → 扩数据；**Phase B** = R12 起。编号上 R11=训练、R11b=Encoder，但执行 **R11b 先于 R11**。初值未过关前，主资源不进搜索/全量。

## 阶段 R7：基线与评测闭环固化

### 原理

先确保最终 Top-1 的定义、候选生成和排序配置一致。当前 MP100 champion JSON 仍使用旧 FOM，`fom_top1_elementwise=0%`，而 valid1400 上 R6-C 已将其修到 14.79%。在此之前无法判断后续增量。

### 方法

固化严格 elementwise 推理 profile：

- `bravais_set=extended`
- elementwise profile 默认关闭尺度变体
- `max_log_volume_ratio_vs_base=log(2)`
- FOM 使用 `ref_volume=NN raw volume`
- mapping 与 elementwise 同时输出，但 elementwise 为主
- 增加 `cs_correct_subset_lattice_elementwise` 字段

对 champion 重新跑 valid1400 和 MP100，形成唯一基线 JSON。

增加 peaks-only 契约检查：

- `IndexingModel.forward` 不接受 atom/formula；
- dataloader 的 `atom_num` 只能用于统计，不能进入模型；
- MP100 推理只从 CIF 生成峰和真值标签。

### 实验设置

- checkpoint：当前 champion
- K：20
- `ltol=0.05`，`atol=3°`
- valid1400 先跑；配置锁定后跑一次 MP100

### 通过 Gate

- 产出唯一 canonical baseline；
- FOM 不再出现 mapping 高、elementwise=0 的明显半胞崩塌；
- 所有后续实验复用同一 profile。

### 失败处理

若 MP100 仍为 FOM elementwise=0，优先修复 `eval_mp100.py` 与 `fom_rerank.py` 参数透传，不启动训练。

---

## 阶段 R8：最基础的回归可学习性闭环

### 原理

R0b 已证明 histogram + MLP 在 700 样本上可达到约 99% train elementwise；但近期 R6-A 的 champion 过拟合探针尚未达到 80%。在增加新 loss / 换规范胞前，必须先确认“encoder + 多专家路由 + matrix6 回归”本身能在小数据上拟合。

### 方法

按复杂度逐级恢复：

1. histogram encoder + 单共享回归头；
2. histogram encoder + 7 晶系头，训练使用 oracle route；
3. 加 cubic setting 分头，训练使用 oracle setting；
4. 加预测分类路由，只评估 deploy gap。

过拟合探针设置：

- 固定 700 样本，训练集即评测集；
- 关闭数据增强；
- 关闭 early stop；
- 训练最多 200 epoch；
- probe 可临时将 dropout 设为 0；
- loss 先只用 baseline SmoothL1 + 必要分类 CE（**不含 SG**）。

### 实验设置

每一级记录：

- train strict elementwise
- angle MAE / length MAPE
- oracle route vs predicted route
- 各头样本数与梯度范数

### 通过 Gate

- 单头与 oracle 多头 train elementwise ≥ **80%**
- angle MAE ≤ **1°**
- 无长期不更新的有效专家头

### 失败处理

- 单头失败：检查 normalizer、matrix6 round-trip、学习率和 loss。
- 单头通过、多头失败：检查路由、样本不平衡、只选中头的梯度流。
- oracle 通过、predicted 失败：只修分类路由，不改回归表示。

只有 R8 通过后，才进入规范胞、**输出头减法**与峰一致性 loss。

---

## 阶段 R9：规范胞标签与评测对齐（必做）

### 原理

同一物理胞的多种 setting / 轴置换会使回归学到条件均值，表现为角向 90° 收缩、低对称极差。规范胞把“学什么”钉死到唯一表示，是输出头做减法与后续峰损失的底座。

### 方法

1. **选定唯一约定**（推荐 **Niggli-reduced**；若工程上更稳，可书面固定另一套 reduced + 轴序约定，但全链路只能有一种）。
2. 离线对 train / valid / MP100 真值重算规范胞；训练与评测使用同一转换。
3. matrix6 / normalizer 在新标签上重 fit；旧 primitive 模型数字不再直接比较。
4. **本阶段暂保留 champion 头结构**，只换标签，避免“换标签 + 改头”混杂；头减法放到 R10。
5. **不做 SG 标签管线**；规范胞只服务 lattice。

### 实验设置

| 对照 | 说明 |
|---|---|
| A | 现有 primitive 标签（R7/R8 基线） |
| B | 规范胞标签 + **同结构** champion 重训（100k） |

P0-700 先确认规范胞 round-trip 与过拟合仍可达 80%+。

### 通过 Gate

- P0：规范胞标签下 train ew ≥ **80%**，round-trip 无系统偏差；
- 100k valid：相对同预算 primitive 重训对照，**angle MAE 下降** 或 **非立方 ew 提升 ≥1pp**，且 cubic 不低于可接受阈值（建议 ≥85%）；
- 评测脚本与 CIF→标签链路全部切到规范胞；产出新的 “R9-canonical baseline” JSON。

### 失败处理

- 若 Niggli 实现有数值抖动：固定实现/库版本，加容差去重；不得混用两套标签。
- 若规范后总分短期下降但 CS-correct lattice / 角误差改善：可接受，以诊断指标裁决，不立刻回退。
- 若完全崩：检查 primitive↔Niggli 转换与 evaluator 是否一致，**不进入 R10 头减法**。

---

## 阶段 R10：输出头做减法并加深（主线②）

### 原理

当前输出头过重的根因，是用 **setting 专家 + 双分类路由** 去补 primitive 标签歧义。规范胞之后，这块补丁应默认拆掉。

单点头再复杂，也解决不了低对称欠定；继续加专家只会增加训练脆弱性（死头、路由 gap），并把工程注意力从多假设/搜索上带走。

本阶段目标：**删掉用错的复杂度，把容量加回更深的共享/FiLM 骨干，抬高 raw 初值**。

### 目标形态

优先收敛到下面之一（由简到繁，先试更简的）：

**形态 S0（首选）**：

```
embedding → 共享 lattice 头 → matrix6（规范胞）
         → 轻量 CS 分类（仅先验 / 诊断，损失权重小）
```

推理时可用 CS 概率做搜索预算，但 **lattice 不依赖 hard route 才能反传**（训练默认直通共享头，或 teacher-force 仅作对照）。

**形态 S1（推荐默认 = 简化 + 加深）**：

```
embedding → CS 分类（软，小权重）
         → 更深的条件 MLP（FiLM / concat；同一套权重）→ matrix6
```

仍是 **一套回归权重**；深度（层数/宽度）相对旧单头 **显式加大**，参数量来自删除的专家丛再分配。

**形态 S2（仅当 S1 仍不够）**：

```
embedding → 7 个轻量 CS 回归头（无 setting 头）
         → CS 路由
```

保留历史“分头有收益”的最小版本；**禁止**再挂 setting 分类与 3 个立方专家。

### 明确删除

- setting classifier
- cubic P/F/I 三个 setting 回归头
- 任何“为单点再拆 setting / 再拆子空间群”的扩展
- 把未按候选目标训练的 7 头输出直接拼成 multi-anchor（R5 已证伪）

### 方法（消融顺序，单因素）

在 **R9 规范胞标签** 上，同数据同预算：

| 臂 | 结构 |
|---|---|
| H0 | R9-canonical：完整 champion 头（含 setting） |
| H1 | 去掉 setting 头与 setting CE；保留 7 CS 专家 |
| H2 | 共享头 + 轻量 CS（S0），可加深 |
| H3 | 条件 FiLM 共享头（S1）+ **加深**（推荐主候选） |

裁决规则：

1. 先看 **CS-correct lattice ew**、angle MAE、非立方 ew、cubic ew；
2. H1 相对 H0 **不显著变差** → setting 头删除，记入默认；
3. 在 H1 基础上，H2/H3 若 raw / CS-correct lattice 不低于 H1 超过容忍阈值（建议 raw ew 回退 ≤1pp，且 CS-correct 不崩）→ 采用更简形态为 **R10-slim 默认骨干**；
4. 若 H2/H3 明显伤分 → 停在 H1（无 setting 的 7 CS 头），**不再加回 setting**。

### 实验设置

- P0-700：每个形态先过拟合 ≥80%，确认简化后仍可学；
- 100k + valid1400：主裁决；
- 训练 loss：SmoothL1(+ 可选小权重 CS CE)；**无 setting CE**；本阶段不加 peak loss（留给 R11）；
- 选模：`strict_raw_top1_elementwise_rate` + 监控 CS-correct lattice；
- 记录参数量、路由死头率、oracle-route vs predicted-route gap。

### 通过 Gate

- **必须**：删除 setting 丛（H1 或更简）成为默认；setting 不再出现在主线 config；
- P0：选定形态 train ew ≥ **80%**；
- 100k：相对 H0，CS-correct lattice / 非立方 **不显著变差**（允许总分小幅波动，但不得靠 setting 补丁挽留）；
- 产出 `R10-slim` checkpoint，作为后续 R11–R12 的唯一骨干。

### 失败处理

- H1 就大幅掉 cubic：检查规范胞是否仍混有 setting 等价；先修标签，不恢复 setting 头；
- 共享头（H2）掉太多：退到 H3 或 H1，**仍禁止**加回 setting；
- 训练不稳：降 CS CE 权重或改 oracle-CS 训练 + 推理软先验，不把头再拆细。

### 与后续阶段的接口

- **下一步先做 R11b（Encoder）**，再做 R11（完整训练策略）；二者都架在 `R10-slim`（及其中选中的加深头）上；
- **R12 多假设** 只架在算法定稿骨干上；
- 多假设 = 在简化头上出 K 个 matrix6，**不是** 7×3 专家再乘 K；
- 搜索阶段的 Bravais / setting 枚举属于 **候选生成**，不属于网络输出头。

---

## 阶段 R11b：Encoder 加深与特征增强（主线① / **A-算法，R10 之后立刻做**）

> **执行位次**：R10 通过后 → **本阶段** → 再进 R11。目标是在结构侧把 raw 初值与有效 loss 拉向 match。

### 原理

直方图 + 浅 MLP 参数量过小（冠军约 2.3M），即使用满 batch，激活显存也上不去，初值上限低。主线①在 **已简化的头 + 最小训练底板**（已验证大 batch + sqrt LR；不必等完整 R11）上加强特征与深度，直接服务 raw 初值。

可借鉴：更深 residual MLP / 更宽 embedding、连续谱 1D CNN 旁路（AlphaDiffract 类）。  
不借鉴：peak-token Bert 主路、formula 条件。

### 方法（单因素）

| 臂 | 改动 |
|---|---|
| E0 | R10-slim + 最小训练底板（对照） |
| E1 | 加深/加宽峰 histogram 骨干（`histogram_num_blocks`、hidden 加宽）；**先不做谱** |
| E2 | E1 + 显式 Q/峰统计拼接（若尚未纳入） |
| E3 | E1/E2 + 连续谱 CNN fusion（可选；不过则关） |

约束：

- 训练路径 Encoder **必须向量化**（禁止 Python 逐样本 `.item()`）；
- 一次只加一档容量；用 dropout / weight decay 防过拟合；
- 长训：`max_epochs` 与 `patience` 要够细收敛（避免 ep15–30 早停砍掉缓降段）；
- 谱旁路必须 `fusion ≥ peak-only`。

### 与仓库内进行中实验的衔接

`configs/scale_100k_r8_deep_long.yaml`（加深 encoder/head、batch=2048、长训）属于本主线的 **E1 方向试点**；其 Gate 与下表对齐后，通过则并入 Phase A 默认骨干，失败则缩容量，不在浅网错误结构上继续堆宽。

### 通过 Gate

相对 E0 / R7-配方 baseline（valid1400 elem ≈15.36%，angle MAE ≈8.68°）：

- elem ≥ **17%**，或 angle MAE ≤ **7.5°**（任一明显进步即可）；
- CS-correct lattice 或非立方不同步变差；
- 显存与 samples/sec 可接受；无严重过拟合（train≫valid）。

### 失败处理

大模型不涨或过拟合 → 缩回容量；优先检查标签口径（规范胞）与头是否仍过重，**不**把主资源切去 Phase B 碰运气；也 **不**先用完整 R11 细扫掩盖结构问题。

---

## 阶段 R11：训练策略与峰对齐 loss（主线③ / **A-训练，算法定稿之后**）

> **执行位次**：R10 + R11b 通过后 → **本阶段** → 再扩数据 / 全量。在已定结构上把收敛与 match 导向目标函数抠到极致。

### 原理

SmoothL1(matrix6) 只要求六个标准化分量接近标签；indexing 真正要求预测晶胞产生的理论峰能解释观测峰。本阶段把 **loss 细调** 与 **显存打满的快速训练配方** 一起做对，在 **R10-slim + R11b 胜出 encoder** 上进一步压 loss、抬 raw / CS-correct lattice。

**KPI 重心**：`cs_correct_subset_lattice_elementwise`、angle/non-cubic、以及 **samples/sec / 墙钟**；CS CE 只作轻量辅助，**不加 SG / setting loss**。

### 方法

主 loss：

\[
\mathcal L=
\mathcal L_{\mathrm{SmoothL1}}
+\lambda_{\mathrm{peak}}\mathcal L_{\mathrm{peak}}
+0.1\mathcal L_{\mathrm{CS}}
\]

（无 setting 项；骨干为 **R10-slim + R11b 胜出 encoder**。）

峰一致性：

1. 预测 matrix6 → 实空间矩阵 \(A\)；
2. 计算倒易度量 \(G^*=A^{-1}A^{-T}\)；
3. 枚举低阶 hkl，得到理论 \(1/d^2\)；
4. 观测 2θ 按 λ 转成 \(1/d^2\)；
5. 使用单向 observed→theory soft-Chamfer，避免因系统消光惩罚未观测理论峰；
6. 使用 robust loss / temperature，防止最近邻切换造成梯度不稳定。

可选条件加权：CS 预测错误样本降低 lattice / peak 权重，或报告时强制拆分 CS-correct / CS-wrong 子集。

不以 peak loss 替换 SmoothL1，只作为辅助项。

### 训练配方（显存打满 / 快速训练）

在 **已定算法骨干** 上与 peak loss **可分开扫**（先配方、后 λ，或先固定小 λ 再打满吞吐）：

1. 抬 batch 至显存临界（必要时 gradient accumulation）；
2. AMP；稳定则试 `torch.compile`；
3. warmup + cosine（或等价）LR；扫 lr 时固定其他；
4. DataLoader workers / pin_memory / persistent_workers，保证 GPU 利用率；
5. 记录：peak reserved mem、samples/sec、step time、valid 指标墙钟。

目标：同结构下 **单位时间见过的有效样本更多、收敛到更低 valid loss**，而不是空转小 batch。

### 实验设置

#### R11-P0

- 数据：P0-700（规范胞标签 + R10-slim + R11b encoder）
- λ sweep：`{0.05, 0.15, 0.30}`（先做量纲归一化）
- hkl：从 `|h|,|k|,|l|≤3` 开始，再尝试 4
- 对照：同骨干、无 peak loss

#### R11-10k

- 只验证不崩；
- 不因 10k 排名决定最终 winner。

#### R11-100k

- 使用算法定稿骨干（R10-slim + R11b）；
- `best_metric=strict_raw_top1_elementwise_rate`，并监控 CS-correct lattice ew；
- 通过的单一 λ 用 seeds 42/43/44 复核。

### 通过 Gate

100k valid 满足：

- raw elementwise ≥ **同结构、最小底板的 R11b 出口 baseline**
- 且 angle MAE 至少下降 **0.5°**，或非立方 / **CS-correct lattice** ew 至少提升 **1pp**
- cubic elementwise 不低于 **85%**（或相对该 baseline 不显著回退）

再运行一次 MP100；目标是 raw / CS-correct lattice 稳定抬升，而不是一次大跳。

### 失败处理

若 P0 不过：优先检查峰 loss 数值尺度与梯度，不启动 100k。  
若 P0 过、100k 不涨：**记录并停用该 λ**；保留算法定稿骨干 + 已验证最小底板，进入 **扩数据阶梯**（仍不跳 Phase B 搜索）。  
（仓库内已有公平复核：`peak_consistency` λ=0.05 在旧 100k 上未优于 baseline，见实验记录；不因此否定主线③的其它项——大 batch+LR 缩放配方仍保留；但细扫应在新骨干上重做。）

---

## 阶段 R12：把单点回归改为真正可训练的多假设（Phase B）

### 原理

同一 PXRD 在低对称情况下可能对应多个近似晶胞；用一个点做 SmoothL1 会输出条件均值。已有 `lattice_norm_all` 失败，是因为那些头按晶系路由训练，并没有按“候选集合必须覆盖真解”训练。

**前提**：多假设架在 R10-slim 上。禁止在未做减法的 champion 头上再乘 K，以免专家数爆炸。

### 方法

在简化骨干上输出 K 个 hypothesis：

```
embedding → slim lattice 头 → K × matrix6
```

若 slim 仍含 CS 条件化，则在条件化后的特征上出 K，而不是 7 套 × K。

从 K=3 开始，不直接上大 K。

训练使用 Multiple Choice / best-of-K：

\[
\mathcal L_{\mathrm{MCL}}=\min_k
\left[
\mathcal L_{\mathrm{SmoothL1}}(L_k,L^*)
+\lambda_{\mathrm{peak}}\mathcal L_{\mathrm{peak}}(L_k)
\right]
\]

为避免所有头坍缩：

- 记录每个 hypothesis 的 winner 使用率；
- 加轻量 usage balancing；
- 只对明显重复的候选施加最小间隔 / 去重。

### 实验设置

1. P0-700：K={3,5}，检查是否至少一个 hypothesis 拟合真值。
2. 100k：只选择 P0 最稳定的 K。
3. 推理时直接取网络 K 个候选，再与 Bravais snap 合并。
4. 排序前先只看 oracle Top-K elementwise，隔离候选生成能力。

### 通过 Gate

valid1400：

- Top-K elementwise 相对 R10-slim canonical pool 提升 ≥ **3pp**
- 非立方 Top-K 提升 ≥ **2pp**
- raw / best-confidence Top-1 不明显下降
- cubic elementwise ≥ **85%**（或相对规范胞基线不显著回退）
- 所有 hypothesis 有实际 winner 使用，不发生单头独占

### 失败处理

- 模式坍缩：K 降为 3、加强 winner balancing。
- cubic 下降：降低 K 或仅对非立方启用多假设；**不**加回 setting 头。
- Top-K 不涨：停止堆 K，进入物理搜索；不复用 R5 旧 multi-anchor。

---

## 阶段 R13：从“单点局部精修”升级为峰驱动的多晶系搜索

### 原理

R6-B 失败的原因不是“搜索一定无效”，而是所有种子仍来自错误 NN 单点，局部 L-BFGS 无法跳出错误盆地。新搜索必须直接从观测 \(Q=1/d^2\) 构造独立种子，NN 只提供软先验。

这一步是最接近 McMaille 的核心增量。

### 方法

#### R13-A：先验证高对称搜索

按晶系自由度做约束搜索：

- cubic：1 个长度参数 × P/F/I setting
- tetragonal：a、c
- hexagonal：a、c

候选评价：

- indexed peak count
- mean / max |ΔQ|
- 未解释观测峰惩罚
- NN 体积偏差（软惩罚，不作唯一硬门）
- CS classifier 概率（软先验，不禁止低概率晶系）

搜索采用 coarse-to-fine grid 或轻量 Monte Carlo + 局部精修；不能只从 NN raw 点启动。

#### R13-B：扩展低对称

高对称 synthetic test 通过后，再依次做：

- orthorhombic
- monoclinic
- triclinic

低对称控制计算量：

- 先由观测 Q 差分 / 低阶峰构造 seed；
- 在 Bravais 流形内优化自由参数；
- 每晶系保留固定预算的局部最优；
- 合并后统一去重和体积护栏。

NN 的作用：

- 提供体积分布中心；
- 提供晶系搜索预算；
- 提供多假设 seed；
- **不允许**硬裁掉其他晶系。

### 实验设置

#### 单元与 synthetic gate

- 为每个晶系生成已知晶胞及 ideal peaks；
- 不使用训练标签初始化；
- 验证候选池能召回真胞；
- 测试轴置换、primitive/conventional、半胞/超胞。

#### valid1400 对照

- current single+guard
- q-search only
- NN prior + q-search

候选生成先保留 Top-100，排序后再截 Top-20；避免生成阶段过早丢真解。

### 通过 Gate

valid1400：

- Top-20 elementwise ≥ **30%**
- 非立方 Top-20 ≥ **10%**
- mapping-elementwise gap 不扩大
- 单样本搜索时间进入可接受预算

通过后再运行 MP100 milestone。

### 失败处理

- synthetic 都召回不了：视为搜索实现错误，不能归因模型。
- q-search pool 高、Top-1 低：进入 R14 排序，不回退局部 L-BFGS。
- q-search pool 仍低：扩展 seed 构造和参数边界，不先扩大 NN。

---

## 阶段 R14：把候选池变成真正的最终 Top-1

### 原理

Top-K 召回是 Top-1 的上限；只有当池里有真解，排序才有意义。R6-C 已证明体积相对项能修复半胞偏好，但候选更丰富后需要新的 peaks-only 排序器。

### 方法

#### R14-A：确定性峰拟合排序

统一候选特征：

- indexed peak count
- mean / median / max |ΔQ|
- unmatched observed peak penalty
- theoretical peak density / complexity penalty
- `|log(V/V_nn)|`
- CS prior
- Bravais setting prior

排序优先峰解释能力，体积和对称性作为正则项；禁止“体积越小越好”。

#### R14-B：可选 learned reranker

只有确定性 FOM 排序效率不够时才加入小型 MLP：

- 输入仅用上述 peaks-only 候选特征；
- 训练候选来自 train/valid，绝不使用 MP100 调参；
- 正样本：严格匹配或连续 lattice error 最小候选；
- 负样本：半胞、超胞、错 setting、峰拟合相近的 hard negatives；
- 使用 pairwise / listwise ranking loss；
- 与 deterministic FOM 做加权融合。

### 实验设置

先测 ranking efficiency：

\[
\text{efficiency}=
\frac{\text{ranked Top-1 elementwise}}
{\text{Top-K elementwise recall}}
\]

对同一候选池比较：

1. confidence
2. R6-C FOM
3. 新 deterministic score
4. learned reranker（如需要）

### 通过 Gate

- ranking efficiency ≥ **75%**
- valid ranked Top-1 elementwise ≥ **25%**
- 半胞/超胞错误率显著低于 legacy FOM
- MP100 Top-1 相对 13% baseline 获得明确提升
- **CS-correct lattice** 同步改善或至少不恶化

### 失败处理

- pool recall 高但排序低：增加 hard-negative 训练，不扩大模型主干。
- deterministic 与 learned 持平：保留简单确定性排序。
- ranking efficiency 已高但绝对 Top-1 低：瓶颈仍是 R12/R13 候选召回。

---

## 阶段 R15：峰条件闭环的迭代 proposal refiner

### 原理

借鉴 RealPXRD“条件在每一步参与更新”的思想，但不引入原子图或 atom diffusion。状态改为当前 lattice 候选及其峰残差：

\[
L_{t+1}=L_t+\Delta_\theta(
E_{\mathrm{PXRD}},
L_t,
R(L_t,\mathrm{PXRD})
)
\]

其中 \(R\) 是当前候选理论峰与观测峰的残差特征。

### 方法

- 初始候选来自 R12 多假设 + R13 搜索；
- 每个候选做 3–5 步 residual update；
- 每步重新计算理论峰 / ΔQ 特征；
- 共享同一个 refiner 权重；
- 输出保持在对应晶系流形上；
- loss 使用最终 lattice error + 每步 peak consistency；
- best-of-K 只要求某条轨迹命中，不强迫所有候选收敛到同一点。

先实现 residual MLP / FiLM，不直接上 diffusion/flow。

### 实验设置

- 对照：0 步、1 步、3 步、5 步；
- valid1400 先测；
- 记录每一步 peak score 与 lattice error 是否同向下降；
- 限制推理时间。

### 通过 Gate

- Top-K 或 ranked Top-1 ≥ **+2pp**
- 低对称提升不以 cubic 大幅下降为代价
- 多步优于一步，且无发散

### 失败处理

若 peak score 下降但 elementwise 变差，说明峰目标仍有等价胞/系统消光歧义；回到 R14 加强体积与 Bravais 判别，不继续加深 refiner；**不**加回网络 setting 头。

---

## 阶段 R16：连续谱 CNN 特征增强（主线① 可选补做）

> 若 R11b-E3 已做过谱消融，本阶段跳过。仅当 Phase A 峰骨干已稳、尚未测谱旁路时补做。

### 原理

Indexing 的定义变量仍是离散 \(Q\) / 峰位；连续谱是 smeared 观测，适合增强鲁棒特征，不宜替换峰主路。AlphaDiffract 式 1D ConvNeXt 可借鉴为 **旁路**，必须用消融证明增益。

### 方法

```
峰表 → Histogram / Q encoder  ─┐
                                ├→ concat / gated fusion → 既有 CS + lattice 头
连续谱 → 1D CNN / ConvNeXt   ─┘
```

约束：

1. **峰分支始终存在**；禁止“只喂谱、不喂峰”作为主模型。
2. 连续谱来源二选一写清：由峰表重建 profile，或使用仿真/实验 `.xy`；训练与评测一致。
3. 先在 100k 做 fusion vs 纯峰对照；**fusion 未优于纯峰则永久关闭谱分支**。
4. **不加 SG 头**；谱分支只服务 embedding / lattice。
5. 若全量训练，仍须先过 100k 消融 Gate。

### 实验设置

| 臂 | 输入 |
|---|---|
| Peak-only（对照） | 当前最佳峰 encoder |
| Spectrum-only（诊断） | 仅 CNN；预期弱于峰，用于确认旁路定位 |
| Fusion | 峰 + 谱 |

报告 raw / Top-K / ranked Top-1 / CS-correct lattice；噪声鲁棒可另做扰动测试。

### 通过 Gate

- Fusion 相对 Peak-only：valid ranked Top-1 或 CS-correct lattice ≥ **+1pp**，且不伤 cubic；
- Spectrum-only 明显弱于 Peak-only（符合预期）；
- 参数量与延迟可接受。

### 失败处理

未过 Gate → 关闭谱分支，主线仍为峰 + 搜索；不把谱 CNN 做成必选项。

---

## 阶段 R17：数据规模与训练预算（可达全量 ~600 万）

### 原理

数据不是瓶颈；扩大数据只能放大正确范式。只有候选生成和排序均过 Gate 后，才值得从 100k 扩到 500k → 1M → **全量约 600 万**。

### 方法

- 规范胞标签在所用子集 / 全量上预先算好；
- 骨干固定为 **R10-slim**（或其上通过的 R11/R12 变体）；
- 按晶系和难度分层，不做简单的低对称重复采样；
- 重点加入搜索产生的 hard cases / hard negatives；
- 使用 R11 peak-aligned loss + R12 multi-hypothesis（及 R16 若已通过）；
- 推理使用 R13–R15 完整链路；
- 三个 seed 复核最终阶梯，不在早期 sweep 上浪费全量算力。

### 实验设置

规模阶梯：

1. 100k 算法对照
2. 500k
3. 1M
4. 全量 ~600 万（仅当阶梯仍持续增长）

每级保持模型结构不变，画 scaling curve。

### 通过 Gate

- 500k 相对 100k ranked Top-1 ≥ **+3pp**
- 非立方 / CS-correct lattice 持续增长
- 训练/推理成本可接受

不满足则停止扩规模，回到搜索与排序；**不以“还有更多数据没用”替代算法返工**。

---

## 6. 阶段目标与现实预期

这些是决策 Gate，不是承诺结果：

### Phase A（主攻：先算法抬初值，再训练/数据拉满）

| 阶段 | 段落 | 建议目标 |
|---|---|---:|
| R7 | 底座 | 固化评测 / baseline JSON |
| R8 | 底座 | P0 train elem ≥80% |
| R9 | **A-算法** | 规范胞新基线 |
| R10 | **A-算法 / ② 头** | 删 setting；FiLM/共享加深 → `R10-slim` |
| **R11b** | **A-算法 / ① Encoder** | 加深特征；elem≥17% 或 angle≤7.5° |
| **R11** | **A-训练 / ③** | 配方与 peakλ 细扫；同结构再涨 |
| 扩数据 → R17 | **A-训练** | 放大已验证算法；中期向 30–40% |
| （最小底板） | 贯穿算法段 | 已验证：`bs256 + lr2e-3`（或等比缩放） |

### Phase B（后置：初值与训练放量过关后再做）

| 阶段 | 主要能力 | 建议目标 |
|---|---|---:|
| R12 | 多假设 | valid pool 比 baseline +3pp |
| R13 | 峰驱动搜索 | valid Top-20 ≥30% |
| R14 | Top-1 排序 | valid final Top-1 ≥25% |
| R15 | 迭代闭环 | final Top-1 再 +2pp |
| R16 | 谱旁路补做 | fusion ≥ 纯峰，否则关闭 |

合理路线：

```
13% raw（浅网 + 复杂头 + 小 batch）
  → 规范胞 + 简化且加深的头（算法）
  → Encoder 加强特征 → raw / 角误差 / 有效 loss 明显贴近 match（算法出口）
  → 训练策略与数据量拉满 → 指标再拔高（训练出口）
  →（可选）多假设 / 搜索 / 排序补召回（Phase B）
```

---

## 7. 推荐执行顺序

### Phase A（严格串行）

1. **R7–R8**：评测与可学习性底座。
2. **R9**：规范胞。
3. **R10（A-算法 / 主线②）**：输出头减法 + 加深 → `R10-slim`。
4. **R11b（A-算法 / 主线①）**：Encoder 加深 / 特征增强（含可选谱 fusion）→ 算法出口。
5. **R11（A-训练 / 主线③）**：完整训练策略与 peakλ 细扫 → 同结构再涨。
6. **扩数据 / R17**：放大已验证算法；未过算法 Gate 不上全量。

**Phase A-算法出口**：结构与目标已定，raw / 角误差 / 有效 loss 相对旧浅网+复杂头有稳定提升。  
**Phase A-训练出口**：同结构下再通过配方与数据量把指标拉到可复现极致。

### Phase B（训练放量过关后）

7. R12 多假设 → R13 搜索 → R14 排序 → R15 refine（R16 仅补谱）。

### 已执行实验与方案编号对照（避免混淆）

仓库实验记录里的编号曾按「训练/容量」叙事使用，与本方案阶段号不完全同一套：

| 实验记录 | 对应本方案 |
|---|---|
| `20260714-R7-训练策略修复与峰一致性复核` | 主线③ 的一部分（配方 + peak 公平复核） |
| `20260714-R8-容量与长训` / `scale_100k_r8_deep_long` | 主线①（及头加深）试点 → 归入 **R11b** |
| 方案中的 R7/R8 | 仍表示评测锁定与 P0 过拟合底座 |

后续文档以 **本方案阶段号 + 主线①②③** 为准；实验记录标题可保留历史名，正文注明映射。

可并行：

- 算法段内不同容量探针（不同 run）；训练段吞吐打磨可与数据准备并行；
- Phase B 的搜索 **原型代码** 可与 Phase A 并行开发，但 **不上主资源调参**。

不能提前做：

- 未完成头简化 / Encoder 定稿就上完整 R11 细扫或全量；
- 未完成算法 Gate 就主攻搜索/reranker；
- 加回 setting 头；SG 多任务；formula 换分。

---

## 8. 第一批立即可执行实验

### E0 — 评测 / 最小训练底板

- 固化 strict profile；
- 算法阶段默认沿用已验证的 **大 batch + sqrt LR 缩放**（如 bs256/lr2e-3）；
- peak_consistency 默认关闭；完整 λ 细扫留给 R11。

### E1 — 规范胞（R9）

- Niggli（或书面约定）+ P0/100k；换标签后重拉 baseline。

### E2 — 输出头减法并加深（R10 / A-算法）

- H0→H1 去 setting → H2/H3；**已冻结 `R10-slim` = H3 FiLM**（3500/Niggli 探路裁决，2026-07-14）：`head_type=film`，无 setting，`head_num_layers=4`，`hidden_dim=512`。

### E3 — Encoder 加深（R11b / A-算法）

- 在 slim 头 + 最小底板上跑加深骨干（含进行中的 `r8_deep_long` 类配置）；
- 过 Gate 再考虑更大容量或谱 fusion。

### E4 — 训练策略与放量（R11 → 扩数据）

- 在算法定稿骨干上细扫 schedule / peakλ / 显存打满；
- Gate 过后再扩数据直至全量。

### E5 — Phase B（其后）

- 多假设 / 搜索 / 排序；仅当 A-算法 + A-训练出口已过。

---

## 9. 最终裁决标准

每一阶段只回答一个问题：

| 问题 | 对应指标 |
|---|---|
| 网络能否学习？ | P0 train elementwise |
| 标签口径是否干净？ | 规范胞 round-trip + R9 基线 |
| 头是否够简单且深？ | R10-slim：无 setting；容量在共享/FiLM |
| Encoder 是否更强？ | R11b vs 浅网：raw / 角 / CS-correct lattice |
| 训练是否拉满？ | 显存/util、samples/sec、墙钟、valid loss（R11） |
| 是否值得扩全量？ | A-算法 + A-训练已过 Gate + scaling curve |
| 搜索是否值得？ | Phase B：pool→Top-1 是否真涨 |
| raw 初值是否更好？ | raw Top-1 + **CS-correct lattice** + angle/non-cubic |
| （Phase B）候选是否覆盖真解？ | Top-K elementwise |
| （Phase B）排序能否选中真解？ | ranked Top-1 / efficiency |
| 是否值得扩全量？ | Phase A 已过 Gate + scaling curve |

一次“有效进步”必须同时满足：

1. peaks-only，无 formula / atom 泄漏；
2. valid 上按预注册 Gate 通过；
3. MP100 严格 0.05/3° elementwise 提升（相对 **对应新基线**）；
4. map-ew gap 不靠伪命中扩大；
5. CS-correct lattice 或至少一个低对称指标同步改善；
6. 有完整 config、checkpoint、JSON 和实验记录。

任何只提升 loose mapping、只提升 oracle Top-K、只提升 cubic、或只提升 **SG/CS 分类而不改善 lattice** 的结果，都不能视为最终 Top-1 攻关成功。

整个方案的核心是：**先把算法初值做准做深（规范胞 + 瘦头 + 强 Encoder），把有效 loss 拉向 match；再把训练策略与数据量拉满；最后才用多假设/搜索补召回。**
