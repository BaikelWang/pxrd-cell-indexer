# CellIndexing 后续优化方案 v4（2026-07-20）

> 前序：v3（`20260715-CellIndexing-可执行优化方案v3.md`）的 Phase A（A0→A1→A2→A2.5→A3）已全部完成并锁定。  
> 本文承接 v3，基于当前指标结构做**理论诊断**，重排后续优化优先级，给出可执行阶段、配置、Gate 与依赖。  
> 主指标：valid1400 **strict raw Top-1 elementwise**（`ltol=0.05` / `atol=3°`），Niggli 标签。  
> 北极星：MP100 strict（对标 JADE ~68% / McMaille ~66%）。

---

## 1. 现状（已锁定的生产栈）

**Peak Transformer T48-geom + `cs_conditional` L=6 + `gstar6` + max_peaks=48**

| 口径 | 数值 |
|------|------|
| valid1400 strict（三 seed mean） | **43.07%**（42.5 / 43.7 / 43.0） |
| loose（0.3 / 10°） | ~78% |
| angMAE / lenMAE | ~6.6° / ~0.54 |
| 分晶系 strict | cubic 95 / hex 57 / trig 53 / tet 50 / ortho 22 / mono 18 / tric 5 |

生产配置：`configs/scale_100k_a3_g1_gstar6.yaml`。已关闭方向（不再碰）：hist 加深、peak⊕hist fusion、MCL、shared head、rel-attn、盲目加峰数、decoded-cell(G2)、严口径 FOM/多锚旧池。

---

## 2. 理论诊断：瓶颈在哪

### 2.1 strict 是「合取（AND）」指标

某晶系 strict 要求其**全部自由参数同时过线**。设每自由参数独立过线概率 \(p\)、自由度 \(k\)：

\[
\text{strict}_{CS}\approx p^{\,k}
\]

用实测 per-param 通过率（`six_param_pass_rates`≈0.6–0.71）反推：

| CS | 自由度 k | 预测 \(p^k\) | 实测 strict |
|----|:--:|:--:|:--:|
| cubic | 1 | ~0.95 | 0.95 |
| tetragonal | 2 | 0.71²≈0.50 | 0.50 |
| orthorhombic | 3 | 0.6³≈0.22 | 0.22 |
| monoclinic | 4 | 0.6⁴≈0.13 | 0.18 |
| triclinic | 6 | 0.6⁶≈0.047 | 0.045 |

**高度吻合**。→ 低对称里六参数近似独立，合取以几何级数吞精度。

### 2.2 三条推论

1. **架构改不动指数 \(k\)，只能微调底数 \(p\)**；而 \(p\) 已从 33→40→43 挖到边际递减。继续堆 encoder/head 收益 <1pp。
2. **loose 78% vs strict 43%（差 35pp）**：识别（进正确盆地）已基本解决，剩下的是**精度 + 合取**。
3. **per-param 小改进在低对称上超线性放大**：\(p\) 0.6→0.7 时，ortho +12pp、mono +11pp、tric +7pp。

### 2.3 三根优化杠杆（正交）

| 杠杆 | 机理 | 手段 | 天花板 |
|------|------|------|--------|
| **A. 抬 per-param \(p\)** | 整条 \(p^k\) 曲线上移 | metric 对齐 loss、放量、更强峰特征 | 中（低对称超线性） |
| **B. 绕过合取** | propose-and-verify，取多候选 max | 搜索 + 物理验证/精修 + rerank | **高**（上限≈loose 78%） |
| **C. 鲁棒性** | sim→real 不变性 | 扰动课程 | 与 clean 正交（保产品可用） |

---

## 3. 阶段总览与排序

> **S0 已完成（2026-07-20）**：详见 `docs/实验记录/20260720-S0诊断-A3winner三探针.md`。  
> 关键修订：现有邻域搜索仅 +3.1pp；MP100 strict 23%（vs JADE 68%）。**A4 提前；S2 须重做候选生成。**
>
> **S1 已完成（2026-07-20）**：`soft_strict`（A5）P0-700 三组超参（τ∈{0.5,2.0}、λ∈{0.02,0.1}）均**未过 95% Gate**（最高 56.0%），直接淘汰，100k 跳过。详见 §5.4。**metric 对齐 loss 这条杠杆基本走到头，主线收敛到 S3(A4) + S2。**
>
> **S3(A4) 已完成（2026-07-20）**：合成 Gate 三 seed **全过**（V-clean +0.05pp / V-mixed +6.74pp），但 **MP100 复评无实质收益**（三 seed mean 20.0%→20.7%，+0.7pp）。详见 §7.6。**生产仍用 A3-G1；合成扰动分布与真实谱误差不匹配，A4  alone 吃不掉 −45pp JADE 缺口。**

```text
S0 诊断 ✓
   │
   ├─ S1 A5 metric-aligned loss ✗ 已淘汰（P0 best 56.0% < 95%，见 §5.4）
   │
   ├─ S3 A4 鲁棒课程 △ 合成 Gate 过 / MP100 无收益（见 §7.6）——生产仍用 A3-G1
   │
   ├─ S2 Phase B（仅独立 q-search 原型达标后）——旧 Bravais/manifold refine 栈否决
   │
   └─ S4 A6 放量（算法冻结后）
```

排序原则：**诊断校准后 → loss 首付（已试，未过）→ 鲁棒课程（已过，待 MP100 验证真实收益）→ 搜索仅在新候选源证明有头寸后投入 → 最后放量。**

---

## 4. S0 · 诊断层（零训练，先做）

**目的**：在写任何新训练代码前，用现有 ckpt 量出各杠杆的真实头寸，避免盲目投入。

### S0.1 MP100 严口径对标
- 脚本：`scripts/eval_mp100.py` / `scripts/a0_protocol_eval.py`
- 产出：A3-winner 在 MP100 的 strict（oracle-CS 与 predicted-CS 两路），与 JADE/Mc 缺口表。
- 意义：判断「clean 精度不足」还是「sim→real 崩盘」谁是主矛盾 → 决定 S1/S2 与 S3 的相对紧迫度。

### S0.2 refine-oracle 探针（B 的上限测量）★
- 脚本：扩展 `scripts/diag_search_topk.py` / `scripts/diag_strict_topk_fom.py`。
- 做法：对预测胞做**峰匹配局部精修/邻域搜索**，测「盆地→strict」可回收比例（当前 43% 能回收到 ≤78% 的哪一段），按晶系拆分。
- 判据：若可回收头寸大（如 ortho/mono 能从 22/18 拉到 40%+），**S2 立即优先**。

### S0.3 per-param 失配分解
- 脚本：复用 `scripts/diagnose_raw_errors.py` / `decompose_joint.py`。
- 做法：对每个自由参数统计过线率、误差分布、以及「仅差 1 维」的近失配样本占比（按晶系）。
- 意义：量化 S1（worst-param loss）的直接靶标人群。

**S0 Gate**：产出三张表（MP100 缺口 / refine 头寸 / per-param 分解），据此确认 S1、S2 力度与 S3 排序。**不通过就不进 S1/S2。**

### S0 实测摘要（已完成）

| 项 | 数值 |
|----|------|
| MP100 raw strict | **23%**（JADE 68.1% / Mc 65.9%） |
| valid1400 → MP100 | 42.5% → 23%（**−19.5pp**） |
| pool recall（Bravais / manifold） | 42.5% → **45.6% / 44.9%**（仅 +3pp） |
| FOM Top-1 | **有害**（40% / 35% < raw） |
| loose_elem − strict | **+21.9pp** 理论精度头寸 |
| 失败中 near-1-dim | **17.8%** |

---

## 5. S1 · A5 metric-aligned soft-strict loss（抬 p 的便宜首付）

### 5.1 原理
SmoothL1 优化**均值误差**，但 strict 要**最差维过线**。对 decoded lattice 构造归一化误差并取软最大：

\[
e_i=\Big[\tfrac{|\Delta a|}{0.05a},\dots,\tfrac{|\Delta\alpha|}{3^\circ},\dots\Big],\quad
\mathcal L_{\text{soft}}=\tau\log\sum_i e^{e_i/\tau}
\]

- 与已淘汰的 **G2 本质不同**：G2 是 decoded 空间**普通 SmoothL1（优化均值）**；A5 直接对齐 0.05/3° 边界、惩罚**最差维**，专打「5 维好 1 维差」的近失配。
- 只在自由参数维计算（复用 `CS_PHYS_PARAM_MASK`），避免在固定 90°/等长上刷 0。

### 5.2 实现
- `losses.py` 新增 `mode="soft_strict"`：`loss = SmoothL1(gstar6_norm) + λ·L_soft(decoded, mask)`。
- `trainer.py` 记录 `L_soft` 与梯度尺度；单测：finite-grad + 「最差维」惩罚方向正确。
- 超参：`τ∈{0.3,0.5}`、`λ∈{0.1,0.3}` 小网格。

### 5.3 流程与 Gate
- P0-700：strict ≥95%（确认不破坏 overfit，G2 的教训）。
- 100k seed42：Gate = strict ≥ **+1pp**（≥44%），且低对称（ortho/mono/tric macro）不回退。
- 通过 → seed43/44 复核；未过 → 记录淘汰，直接进 S2。
- **预期**：+0–2pp，低对称占主要贡献。低成本，无论成败都为 S2 减负。

### 5.4 实测结果（已完成，2026-07-20）★ **未过 P0 Gate，已淘汰**

`soft_strict_loss`（log-sum-exp 最差自由维，归一化到 0.05/3° 边界）已实现并通过单测（finite-grad / 最差维方向 / masked 维不干扰）。P0-700 三组超参网格结果：

| 配置 | τ | λ (physical_weight) | P0-700 best strict（1200 ep 内） | 备注 |
|------|:--:|:--:|:--:|------|
| 原始网格点 | 0.5 | 0.1 | **23.1%**（ep1180） | 与纯 SmoothL1(gstar6) 基线（96%@ep935）相比严重破坏 overfit |
| 降权 | 0.5 | 0.02 | **55.7%**（ep1187） | 明显改善但仍远未达 95%；ep1000 后基本收敛平台 |
| 软化 τ | 2.0 | 0.1 | **56.0%**（ep1167） | 同上，与降权结果几乎一致的天花板 |

- 三组梯度诊断（checkpoint 手动 backward）显示 `loss_reg` 与 `loss_phys` **不对抗**（cosine≈+0.29，非负），排除了"梯度互相抵消"的实现 bug；三组 P0 末期 per-dim pass rate 已到 74–85%，但 700 样本 × 6 维 **AND** 合取仍卡在 ~56%，说明瓶颈是"worst-dim 优化目标在小样本满秩 overfit 下天然更难同时压满全部维"，而非某个具体超参没调对。
- 与已淘汰的 **G2**（`decoded_cell`，λ=0.05，SmoothL1 均值，best 78.6%）对比：A5 的最差维聚焦目标比 G2 的均值目标**更难通过纯 overfit 检验**（56% < 78.6%），佐证"惩罚最差维"这一目标函数形态本身在 decode 空间比均值损失更陡峭、更不利于 700 样本满秩收敛。
- **决策**：按 §5.3 Gate 规则（P0 <95% → 直接淘汰，不进 100k），**A5 soft_strict 淘汰**，100k seed42 跳过。三份 P0 日志/配置保留供复盘：`configs/overfit700_a5_s1_soft_strict{,_lam002,_tau2}.yaml`。
- **含义**：结合 G2 的先验证据，**decode 空间的辅助 loss（无论均值还是最差维形式）在当前架构/数据规模下都会显著拖慢甚至破坏 P0 overfit**，"metric 对齐 loss" 这条杠杆（表 2.3 中的 A. 抬 per-param p）在当前 SmoothL1(gstar6) 主干上基本走到头；后续应转向 **B（绕过合取，propose-and-verify）** 与 **C（鲁棒性）**，即优先推进 S3(A4) 与谨慎评估 S2 的独立候选源，而不再在 loss 形式上做更多尝试（S5b 支线同理下调优先级）。

---

## 6. S2 · Phase B propose-and-verify（最高天花板）★

把「一次性 6D 回归」换成「**多候选生成 + 物理验证 + 排序**」，用 78% 的识别兑换 strict。分三步、严格串行。

### 6.1 B1 · 候选生成与召回
- 模块：新建 `src/pxrd_cell_indexing/search/`（**不塞进训练 forward**）。
- **S0 否决**：仅「回归胞邻域 + Bravais/manifold refine」pool 只 +3pp，**不能作为 B1 主体**。
- 来源（必须含独立源）：
  1. **主**：独立 q-search（低角峰 `1/d²` 线性组合枚举，McMaille 式；**不依赖** NN 初值；先 synthetic）；
  2. 辅：回归胞 + 小邻域 / Bravais 变体（仅作并集补漏）；
  3. 禁止把现有启发式 FOM 当生产选模（S0 显示严口径下有害）。
- 指标：**oracle Top-K recall**，按晶系与 K∈{5,10,20} 报。
- Gate：独立源 pool strict-recall 相对单点 43% ≥ **+15pp**（S0 现栈未达标；不达标则 S2 停，资源回 A4/A5）。

### 6.2 B2 · 候选排序（rerank）
- FOM/reranker：峰匹配一致性（obs `1/d²` vs 候选理论线）+ 体积/对称先验；复用 `sweep_fom_rerank.py` 调参。
- 可选：轻量学习式 reranker（输入候选特征，输出打分），但**候选 schema 固定后**再做。
- Gate：ranked Top-1 strict 相对 43% ≥ **+3pp**（真正把 recall 转成 Top-1）。

### 6.3 B3 · 局部精修（可选）
- 对 Top-K 做 1–3 步峰匹配 residual 精修（reciprocal metric 空间）。
- Gate：Top-1 再 ≥+2pp，或 Top-20 recall 不变而 angMAE 显著降。
- 护栏：若 peak score 改善但 strict 变差（偏好伪胞/等价胞），停 refiner。

**S2 交付**：`search/` 模块 + 候选 schema（记录来源/CS/体积/peak score/去重键）+ 三步 Gate 报告 + MP100 复评。

### 6.4 B1-S0 实测结果（已完成，2026-07-20；2026-07-20 向量化重构后复测）★ **4/7 晶系已过 Gate**

实现：`src/pxrd_cell_indexing/search/qsearch.py`——从观测 `1/d²` 出发，给低角峰枚举小整数 hkl，用「`1/d²=hᵀG*h` 对 G* 六分量线性」这一恒等式精确解各晶系自由度量参数（不依赖 NN、不做梯度下降），再用全峰一致性校验（de Wolff 式匹配）过滤错误 hkl 分配。`scripts/run_b1_s0_synthetic.py` 做 v3 §11.3 B1-S0 synthetic 单元测试（每晶系 15-20 个随机已知胞 + ideal 峰，无 NN 参与）。

第一轮朴素 Python 循环实现吞吐不足，把大量真解挤到了时间预算之外（假阴性）。重构为批量线性代数 + 两级向量化预筛（求解/SPD/一致性校验全链路向量化，吞吐提升 2-3 个数量级）+ 修复单斜晶系轴约定 bug（`monoclinic_a/b/c` 三变体分派，见 [B1-S0 实验报告 §6](../实验记录/20260720-B1-S0独立q-search原型.md#6-第二轮向量化重构把实现细节和方法学解耦)）后复测：

| 晶系 | 第一轮 recall | **第二轮 recall**（n=20） | Gate(95%) |
|---|:--:|:--:|:--:|
| cubic | 100% | **100%** | **PASS** |
| tetragonal | 73% | **100%** | **PASS** |
| hexagonal | 80% | **100%** | **PASS** |
| trigonal | (未测) | **100%** | **PASS** |
| orthorhombic | 27% | **80%** | FAIL |
| monoclinic | 40–47% | **80%** | FAIL |
| triclinic | 0% | **0%** | FAIL |

**结论（更新）**：第一轮"未过 Gate"里有相当一部分是实现效率问题而非方法学问题——重构后**高对称晶系（cubic/tet/hex/trig，覆盖多数真实样本）已稳定 100% 过 95% synthetic Gate，可以推进到 B1-S1（valid1400 小规模验证）**。orthorhombic/monoclinic 大幅改善（27%→80%、40-47%→80%）但仍受"联合 k 元线性求解的组合数按 `M^k` 增长"这一硬约束限制，尚未过 Gate；triclinic 仍 0%（预期内）。真正的下一步修复是把"联合求解 k 个自由度"改成"顺序求解"（先用单轴反射独立解 3 个对角分量，再逐个用 1 个额外峰解非对角项，组合数从 `M^k` 降到 `M_axis^3+M_dense`）——这对 ortho/mono/tric 是同一套框架，比继续加预算/调参的性价比更高。完整报告：[B1-S0 实验报告](../实验记录/20260720-B1-S0独立q-search原型.md)。

### 6.x B1-S1：valid1400 高对称子集真实数据验证（已完成，2026-07-21）★ **Gate 通过**

把 4 个已过 synthetic Gate 的晶系放到 valid1400 真实数据（各 40 样本），对比独立 q-search（oracle CS 路由）vs 现有 NN 邻域搜索（`build_top_k_candidates`）：

| 晶系 | q_search@20 | nn_pool@20 | Δ | merged@100 |
|---|:--:|:--:|:--:|:--:|
| cubic | 95.0% | 95.0% | +0pp | 97.5% |
| tetragonal | 55.0% | 45.0% | +10pp | 67.5% |
| hexagonal | 82.5% | 40.0% | +42.5pp | 85.0% |
| trigonal | 25.0% | 35.0% | **-10pp** | 47.5% |
| **整体** | **64.4%** | 53.75% | **+10.6pp** | 74.4% |
| **non-cubic** | **54.2%** | 40.0% | **+14.2pp** | 66.7% |

v3 §11.4 Gate（Top-20≥30%、non-cubic≥15%、+8pp vs NN 本地池）**聚合层面全部通过**。过程中发现并修复一个真实数据 bug：valid1400 标签是 **Niggli 原胞**参数，F/I 心立方原胞是菱方形（α=β=γ=60°/109.47°）而非传统 90/90/90，导致 cubic 搜索基组假设错误（修复前 17.5%→修复后 95%，加了 `cubic_p/f/i` 三变体，同 monoclinic 的处理方式）。唯一不达标的是 trigonal（个体 -10pp），怀疑是同类"六方轴 vs 菱方原胞"双设定问题，留作后续修复项，不影响整体推进决策。完整报告：[B1-S1 实验报告](../实验记录/20260721-B1-S1-valid1400高对称验证.md)。

**下一步**：高对称线推进到 **B2（候选排序，先用确定性 score，不训练 learned reranker）**；trigonal 顺手加 `trigonal_hex/rhomb` 变体修复；中低对称继续顺序求解重构，两条线并行。

### 6.y B2：固定 B1 pool 候选排序对比（已完成，2026-07-21）★ **Gate 通过，无需 learned reranker**

在 B1-S1 同一套固定候选池（cubic/tet/hex/trig 各 40 样本，pool_budget=20）上对比三种排序：NN proximity（只信 NN 初值）、legacy R6-C FOM（`model/fom.rerank_candidates_by_fom`，配 NN 体积先验）、新写确定性 score（`search/rank.py`，覆盖 v3 §12.1 全部特征）：

| | ranked Top-1（整体） | ranking efficiency（整体） | Gate(≥25%/≥75%) |
|---|:--:|:--:|:--:|
| nn_proximity | 46.9% | 72.8% | efficiency 未过 |
| **legacy_fom**（+NN 体积先验） | **57.5%** | **89.3%** | **PASS，最优** |
| deterministic（新） | 55.6% | 86.4% | PASS |

**结论：不需要新排序算法，也不需要 learned reranker——现有 legacy FOM 接上 NN 体积先验（`ref_volume`，`FomRerankConfig` 已有字段，此前只用在 NN-seeded 候选池，没接到 q-search 池上）就已经通过 Gate 且优于新写的确定性 score。** 关键坑：忘记传 `ref_volume` 时 legacy FOM 会退化成"偏好小体积"，cubic Top-1 从 92.5% 掉到 80%——排序对比必须保证三种方法拿到同等的 NN 先验信息，否则不是公平比较。完整报告：[B2 实验报告](../实验记录/20260721-B2-候选排序对比.md)。

**下一步**：把 B1（独立 q-search）+ B2（legacy FOM + NN 体积先验）接成完整流水线，评估是否进入 B3（可选迭代 refiner）或直接推进真实场景/放量评估；trigonal 修复与中低对称顺序求解重构继续并行推进。

### 6.z B3：迭代局部精修（已完成，2026-07-21）★ **Gate 未过——干净的空结果**

在 B1+B2 流水线（固定 pool + legacy FOM 排序）的 ranked Top-5 上加 0/1/3 步最小二乘局部精修（`search/refine.py`：用全部匹配峰重新解 G*，精确线性解，非梯度下降）：

| | step=0 ranked Top-1 | step=3 | angle MAE |
|---|:--:|:--:|:--:|
| 整体（n=160） | 57.5% | 58.1%（+0.6pp，未过 +2pp Gate） | 5.024°→5.024°（**完全无变化**） |

**原因**：valid1400 是无噪声模拟数据（2θ 直接来自 pymatgen 确定性计算），q-search 的初始精确解本身就在浮点精度内是精确解，没有残差可供最小二乘收紧——精修前后候选参数逐 bit 相同。微小的 +0.6pp 波动是 Top-5 中其他候选匹配数漂移导致的重排序噪声，不是精修本身的收益。**决策：不采用 B3，生产流水线维持 B1+B2；`search/refine.py` 代码保留，留给未来引入真实谱噪声（零点漂移/峰位抖动等）场景后再复测。** 完整报告：[B3 实验报告](../实验记录/20260721-B3-迭代局部精修.md)。

**下一步**：B1（q-search 高对称）+B2（FOM 排序）已经是一条经过 Gate 验证、可以直接使用的候选生成流水线；trigonal 原胞设定 bug 修复、中低对称顺序求解重构可以并行推进；若要继续挖 B3 的价值，需要先有真实/含噪声的观测谱（而不是当前无噪声模拟）。

---

## 7. S3 · A4 Clean→Perturbed 鲁棒课程（产品轴，可与 S2 并行）

沿用 v3 §8。要点：

- 增强拆成可审计参数：global zero-shift、per-peak jitter、peak dropout、impurity、intensity noise、preferred orientation、clean 概率。
- 固定冻结 robust-valid：V-clean / V-zero / V-jitter / V-drop / V-impurity / V-mixed（离线生成，永久固定 seed）。
- 课程：C0 clean-only → **C2 clean 训满后 80/20 perturb 微调（LR×0.1，10–20 ep）**；C1 全程 perturb 仅作对照。
- Gate：V-clean 回退 ≤0.5pp 且 V-mixed ≥+3pp，各单项无崩盘、低对称不恶化。
- 说明：A4 **不抬 clean 43%**，但决定 MP100/真实谱可用性；若 S0.1 显示 sim→real 崩盘，则 A4 提前到与 S1 并列。

### 7.6 实测结果（已完成，2026-07-20）★ **三 seed 全过 Gate**

实现：`src/pxrd_cell_indexing/data/robust_perturb.py`（拆分出 6 个可审计扰动组件）+ `scripts/build_robust_valid.py`（冻结 V-zero/jitter/drop/impurity/mixed，各 1400 条，固定 seed）+ `scripts/eval_robust_valid.py`。C0 直接复用已有 A3-G1 三 seed checkpoint；C2 用 `model.warm_start_checkpoint` 从 A3-G1 best.pt 热启（153/153 参数 100% 命中），`augment_mode=robust`、`clean_probability=0.8`、LR×0.1、15 epoch 微调。C1 未跑（非阻塞，见下）。

三 seed 均值：

| 验证集 | C0 | C2 | Δ | Gate |
|---|:--:|:--:|:--:|:--:|
| V-clean | 43.07% | 43.12% | **+0.05pp** | ≤0.5pp 回退 → **PASS** |
| V-zero | 42.55% | 42.86% | +0.31pp | 无崩盘 → PASS |
| V-jitter | 42.86% | 42.60% | −0.26pp | 无崩盘 → PASS |
| V-drop | 24.60% | 28.86% | +4.26pp | — |
| V-impurity | 31.21% | 34.64% | +3.43pp | — |
| **V-mixed** | 19.38% | 26.12% | **+6.74pp** | ≥+3pp → **PASS**（超阈值 1.2×） |
| 非立方晶系宏观（clean/mixed） | 34.50%/16.58% | 34.61%/19.39% | +0.11pp/+2.81pp | 不恶化 → PASS |

三个 seed（42/43/44）独立复现同一模式，排除单 seed 噪声。完整报告：[S3 实验报告](../实验记录/20260720-S3-A4鲁棒课程.md)。

**决策（含 MP100 复评）**：
- 合成 Gate：**PASS**（V-clean +0.05pp / V-mixed +6.74pp）。
- MP100 严口径（niggli / 0.05/3° / raw）：C0 mean **20.0%** → C2 mean **20.7%**（+0.7pp；seed42 23%→22%，seed43 18%→20%，seed44 19%→20%）。逐样本翻转净变 ±1–2，落在 n=100 噪声内。
- **生产仍锁定 A3-G1**；A4-C2 不替换生产 checkpoint。
- 含义：当前合成扰动（zero/jitter/drop/impurity）与 MP100 真实误差分布**系统不匹配**——在合成 V-mixed 上学到的不变性转不到真实谱。若继续 A4，须用真实谱误差统计重标定扰动，或直接真实噪声微调；否则主线应转向更高天花板的 S2（独立候选生成）/ 低对称精度。完整报告：[S3 实验报告](../实验记录/20260720-S3-A4鲁棒课程.md)。

---

## 8. S4 · A6 放量（算法冻结后）

- 只有在表示/loss/搜索都锁定后，才画 scaling curve（100k→500k→1M→…）。
- 目的：确认收益来自数据而非混杂改动；预期低对称随规模继续爬升。
- 资源：串行、长跑，用 `setsid + --resume` 持久化。

---

## 9. 可选支线（低优先，不阻塞主线）

- **S5a 低对称专项**：triclinic/monoclinic 过采样或专家头（注意 A2.5 已证 shared/MCL 无效，专项须以 S0.3 证据驱动）。
- **S5b 表示继续**：仅当 S0 显示 angle 是主失配源，再考虑 length/angle 解耦或度规分量按解码影响加权（G2 均值形式、S1/A5 最差维形式均已证 decoded-space 辅助 loss 无效，须彻底换思路而非再调超参）。
- **S5c 学习式 reranker**：Phase B 候选池稳定后的增量。

---

## 10. 依赖、排序与纪律

### 决策树（S0 后已锁定分支）
```text
S0 ✓
 ├─ sim→real 崩盘（MP100 23%，−19.5pp）→ A4(S3) 提前，与 S1 并列主线
 ├─ 现有 refine 头寸小（+3pp）        → S2 不优先；须先独立 q-search 原型
 └─ 近失配 / 精度头寸中等（+21.9pp） → S1 立即做
S1 ✗ 未过 Gate（best 56.0% < 95%），已记录淘汰，未进 100k
S3(A4) △ 合成 Gate 过，MP100 无实质收益（20.0%→20.7%）→ 生产仍用 A3-G1；扰动需重标定或主线转 S2
S2-B1-S0 ✓（高对称）/△（中低对称） 向量化重构后 cubic/tet/hex/trig 100% 过 95% Gate → 推进 B1-S1；ortho/mono 80%（未过，需顺序求解重构）；tric 0%（需换算法）→ 高对称先进 valid1400，ortho/mono/tric 继续算法迭代
S2-B1-S1 ✓ valid1400 真实数据 Gate 通过（整体 Top-20 64.4% vs NN 53.75%，+10.6pp；non-cubic 54.2% vs 40.0%，+14.2pp；均超 v3 §11.4 阈值）。修复 cubic Niggli 原胞（F/I 心）bug 后 17.5%→95%。trigonal 个体未达 +8pp（-10pp，疑似同类原胞设定问题）→ 高对称推进到 B2（候选排序），trigonal 顺手修复，中低对称并行顺序求解重构
S2-B2 ✓ Gate 通过（ranked Top-1 57.5%≥25%，efficiency 89.3%≥75%），legacy FOM+NN 体积先验已够用，无需新排序算法/learned reranker → 把 B1+B2 接成流水线，评估进 B3 或直接推进放量
S2-B3 ✗ Gate 未过（+0.6pp < +2pp，angle MAE 完全不变）——无噪声模拟数据下没有残差可精修，干净空结果，不采用 → 生产流水线维持 B1+B2；B3 代码留给未来真实谱噪声场景复测
S2-B1-S0′ ✓ trigonal hex/rhomb 100%；顺序求解后 ortho/mono/triclinic **全部 ≥95%** synthetic Gate（tric 单独跑 100%）。详见 [顺序求解报告](../实验记录/20260722-顺序求解与trigonal修复.md)
S2-B1-S2 ✗ ortho+mono valid1400 Top-20 **15.0%** < NN **18.8%**。诊断：空池为主（13/15）；(A) 标签 CS≠Niggli 几何（ortho/mono 各仅 ~42% 几何一致）；(B) 轴向消光打断顺序求解。下一步 ortho 心型/Niggli 变体 + 非轴向对角；**不进 S4/600万**。详见 [报告 §5](../实验记录/20260722-顺序求解与trigonal修复.md)
S4 → 前序锁定后放量（须搜索全晶系 Gate + MP100 复评稳定后）
```

### seed / Gate 纪律（沿用 v3）
- 探路全 seed42；仅过 Gate 的候选跑 43/44。
- 不对失败方向做「多 seed 找最好值」。
- 架构/表示/loss 决策一律以 **100k** 为准；3500/P0 只淘汰「完全不收敛」。
- 单点回归改动看 valid1400 strict；搜索改动看 pool recall → ranked Top-1 → MP100。

### 资源
- 主训练串行（S1、S2 的学习式组件、S4）。
- 可并行：S0 诊断、S3 robust-valid 构建、B1 synthetic 搜索原型（不得用 valid/MP100 调参）。

---

## 11. 指标看板（每阶段更新）

| 阶段 | 关键指标 | 目标 |
|------|----------|------|
| S0 | MP100 strict / refine 头寸 / per-param 表 | 定位主矛盾 |
| S1 | valid1400 strict + 低对称 macro | ✗ 已淘汰（P0 best 56.0% < 95% Gate，未进 100k） |
| S2-B1-S0 | synthetic recall（无 NN，理想峰） | ✓ 向量化重构后 cubic/tet/hex/trig 100%（过 Gate）；△ ortho/mono 80%（未过）；✗ tric 0% |
| S2-B1-S1 | valid1400 高对称子集 pool recall vs NN | ✓ Gate 通过：整体 Top-20 64.4%（NN 53.75%，+10.6pp）；non-cubic 54.2%（NN 40.0%，+14.2pp）；cubic bug 修复后 17.5%→95% |
| S2-B1 | oracle pool strict-recall | ≥58%（+15pp） |
| S2-B2 | ranked Top-1 strict（固定 B1 pool） | ✓ Gate 通过：legacy FOM+NN 体积先验 57.5%（efficiency 89.3%），优于新写确定性 score（55.6%/86.4%），无需 learned reranker |
| S2-B3 | 精修后 ranked Top-1 / angle MAE 变化 | ✗ Gate 未过：+0.6pp（<+2pp），angle MAE 完全不变（无噪声数据没有残差可收）；不采用，B1+B2 已够用 |
| S2-B3 | Top-1 / angMAE | +2pp 或角度显著降 |
| S3 | V-clean / V-mixed / MP100 | △ 合成 Gate 过（+0.05/+6.74pp）；MP100 20.0%→20.7% 无实质收益；生产仍 A3-G1 |
| S4 | strict vs 数据规模 | 单调且低对称爬升 |
| 北极星 | MP100 strict | A3-G1 raw 锁定 **23%**；B1+B2（pred CS）**36%** / oracle **42%**（口径修复复跑终值，+13/+19pp；高对称 52%/62%；低对称 12.5%）。详见 [B1B2-MP100 复评](../实验记录/20260721-B1B2-MP100基线复评.md)。距 JADE 68% 仍差 ~26pp |

---

## 12. 一句话结论

指标结构证明瓶颈在「**低对称精度 + 合取**」与「**sim→real**」。S0：邻域搜索几乎耗尽（+3pp），MP100 掉到 ~20–23%。S1：decode 辅助 loss（G2/A5）均不过 P0。S3：合成扰动课程能抬 V-mixed（+6.74pp）但 **MP100 无实质收益（+0.7pp）**——合成鲁棒 ≠ 真实谱鲁棒。S2-B1-S0：独立 q-search 方法学验证可行且**高对称晶系（cubic/tet/hex/trig）已过 95% synthetic Gate**（向量化重构后，第一轮的吞吐不足被误判为方法学不足）；中低对称（ortho/mono 80%、tric 0%）仍需顺序求解算法重构才能过 Gate。**S2-B1-S1**：高对称子集放到 valid1400 真实数据，独立 q-search 相对现有 NN 邻域搜索基线整体 **+10.6pp（64.4% vs 53.75%）、non-cubic +14.2pp（54.2% vs 40.0%）**，v3 §11.4 Gate 全部通过；过程中修复了一个真实数据特有 bug（cubic 在 Niggli 原胞标签下的 F/I 心表示，17.5%→95%）；trigonal 个体未达标（-10pp），推测同源问题待修。**S2-B2**：在固定 B1 pool 上对比排序方法，**现有 legacy FOM 接上 NN 体积先验后即通过 Gate（ranked Top-1 57.5%、efficiency 89.3%）且优于新写的确定性 score**——B1 的 pool recall 优势能被现成组件兑现成 Top-1，不需要新算法也不需要 learned reranker。**S2-B3**：在 B1+B2 之上加迭代局部精修，Gate 未过（+0.6pp<+2pp，angle MAE 完全不变）——无噪声模拟数据下没有残差可精修，干净空结果，不采用（代码留给未来真实谱噪声场景）。因此后续**不改骨架、不再纠结 loss / 当前合成扰动配方**：生产锁定 **A3-G1**；主线独立 q-search 到此已经跑完 B1→B2→B3 全链路，**B1（高对称 q-search）+B2（legacy FOM+NN 体积先验）是一条已过 Gate、可直接使用的候选生成流水线**（trigonal 原胞设定 bug 待修）；中低对称并行做顺序求解重构冲 95% synthetic Gate；这仍是天花板最高的候选方向；辅线可选用真实谱误差重标定扰动后再开 A4′（届时可重新评估 B3）；最后放量（S4）。
