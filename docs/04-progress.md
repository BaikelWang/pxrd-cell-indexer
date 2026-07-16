# Step 6 — 进度追踪

> **最后更新**：2026-07-13

---

## 里程碑状态

| 里程碑 | 目标 | 状态 | 备注 |
|---|---|---|---|
| M0 | 任务目录 + Agent 契约脚手架 | ✅ 完成 | 2026-07-06 |
| M0.1 | PM 确认核心目标与数据源 | ✅ 完成 | 2026-07-06 |
| M1 | 数据管线 + 模型方案设计 | ✅ 完成 | M1.2–M1.6 完成 |
| M1.9 | 评测/训练工程修复 | ✅ 完成 | D28；pytest 43/43 |
| M2.0 | 100k scaling 实验 | ✅ 完成 | D29：数据红利 +6.5pp joint；冲更大规模 |
| M2.1 | Decision A：去晶系分类头 | ✅ 完成 | D30：主指标持平（39.7%→39.1%）；Bravais Top-K 召回不变 |
| M2.2 | Decision B：矩阵表示消融 | ✅ 完成 | D32：Top-1 +1.2pp（39.1%→40.3%）；Top-20 持平；matrix6 为 scaling 默认 |
| M2.3 | Decision B 修正：真 9 维矩阵回归对照 | ✅ 完成 | D33：真矩阵回归（40.4%）≈ matrix6重参数化（40.3%），无显著差异；确认 matrix6 已足够，不采纳 matrix9 |
| M2.4 | FOM 候选重排序（de Wolff M(N)） | ✅ 完成 | D34：Top-1 +13.8pp（40.3%→54.1%）；Top-20 召回不变；推理默认 fom rerank |
| M2.5 | FOM 排序优化 + Test 集天花板 | ✅ 完成 | D35：修复 λ/容差后 Top-1 85.9%（valid）/87.6%（test）；排序天花板 ~88% |
| M2.6 | 回归精度提升 Phase 0–3 | ✅ | sweep 完成；宽松 valid raw 最优 46.6%；严口径仍崩 |
| M2 | 训练 pipeline 首跑 | 🟡 | 100k 完成；全量推迟至段 3 |
| M3 | MP100 benchmark（宽松口径） | 🟡 | 曾报 raw58%/fom88%；**严口径仅 13%，未超引擎** |
| **M4** | **完胜 JADE/Mc 攻关（段 0–3）** | 🟡 | R3–R5 闭环：冠军 `cubic_split_clf` **elem 15.43%（过 Gate）**；ang/非立方/pull90 **未过**；R4/R5 先验·多锚·输入均未超冠军。见 R5-C 定案 |

## 变更日志

### 2026-07-13（M4-R5：严口径诊断 + 多锚点 + 输入消融 — 定案）

- R5-Diag：pool_elem 16.5%（仅 +1pp）；FOM elem **0%**（半胞偏好）；mapping 池 ~25%
- R5-A：多锚点合并 pool 掉到 ~3% → 否决
- R5-B：hist512 14.86% / imin0 14.71% → 否决
- **定案**：保持 `cubic_split_clf`；PXRD-only 单点回归对低对称欠定；下一步需组成条件或改评测/产品口径
- 文档：[`20260713-R5-C-定案与天花板.md`](实验记录/20260713-R5-C-定案与天花板.md)

### 2026-07-13（M4-R4：物理先验 + 难晶系专训 — 未超 R3 冠军）

- 零训练 Bravais snap：elem 15.43%→15.86%（+0.4pp）
- `manifold_consistency` λ=0.1/0.25：14.93%/15.21%，pull90 恶化，否决
- hard-CS ft2：修 warm-start 跳过 encoder 的 bug 后 14.86%，仍低于冠军；否决
- **保持冠军** `cubic_split_clf`；进入 R5 严口径 Top-K/FOM
- 文档：[`20260713-R4-方案-物理先验与难晶系专训.md`](实验记录/20260713-R4-方案-物理先验与难晶系专训.md)

### 2026-07-10（M4-R2：晶系条件化 — G2 Gate 未过，接近）

- 实现 `head_type=cs_conditional`（7 头）+ 轻量 CS 分类路由；P0 否决 Bravais `angle_prior`
- **P0-700**：oracle_cs PASS（noncub 99%/pull90 26%）；angle_prior FAIL
- **10k**：oracle 6.0% / cs_pred 5.7% vs shared 2.4%；pull90 73.5%→52–58%
- **100k**：cs_pred strict elem **13.57%** / ang 8.4° / 非立方 **2.9%** / pull90 **56.4%** — 未过 Gate（需 15%/6°/5%/55%），但相对 R1 shared **+3.8pp elem**
- predicted≈oracle → 分类路由可用；停止 500k/Top-K/FOM
- 文档：[`20260710-R2-P0-700过拟合探针.md`](实验记录/20260710-R2-P0-700过拟合探针.md)、[`20260710-R2-P0决策与10k上界.md`](实验记录/20260710-R2-P0决策与10k上界.md)、[`20260710-R2-G2-100k晶系条件化.md`](实验记录/20260710-R2-G2-100k晶系条件化.md)

### 2026-07-10（M4-R1：PXRD 输入优化 — G2 Gate 未过）

- 完成 Stage0→G2：物理特征层、`InverseD2HistogramEncoder`、peak-token 消融、700/10k/100k Gate
- **G0**：仅 histogram MLP 可过拟合（elem~98%）；物理峰-token Bert 全 FAIL
- **G1**：冠军 `hist + intensity_min=0` → valid strict elem **7.8%**（临界）；弱峰保留是关键
- **G2**：100k 最佳 `hist_imin5` → strict elem **9.79%** / ang **9.06°** / 非立方 elem **~1%** / pull90 **67.5%** — **未过 Gate（需≥15%/≤6°/≥5%/≤55%）**
- 裁决：输入主线有效但不足以单独打开产品 raw；**停止 500k/Top-K/FOM**；转晶系条件化与目标表示
- 文档：[`20260710-R1-Stage0-谱标签口径审计.md`](实验记录/20260710-R1-Stage0-谱标签口径审计.md)、[`20260710-R1-G0-输入表示过拟合.md`](实验记录/20260710-R1-G0-输入表示过拟合.md)、[`20260710-R1-G1-10k泛化与预处理消融.md`](实验记录/20260710-R1-G1-10k泛化与预处理消融.md)、[`20260710-R1-G2-100k严口径验证.md`](实验记录/20260710-R1-G2-100k严口径验证.md)

### 2026-07-10（M4-R0b：Loss 下不去的彻底诊断，7 组小实验）

- 脚本 `scripts/{probe_encoder,overfit_capacity,overfit_mlp_hist,analyze_quantization}.py` + 2 个 overfit config
- **根因**：RealPXRD 峰-token encoder 无法把「峰位→晶格」编码进表示 → 输出坍缩到均值（=pull to 90°）
- 逐项证伪：数据量❌(train≈valid)、容量❌(32d→3.5M全卡12°)、2θ取整❌(d误差中位0.5%)、歧义❌(碰撞32/1400)、head❌(线性探针≈整模型)、任务可学✅(MLP+Q→0.37°/99%)
- train loss 平台 0.25 = 700 过拟合平台 = 预测均值；欠拟合无泛化间隙
- **R1 方向**：显式 Q=1/d² 几何特征通道 / 重估 encoder 架构；新表示先过 700 过拟合闸门(train elem≥80%)再 scaling
- 文档：[`docs/开发日志/20260710-Loss下不去的彻底诊断R0b.md`](开发日志/20260710-Loss下不去的彻底诊断R0b.md)；canvas `raw-loss-rootcause-r0b`

### 2026-07-10（M4-R0：Raw 精度根因诊断）

- 脚本 `scripts/diagnose_raw_errors.py`；B3a/baseline × valid+MP100
- **核心**：非立方 strict elem **0.17%**（cubic 45%）；失败 85% both；pull→90° **75%**；hex γ 120→95、trig γ 60→91
- 峰数相关是晶系混杂；近失极少（失败 median 16.7° / 24.6%）
- 文档：[`docs/开发日志/20260710-Raw精度根因诊断R0.md`](开发日志/20260710-Raw精度根因诊断R0.md)

### 2026-07-09（M4 段1+段2 收口）

- B 系列完成：B4(+2pp) / B3a(+1pp, MP100 fom 14%) / B1a负 / B1b崩 / B2a略负
- Gate-1（topk_elem≥40%）与 Gate-2（fom≥30%）均未过
- 收尾大报告：[`docs/开发日志/20260709-段1段2收尾报告.md`](开发日志/20260709-段1段2收尾报告.md)
- 下一步：**提高 raw 回归精度**（数据规模 / 容量 / 相加式监督）

### 2026-07-09（M4-B2 完成：低对称过采样 — 略负）

- hard CS ×2（10万→15.7万行）；valid strict elem **5.6%→4.8%**
- 实验记录：[`docs/实验记录/20260709-B2-低对称过采样.md`](实验记录/20260709-B2-低对称过采样.md)

### 2026-07-09（M4-B3 完成：关 2θ shift — 小幅正向）

- `augment_shift_range=0`；valid strict elem **5.6%→6.57%**；MP100 fom **9%→14%**
- 实验记录：[`docs/实验记录/20260709-B3-减弱2θ增强.md`](实验记录/20260709-B3-减弱2θ增强.md)

### 2026-07-09（M4-B1 完成：严口径 loss — 未达 / 崩塌）

- B1a hinge：valid elem **5.6%→2.8%**（负）；B1b angle_heavy：**崩至 0.14%**
- 实验记录：[`docs/实验记录/20260709-B1-严口径对齐损失.md`](实验记录/20260709-B1-严口径对齐损失.md)

### 2026-07-09（M4-B4 完成：严口径选模 — 小幅改善）

- Trainer 双轨 loose/strict；`best_metric=strict_raw_top1_elementwise_rate`
- valid 严口径 elementwise **3.6%→5.6%（+2pp）**；MP100 持平量级；Gate-2 未达
- 修 warm_start：仅 continuous 跳过 embed_positions
- 实验记录：[`docs/实验记录/20260709-B4-严口径选模.md`](实验记录/20260709-B4-严口径选模.md)

### 2026-07-09（M4-A4 完成：低对称 Bravais 扩展 — 未达门槛）

- `--bravais-set extended`：+mono 三自由角 + hex_strict；default 8 假设不变
- 严口径 topk_elementwise：MP100 **8%→7–8%**；valid **10.7%→9.2%**
- **段1收口**：推理/轻编码均无法打开 Gate-1 → 转段2训练策略（B4 严口径选模）
- 实验记录：[`docs/实验记录/20260709-A4-低对称Bravais扩展.md`](实验记录/20260709-A4-低对称Bravais扩展.md)

### 2026-07-09（M4-A3 完成：连续 2θ 位置编码 — 未达门槛）

- `position_encoding: continuous` + warm_start（跳过 embed_positions）；100k × 10ep，best@ep5
- 严口径全面变差：MP100 topk_elementwise **8%→0%**；valid angle_mae **9.8°→10.7°**
- 结论：warm_start+短训无法对齐新 pos MLP；A3 降级 P2
- 实验记录：[`docs/实验记录/20260709-A3-连续2θ位置编码短训.md`](实验记录/20260709-A3-连续2θ位置编码短训.md)

### 2026-07-09（M4-A5 完成：Q-match refine — 未达门槛）

- 新增 `model/refine.py`（soft Q loss + L-BFGS-B）；`--refine-steps` 接入评测
- MP100 严口径：fom 最多 **+1pp**（12%→13%）；`topk_elementwise` 仍 **7–8%**
- 结论：seed 离真解太远，局部 refine 无效；推理侧微调到顶
- 实验记录：[`docs/实验记录/20260709-A5-推理侧Qmatch局部refine.md`](实验记录/20260709-A5-推理侧Qmatch局部refine.md)

### 2026-07-09（M4-A2 完成：候选池尺度+体积护栏 — 未达门槛）

- Top-K：去掉生成期 `len>=k` 截断；`--scale-set` / `--pool-max-log-volume-ratio`
- MP100 严口径 sweep：`topk_elementwise` 卡在 **7–8%**（目标 +10pp 未达成）
- A2b 无护栏 extended：mapping 94% / elementwise 7% / gap **87%** → 伪命中复现
- 结论：**仅扩池不够**；转 A5 refine / A3 encoder
- 实验记录：[`docs/实验记录/20260709-A2-候选池尺度与体积护栏.md`](实验记录/20260709-A2-候选池尺度与体积护栏.md)

### 2026-07-09（M4-A1 完成：严口径评测护栏）

- 实现 `elementwise` / `volume_guarded` / `mapping_vs_elementwise_gap`；接入 `eval_mp100.py` / `eval_valid.py`
- baseline 四份 json：`results/beat_engine/a1_{mp100,valid}_{loose,strict}.json`
- **关键发现**：宽松 FOM find_mapping ~86–89%，但 **fom elementwise ≈ 0–1%**；严口径 `topk_elementwise` 仅 **8% / 10.7%**
- 实验记录：[`docs/实验记录/20260709-A1-严口径评测护栏.md`](实验记录/20260709-A1-严口径评测护栏.md)
- 测试：`pytest tests/test_eval.py` **11/11**

### 2026-07-09（M4 启动：完胜引擎攻关方案 v1）

- 新增总纲：[`docs/开发日志/20260709-完胜引擎攻关方案v1.md`](开发日志/20260709-完胜引擎攻关方案v1.md)
- 方法论：**算法 → 训练 → 全量**；**调研 → 方案 → 小步验证 → 生产线**
- 北极星：MP100 @ **0.05/3°** fom_top1 **> 68.1%**（当前 ~9–13%）
- 段 1 立即项：**A1 评测护栏** → **A2 体积约束搜索** → A3 Encoder → A4 低对称 → A5 FOM（后置）
- 关联：[`20260709-当前问题总览.md`](开发日志/20260709-当前问题总览.md)

### 2026-07-09（算法现状与理论缺陷文档）

- 新增：[`docs/开发日志/20260709-算法现状与理论缺陷分析.md`](开发日志/20260709-算法现状与理论缺陷分析.md)
- 固化：管线现状、严口径 vs JADE/Mc 差距、代码级理论缺陷（encoder/loss/Bravais/FOM/选模/数据）、后续逐项排查框架
- 北极星明确为 MP100 @ **0.05/3°** 完胜 JADE **68.1%** / Mc **65.9%**（当前 NN fom 仅 7–13%）

### 2026-07-09（M3 / M2.6：MP100 双指标 + 消融续跑）

- `eval_mp100.py` 对齐 dual metrics：`raw_top1` / `fom_top1` / `topk`
- MP100（matrix6 基线）：raw **58%** / fom **88%** / topk **100%**（超 Mc 76% / JADE 73%）
- MP100（`length_angle`）：raw 47% / fom **92%** / topk 100%
- Phase0–3 流水线：`length_angle` 完成；当前训 `cs_mask`（~ep7）；随后 `cs_reweight` → `combined` → sweep
- 结果写入：[`docs/实验记录/20260708-回归精度提升Phase0-3.md`](实验记录/20260708-回归精度提升Phase0-3.md) §9

### 2026-07-08（M2.6 启动：回归精度提升 Phase 0–3）

- **目标**：单卡 100k 上提升模型**直接回归**精度；评测同时报告 raw / rerank / FOM 三档 Top-1
- Phase 0：`eval_valid.py` 双指标评测；训练验证拆分 `length_mae`/`angle_mae`；`raw_top1` 写入 valid 指标
- Phase 1：DataLoader `prefetch_factor`/`persistent_workers`；`profile_dataloader.py`；Trainer 首 epoch 计时
- Phase 2：损失升级 5 种 mode（`baseline`/`length_angle`/`cs_mask`/`cs_reweight`/`combined`）+ 4 个消融 config
- Phase 3：`sweep_train_hyperparams.py` 超参网格 launcher
- **实验执行**（2026-07-08）：profile 38.8ms IO / 5.1ms compute；baseline raw_top1=40.9%；修复 Trainer init bug；`length_angle` 训练中（epoch 2/20 raw=24.8%）；后台 watcher 将自动续跑 cs_mask → cs_reweight → combined → sweep → test1400
- 实验记录：[`docs/实验记录/20260708-回归精度提升Phase0-3.md`](实验记录/20260708-回归精度提升Phase0-3.md)
- **未做**：多卡 DDP、模型容量放大、500k scaling（留 Phase 4–6）

### 2026-07-08（维护：评测默认 FOM rerank + 文档对齐）

- `eval_valid.py` / `decompose_joint.py`：`--rerank` 默认改为 **`fom`**（与 D35 一致；`--rerank none` 保留作消融）
- `README.md` 架构图更新为 Decision A + matrix6 + FOM v2 管线；里程碑同步至 M2.5
- 清理 `fom.py` 未使用 helper；`types.py` 候选排序注释对齐 FOM 行为

### 2026-07-08（M2.5 完成：FOM 排序优化 + Test 集天花板）

- **D35**：修复 FOM 波长 bug（1.54056→**1.54184**，对齐 pymatgen `XRDCalculator()`）；重校准 `q_match_abs_tol=1e-6`（替代 v1 的 8% rtol）；新增 strict de Wolff / 强度加权 / scale 变体去重消融
- 新增：`scripts/sweep_fom_rerank.py`、`scripts/investigate_test_sample.py`、`configs/scale_100k_no_cs_matrix6_testset.yaml`、`model/fom_rerank.py`
- valid1400 网格（18 组）：最优 **heuristic + tol=1e-6** → **top1 85.9%**（vs v1 54.1%，**+31.8pp**）
- test1400 天花板（一次性，未调参）：**top1 87.6%** / topk 99.8%；池内 rank 1 集中度 **87.8%**
- **结论**：排序能力上限约 **88%**；与 ~99% 召回之间 **~12pp 不可由 FOM 挽回** → 下一步应冲 500k/全量训练拉高回归精度
- 实验记录：[`docs/实验记录/20260708-FOM排序算法优化与Test集天花板测试.md`](实验记录/20260708-FOM排序算法优化与Test集天花板测试.md)

### 2026-07-08（M2.4 完成：FOM 候选重排序）

- **D34**：Top-20 池内真解 ~99% 前提下，新增 McMaille 式 de Wolff M(N) 峰表拟合重排序（`model/fom.py`），完全替换 Bravais confidence 排序
- 新增：`theoretical_two_theta`、`de_wolff_fom`、`rerank_candidates_by_fom`、`slice_observed_two_theta`；`LatticeCandidate.fom_score`
- 评测：`eval_valid.py` / `decompose_joint.py` 增加 `--rerank {none,fom}`
- 结果：`scale_100k_no_cs_matrix6_seed42` + fom rerank；valid1400 **top1_lattice_match_rate 54.1%**（vs 40.3%，**+13.8pp**）；topk 99.1% 不变
- 排序键：n_matched ↓ → volume ↑（Occam）→ mean\|ΔQ\| ↑；诚实标注与严格 McMaille M20 的差异
- 测试：`pytest` **70/70** 通过
- 实验记录：[`docs/实验记录/20260708-FOM候选重排序.md`](实验记录/20260708-FOM候选重排序.md)

### 2026-07-08（M2.3 完成：Decision B 修正 — 真 9 维矩阵回归对照）

- **背景**：PM 指出 M2.2 的"matrix6"实际是 head 仍输出 6 个数的**目标重参数化**，并非 RealPXRD 真实做法；核实 RealPXRD 源码确认其 head 输出 **9 维无约束**、loss **直接在矩阵元素上算 MSE**（flow matching noise-prediction），全程不出现 6 参数表示
- **D33**：补做与 RealPXRD 设计对齐的"真矩阵回归"（head 输出 9 维，`Matrix9Normalizer` 9 分量 z-score，含零方差位保护，解码复用 `lattice_lengths_angles` 天然鲁棒）
- 新增：`HeadConfig.output_dim`（可配置 6/9）、`Matrix9Normalizer`、`head_output_dim()` 工厂、`scripts/compute_matrix9_stats.py`、`configs/scale_100k_no_cs_matrix9.yaml`
- 结果：`scale_100k_no_cs_matrix9_seed42` best@epoch7；valid1400 **top1_lattice_match_rate 40.4%**（vs matrix6 40.3%，**无显著差异**）
- **结论**：真 9 维无约束回归与 6 维重参数化性能几乎相同——提升来源是"目标坐标系非线性重参数化"本身，不是"更多自由度"；**推荐继续用更简单的 matrix6，不采纳 matrix9**
- 测试：`pytest` **64/64** 通过
- 实验记录：[`docs/实验记录/20260708-DecisionB-真矩阵回归消融.md`](实验记录/20260708-DecisionB-真矩阵回归消融.md)

### 2026-07-08（M2.2 完成：Decision B 矩阵表示消融）

- **D32**：Decision B 采纳（温和版）——lattice 回归改为规范 3×3 矩阵 6 自由分量归一化（`MatrixLatticeNormalizer`），模型头/loss/Top-K/评测下游不变
- 新增：`MatrixLatticeNormalizer`、`build_lattice_normalizer`、`scripts/compute_matrix6_stats.py`、`tests/test_normalization_matrix6.py`
- 配置：`configs/scale_100k_no_cs_matrix6.yaml`（严格 ablation，同 100k 数据同 seed，仅 `representation: matrix6`）
- 训练+评测：`scale_100k_no_cs_matrix6_seed42` best@epoch11；valid1400 **top1_lattice_match_rate 40.3%**（vs 方案 A 39.1%，**+1.2pp**）；topk 99.1% 持平
- 测试：`pytest` **57/57** 通过
- 实验记录：[`docs/实验记录/20260708-DecisionB-矩阵表示消融.md`](实验记录/20260708-DecisionB-矩阵表示消融.md)

### 2026-07-08（M2.1 完成：Decision A 去晶系分类头）

- **D30**：Decision A 采纳——移除晶系分类头/CE loss，Top-K 改为 Bravais 8 假设 snap+几何打分排序；晶系降级为事后观测指标
- 新增：`model/bravais.py`（约束表 + `generate_bravais_hypotheses`）、`tests/test_bravais.py`
- 重写：`model/topk.py`（去掉 `crystal_system_logits` 依赖）、`heads.py`（仅回归头）、`losses.py`（纯回归）
- 评测：`eval.py` 新增 `infer_crystal_system_idx_from_lattice`；`crystal_system_accuracy` 改为事后几何推断版
- 配置：`configs/scale_100k_no_cs.yaml`（严格 ablation，同数据同 seed）
- 训练+评测：`scale_100k_no_cs_seed42` best@epoch5；valid1400 **top1_lattice_match_rate 39.1%**（基线 39.7%，−0.6pp）；**topk 召回 99.3% 持平**
- 测试：`pytest` **51/51** 通过
- 实验记录：[`docs/实验记录/20260708-DecisionA-去晶系分类头.md`](实验记录/20260708-DecisionA-去晶系分类头.md)
- 前置验证：[`docs/实验记录/20260708-Bravais原胞约束验证.md`](实验记录/20260708-Bravais原胞约束验证.md)

### 2026-07-07（M2.0 完成：100k Scaling 实验）

- **D29**：100k scaling 判定——数据红利明确，真解 Top-1 **20.9% → 27.4%（+6.5pp）**；决策冲更大规模（500k/全量）
- 新增：`train100k_seed42.jsonl`（1.2M 池均衡抽样）、`lattice_stats_100k_seed42.json`、`configs/scale_100k.yaml`
- 新增：`scripts/decompose_joint.py`（真解漏斗分解）
- `best_metric` 实验口径改为 `top1_joint_match_rate`（产品 KPI）
- 实验记录：[`docs/实验记录/20260707-M2.0-100k-scaling.md`](实验记录/20260707-M2.0-100k-scaling.md)

### 2026-07-07（M1.9 完成：评测/训练工程修复）

- **D28**：code review 收尾——checkpoint/config 一致性校验、valid loss 口径修正、`best_metric` 默认改为 pymatgen `top1_lattice_match_rate`、新增 `top1_joint_match_rate`、共享 `resolve_paths`、`normalize_embedding` 接线
- 修复：`eval_valid.py` / `eval_mp100.py` / `diagnose_10k.py` 结果 json 的 `"experiment"` 字段取自 checkpoint 内配置（避免误标）
- 修复：验证阶段 `loss` 统一为真实 `IndexingLoss`（不再误用 `lattice_mae`）
- 新增：`eval.py::top1_joint_match_rate`（lattice match ∧ 晶系正确）
- 重跑 valid1400 评测：Top-1 real 数值与 M1.8 一致（smoke 32.4%、tune_long 35.4%）；`top1_joint_match_rate` ≤ `top1_lattice_match_rate` 成立
- 测试：`pytest` **43/43** 通过

### 2026-07-07（M1.8 完成：10k 调优与诊断）

- **D27**：`best_metric` 默认改为 `top1_lattice_match_proxy`；支持 `early_stop_patience`
- 新增：`scripts/diagnose_10k.py`（混淆矩阵 + hex/trig 根因）
- 新增：`configs/tune_long.yaml`、`tune_loss_*.yaml`；`losses.py` 接入 Kendall uncertainty weighting
- 诊断：hex/trig 短板 = 分类混淆 + primitive 角度回归失败；proxy 对 hex 过严（0% vs pymatgen 27.5%）
- 调优：valid Top-1 real **32.4% → 35.4%**（tune_long / uncertainty，+3.0pp）
- 实验记录：[`docs/实验记录/20260707-M1.8-10k调优与诊断.md`](实验记录/20260707-M1.8-10k调优与诊断.md)
- 测试：`pytest` **36/36** 通过

### 2026-07-07（M1.7 完成：Top-K + 真实 lattice match + MP100 评测链路）

- **D26**：D25 单头下的 Top-K 生成策略（主候选 + 晶系变体 + 倍胞/子胞变换）
- 新增：`model/topk.py`（`build_top_k_candidates`、`build_top_k_with_mc_dropout`）
- 新增：`eval.py` pymatgen 真实指标（`top1_lattice_match_rate`、`topk_lattice_match_rate`）；保留 proxy 用于训练监控
- 新增：`data/mp100.py`（CIF→conventional→reduced→XRDCalculator 模拟 + primitive truth）
- 新增：`scripts/eval_mp100.py`、`scripts/eval_valid.py`
- 产出：
  - [`results/valid1400_real_match_smoke_unfrozen_seed42.json`](../results/valid1400_real_match_smoke_unfrozen_seed42.json)（Top-1 **32.4%** / Top-20 **99.1%**）
  - [`results/mp100_eval_smoke_unfrozen_seed42.json`](../results/mp100_eval_smoke_unfrozen_seed42.json)（Top-1 **47%** / Top-20 **100%**）
- 实验记录：[`docs/实验记录/20260707-M1.7-TopK-MP100评测链路.md`](实验记录/20260707-M1.7-TopK-MP100评测链路.md)
- 测试：`pytest` **31/31** 通过

### 2026-07-07（M1.5/M1.6 完成：模型头 + 损失 + 10k smoke 训练）

- **D24/D25**：lattice 归一化（log+z-score / z-score）；全 6 primitive 参数回归 + 辅助晶系头
- 新增：`model/heads.py`（`IndexingModel`）、`losses.py`、`eval.py`、`geometry.py`、`data/normalization.py`
- 新增：`training/config.py`、`training/trainer.py`、`scripts/train.py`、`configs/smoke_*.yaml`
- 信号地板：[`results/baseline_floor_seed42.json`](../results/baseline_floor_seed42.json)（cls 14.3%，match 4.5%）
- Smoke 实验：仅分类 valid acc **50.9%**；联合 frozen/unfrozen valid match **14.4%/17.1%**（>> 4.5% 地板）
- 实验记录：[`docs/实验记录/20260707-M1.5-M1.6-10k-smoke训练.md`](实验记录/20260707-M1.5-M1.6-10k-smoke训练.md)
- 测试：`pytest` **23/23** 通过

### 2026-07-07（M1.4 完成：10k Dataset / DataLoader）

- 新增 [`scripts/investigate_valid_sample.py`](../scripts/investigate_valid_sample.py)：从 `pxrd_241113_valid.lmdb` 全量扫描分层抽 **1,400 条 valid**（D22）
- 产出：`data/processed/valid1400_seed42.jsonl`、`investigate_valid_stats_seed42.json`（7 晶系各 200，无 shortfall）
- 实现 [`src/pxrd_cell_indexing/data/dataset.py`](../src/pxrd_cell_indexing/data/dataset.py)：`PXRDDataset`（lazy LMDB、`lmdb_key` 定位）、`filter_peaks`、`augment_spectrum`（D23 upstream no-op 复现）、`collate_peak_batch`、`build_dataloader`
- `types.py` 补充 `CRYSTAL_SYSTEM_TO_IDX`
- 测试：[`tests/test_dataset_smoke.py`](../tests/test_dataset_smoke.py) 7/7 通过，含 **encoder 集成**（M1.3↔M1.4 端到端 `[B,512]`）
- 全量 smoke：[`scripts/smoke_dataloader.py`](../scripts/smoke_dataloader.py) 10k 一遍扫描 **2.37s**，峰数 mismatch=0
- 依赖：新增 `lmdb>=1.4` 到 `requirements.txt` / `pyproject.toml`
- 决策记录：D22（valid 来源）、D23（augment 二次过滤复现）

### 2026-07-07（D21 数据增强策略确认）

- 核实预训练权重 `pxrd-all` 训练时 **`xrd_augment: true`**（train 开、valid/test 关）
- **PM 决策 D21**：dataloader 调试期关增强；正式 smoke 训练 **train 开 / valid 关**，参数完全沿用 RealPXRD 原值（噪声5%、位移±0.1°、缩放0.8-1.2）
- 落地 `data/dataset.py::SpectrumAugmentConfig`（默认原值）+ `augment_spectrum` stub；更新 `01-design.md` §5.1、PM 决策清单 D21

### 2026-07-07（RealPXRD-compatible baseline 策略 + M1.3 encoder vendoring）

- **执行顺序修正**：首个 10k baseline 采用 **2θ + y>5 + 无 max_peaks 硬截断**，与 RealPXRD 预训练口径对齐；**d-I 表征延后到消融分支**
- 更新 `docs/01-design.md` §5.1、风险表、§9.3；更新 PM 决策清单 D16/D11
- **M1.3 启动**：vendoring `BertModel` + `transformer/` 至 `src/pxrd_cell_indexing/model/encoder/`
- 新增 `xrd_encoder.*` checkpoint loader + encoder smoke test（key coverage、`[B,512]` forward）
- Step 3 骨架：扩展 `types.py`（`PXRDPeakTable` / `LatticeCandidate` / Top-K `IndexingResult`），新增 `data/`、`heads/`、`topk/`、`losses/`、`eval/` 接口 stub

### 2026-07-07（M1.2 完成：10k 分层抽样 + 峰数分布统计）

- 新增只读调研脚本 [`scripts/investigate_10k_sample.py`](../scripts/investigate_10k_sample.py)（20 万随机池 → 晶系推导 → 10k 分层抽样）
- 产出：
  - [`data/processed/train10k_seed42.jsonl`](../data/processed/train10k_seed42.jsonl)（10,000 条，7 晶系各 1428–1429）
  - [`data/processed/investigate_10k_stats_seed42.json`](../data/processed/investigate_10k_stats_seed42.json)
- 关键统计（过滤后峰数 `y>5`）：20 万池 median=17、p95=51、p99=75、max=543
- `max_peaks` 候选供 PM 选择：**51**（~p95）、**75**（~p99）、**543**（不截断）
- 报告：[`docs/开发日志/20260707-数据调研-10k抽样与峰数分布统计.md`](开发日志/20260707-数据调研-10k抽样与峰数分布统计.md)
- 运行耗时 ~50s（16 workers）；LMDB 全程只读

### 2026-07-07（PM 第五轮确认：D17–D20，技术栈定稿）

- PM 逐项确认 `01-design.md` §9：
  - **D17** Encoder 依赖：精简 vendoring（只拷 `bert.py`+`transformer/`，手动摘 state_dict），不依赖 lightning/hydra/torch_geometric/torch_scatter
  - **D18** 训练框架：原生 PyTorch + 简单 yaml/dataclass 配置，不引入 Lightning/Hydra
  - **D19** Top-K 的 K 值：**K=20**；因晶系头仅 7 个，设计"主候选(7)+按概率分配次候选"方案填满 20（`01-design.md` §6.4）
  - **D20** 实验管理：TensorBoard
  - `max_peaks`（§9.3）：PM 明确不着急，留到 M1.2 数据统计后再定
  - 晶系细化到 space group/extinction group：PM 明确不考虑，任务边界维持 7 大晶系
- 至此 **D1–D20 全部确认**，`docs/01-design.md` 定稿（状态 🟢）
- 已按决策更新 `requirements.txt` / `pyproject.toml` 依赖清单（torch/numpy/pyyaml/tensorboard/pymatgen）
- 详情：[`docs/开发日志/20260707-PM决策与待确认清单.md`](开发日志/20260707-PM决策与待确认清单.md)、[`docs/01-design.md`](../01-design.md)

### 2026-07-07（Step 2 方案设计稿初版）

- 完成 `docs/01-design.md` 初稿：架构图、模块划分、数据/标签流程、模型细节（encoder+晶系条件化回归头+Top-K）、损失函数、评测方案、风险清单、候选方案对比留痕、里程碑细化
- 汇总 D1–D16 全部决策落稿
- 新发现并列出 **8 项此前未讨论的问题**（§9）：encoder vendoring 策略、训练框架/配置管理技术栈、峰数上限统计、Top-K 具体 K 值、实验管理规范、部署寻峰对齐、MP100 峰口径、晶系细化空间（后两项为遗留非阻塞项）
- 状态：等待 PM 对 §9 逐项确认，之后才更新 `02-skeleton.md` 真实模块目录

### 2026-07-07（PM 第四轮确认：D9 / D15 / D16）

- PM 采纳文献调研建议，正式确认：
  - **D9 损失函数**：晶系条件化 mask 回归 + 长度 MAPE/log-MAE、角度 Huber + uncertainty weighting 自动配权
  - **D15 Top-K 实现**：方案 A，7 个晶系头天然 Top-K（WTA 子头留后续增强）
  - **D16 encoder 冻结 + 2θ→d 预处理提前**：10k smoke 做 frozen vs unfrozen(小LR) A/B；**2θ→d 转换提前到 smoke 阶段**做（不再等 P1）
- 至此 D1–D16 均已确认，仅剩「峰数计算成本/max_peaks 设计」「部署寻峰对齐」「MP100 峰口径」为非阻塞后续议题
- 详情：[`docs/开发日志/20260707-文献调研-损失函数TopK冻结策略建议.md`](开发日志/20260707-文献调研-损失函数TopK冻结策略建议.md)、[`docs/开发日志/20260707-PM决策与待确认清单.md`](开发日志/20260707-PM决策与待确认清单.md)

### 2026-07-07（文献调研：D9 / Top-K / encoder 冻结 / D11 新证据）

- 检索并精读 5 篇相关文献：RealPXRD-Solver 正式论文（arXiv:2603.00965）、AIdex（JCIM 2025）、Chitturi et al.（J. Appl. Cryst. 2021）、Multiple Choice Learning / Annealed MCL、Kendall et al. uncertainty weighting（CVPR 2018）
- 建议 D9：**晶系条件化 mask 回归**（每晶系只对自由参数计 loss）+ 长度 MAPE/log-MAE、角度 Huber + uncertainty weighting 自动配权
- 建议 Top-K：**方案 A，7 个晶系头天然 Top-K**（WTA 子头留后续增强，不阻塞初实验）
- 建议 encoder 冻结：10k smoke 做 frozen vs unfrozen(小LR) A/B
- **重要新发现**：RealPXRD-Solver 正式论文披露其输入表征已升级为 **d-I（interplanar spacing）而非 2θ-I**，并实测验证对仪器条件（λ/FWHM/2θ范围/步长）近似不变——这正是 D11 候选方向 (d)，且改动成本很小，建议重新评估优先级
- 报告：[`docs/开发日志/20260707-文献调研-损失函数TopK冻结策略建议.md`](开发日志/20260707-文献调研-损失函数TopK冻结策略建议.md)

### 2026-07-07（PM 决策 D4 / D12 / D13）

- D4：输入正式定为 **RealPXRD 风格变长峰表**；前端/Process 提供峰表 + λ
- D12：模型输出 **Top-K** primitive lattice + 晶系候选
- D13：baseline 站在 RealPXRD 上，保留 encoder/lattice 思路，**砍掉原子结构生成**
- 10k 抽样约束补充：**atom_num < 25**、晶系尽量均匀、固定 seed
- D14：Top-K 参考 RealPXRD `num_evals`；峰表只做 `y>5` 过滤，`max_seq_len=180` 不是峰数硬上限

### 2026-07-07（D11 训练输入与仪器泛化）

- 确认训练输入 = **Cu Kα（λ≈1.54184 Å）理想模拟峰表**；非实测、无 λ 字段
- PM：仪器泛化**后续可能要做**；初实验先按现有数据推进

### 2026-07-07（输入形态调研）

- 对照 RealPXRD / McMaille / JADE9 / GSAS-II 输入要求
- 建议：用户上传连续谱+λ → Process 寻峰 → **变长峰表**喂模型；λ 作全局条件
- 报告：[`docs/开发日志/20260707-输入形态调研-实验PXRD与Indexing工具对比.md`](开发日志/20260707-输入形态调研-实验PXRD与Indexing工具对比.md)

### 2026-07-07（RealPXRD-Solver 深度调研 + D10）

- PM 确认初实验 **10k** 抽样（241113 train）
- 完成 `archive/RealPXRD-Solver` 只读调研：BertModel encoder、峰表输入、权重路径、与 indexing 边界
- 报告：[`docs/开发日志/20260707-RealPXRD-Solver深度调研.md`](开发日志/20260707-RealPXRD-Solver深度调研.md)
- D4 倾向 **变长峰表（3a）**，待 PM 正式拍板

### 2026-07-07（PM 决策 D2–D8 第二轮）

- D2：训练用 **`pxrd_241113_*`**；初实验从 train **抽样**
- D3：PXRD 直接用 LMDB 现有谱
- D5：valid 调参，MP100 最终对照
- D6：benchmark 评测 **CIF truth → primitive**
- D7：晶系由 primitive 推导
- D8：baseline **复用 RealPXRD encoder**
- D4、D9：**待定**（输入形态、损失函数）

### 2026-07-07（PM 决策 D1）

- PM 确认训练标签：**监督 primitive 六参数**（`p_lattice_matrix`）
- 建立待确认清单 Q1–Q8：[`docs/开发日志/20260707-PM决策与待确认清单.md`](开发日志/20260707-PM决策与待确认清单.md)
- **暂不动手**：等口径批清后再写 dataloader / 设计稿

### 2026-07-06（训练数据调研）

- 完成 `alex_aflow_oqmd_mp` LMDB 只读摸底：条数、字段、PXRD 生成方式、数据来源
- 发现 **PXRD（conventional reduced）与 lattice 标签（primitive）可能口径不一致**，待 PM 拍板
- 报告：[`docs/开发日志/20260706-训练数据调研-alex_aflow_oqmd_mp.md`](开发日志/20260706-训练数据调研-alex_aflow_oqmd_mp.md)

### 2026-07-06（文档约定）

PM 确认文档分工，已写入 `AGENT.md` 与各目录 README：

- **`docs/开发日志/`** — 工作性质（做了什么、优化、设计、周报）；随手记录
- **`docs/实验记录/`** — 实验性质（设置、过程、结果、分析）；每次实验后留档

首条开发日志：[`docs/开发日志/20260706-任务初始化.md`](开发日志/20260706-任务初始化.md)

### 2026-07-06（PM 确认）

PM 确认四项核心信息，已写入 `docs/00-requirements.md`：

1. **目标**：训练模型，输入 PXRD → 输出晶系 + lattice
2. **训练数据**：`alex_aflow_oqmd_mp/datasets/`
3. **Benchmark**：`data/MP-100samples-benchmark/`（100 CIF）
4. **指标与背景**：`docs/开发日志/起点.md`

### 2026-07-06（脚手架）

- 创建任务目录 `PRXD-Cell-indexing-model-0706`
- 按 Agent 契约初始化 docs/、src/、tests/、guardrails
