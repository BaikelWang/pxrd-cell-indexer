# Step 6 — 进度追踪

> **最后更新**：2026-07-07

---

## 里程碑状态

| 里程碑 | 目标 | 状态 | 备注 |
|---|---|---|---|
| M0 | 任务目录 + Agent 契约脚手架 | ✅ 完成 | 2026-07-06 |
| M0.1 | PM 确认核心目标与数据源 | ✅ 完成 | 2026-07-06 |
| M1 | 数据管线 + 模型方案设计 | ✅ 完成 | M1.2–M1.6 完成 |
| M1.9 | 评测/训练工程修复 | ✅ 完成 | D28；pytest 43/43 |
| M2 | 训练 pipeline 首跑 | 🟡 | 10k 调优完成；待全量训练 |
| M3 | MP100 benchmark 评测 + 基线对照 | 🟡 | smoke 权重 MP100 评测已产出（ plumbing 验证） |

## 变更日志

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
