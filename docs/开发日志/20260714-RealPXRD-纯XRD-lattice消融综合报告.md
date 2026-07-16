# 2026-07-14 — RealPXRD Without-L「纯 XRD→lattice」消融实验综合报告

> **性质**：设计 → 实现 → MP100 实测 → 结论落盘  
> **状态**：✅ 完成  
> **关联设计**：[`../实验记录/20260714-RealPXRD-formula消融-lattice依赖实验设计.md`](../实验记录/20260714-RealPXRD-formula消融-lattice依赖实验设计.md)  
> **前序报告**：[`20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md`](20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md)

---

## 1. 为什么做这次实验

前序 Without-L @ MP100 得到 **Top-20 lattice match 67%**（峰 + primitive formula）。随后核对代码发现：

- Bert **只吃 PXRD**，不看 formula；
- 但 `lattice_out` 与 `coord_out` **共用** CSPNet：`atom_types` 进节点 Embedding，与 XRD emb 拼接后消息传递，再 mean-pool 出 lattice。

因此出现争议：**67% 到底多大程度靠峰，多大程度靠 formula？**  
若掐断 formula 后 lattice 仍高，则 Without-L 的峰通路对 indexing 有直接参考价值；若崩掉，则确认该 ckpt **不能**当 peaks-only indexing 用。

本轮按 PM 决策：**只做最干净的「纯 XRD→lattice」一臂（设计中的 A2）**，在 MP-100 上测 `ltol=0.05` / `atol=3°`。

---

## 2. 我们做了什么

### 2.1 实验问题（一句话）

> 用预训练 `pxrd-all`，在推理时去掉 formula / 原子语义 / 坐标演化，只保留 XRD emb → 节点 → MP → lattice，MP100 严口径 match rate 还剩多少？

### 2.2 消融协议（A2）

| 项 | 设定 |
|---|---|
| 权重 | `archive/RealPXRD-Solver/pretrained/weight/2501/pxrd-all/last_one.ckpt` |
| 数据 | `MP-100samples-benchmark`（100 CIF） |
| 峰 | conventional → reduced → XRDCalculator，`y>5`（与 A0 相同） |
| 真值 | primitive `(a,b,c,α,β,γ)` |
| **num_atoms** | **固定 1**（哑节点） |
| **atom embedding** | lookup 后 **整段置零**（不喂真实化学式） |
| **分数坐标** | 冻结在 **0.5**，不更新 `coord_out` |
| **更新量** | 仅 lattice ODE（200 步，同原 `sample`） |
| K | `num_evals=20`（独立噪声，oracle Top-20） |
| 尺子 | `ltol=0.05`，`atol=3°`；mapping + elementwise |

数据流（消融后）：

```
PXRD → Bert → xrd(512)
                ↓ 注入唯一哑节点
         atom_emb = 0（切断 formula）
         frac_coords = 0.5（冻结）
                ↓
         CSPNet MP → mean-pool → lattice_out
         coord_out 计算但不用于更新
```

对照臂 **A0**（前序已跑）：真 primitive formula + 联合 lattice/坐标采样。

### 2.3 工程产物

| 产物 | 路径 |
|---|---|
| 推理脚本 | `archive/RealPXRD-Solver/scripts/eval_mp100_xrd_only_lattice.py` |
| 结果 JSON | `archive/RealPXRD-Solver/实验/mp100_without_l_lattice/ablation_A2_xrd_only_ltol0.05_atol3.json` |
| 运行日志 | 同目录 `ablation_A2_run.log` |
| A0 对照 | `.../mp100_without_l_ltol0.05_atol3.json` |
| 设计文档 | `Task/.../docs/实验记录/20260714-RealPXRD-formula消融-lattice依赖实验设计.md` |

实现要点（避免踩坑）：

- 不用 `atom_types=0`（`Embedding(atom_types-1)` 会越界）；用合法占位 type + **emb×0**；
- 不只把 `coord_out` 置零——坐标若仍随机演化会进边特征；本实验 **冻结坐标**；
- 未改训练代码；仅推理路径包装 `decoder_forward_xrd_only` + 自定义 sample 循环。

---

## 3. 结果

### 3.1 主表（MP100，K=20，0.05/3°）

| 臂 | 输入条件 | Top-1 map | Top-20 map | Top-1 ew | Top-20 ew |
|---|---|---:|---:|---:|---:|
| **A0** | 峰 + primitive formula，联合采样 | **34%** | **67%** | **32%** | **62%** |
| **A2** | 仅峰（N=1，emb=0，冻坐标） | **2%** | **35%** | **0%** | **0%** |

A2 全量约 **94 s**（RTX 4090）。

### 3.2 结果解读

1. **Elementwise 归零（0% / 0%）**  
   在拒子胞/超胞的严尺子下，纯 XRD 通路用该预训练权重 **得不到任何有效晶胞命中**。

2. **相对 A0 断崖**  
   Top-1 elementwise：32% → 0%；Top-20 elementwise：62% → 0%。  
   说明此前 Without-L 的 lattice 命中 **强依赖** 原子图条件（formula / `atom_types` / 图规模），不是「Bert 看完峰就独立出晶胞」。

3. **Top-20 mapping 35% 不可当真**  
   mapping>0 而 elementwise=0：抽查命中样本多为半胞、错角（例如真胞 ~6 Å 而预测 ~3 Å）。  
   属 `find_mapping` 宽松伪命中，**不能**解读为「XRD-only 仍有 35% 召回」。

4. **与 indexing 任务的关系**  
   本消融证明：现成 `pxrd-all` **不能**在掐断 formula 后充当 peaks-only indexing 引擎。  
   Indexing 正统对标仍是 **McMaille / JADE**；NN indexing 仍需在 peaks-only 设定下自己解决搜索与回归问题。

---

## 4. 结论（给后续攻关用）

| 结论 | 含义 |
|---|---|
| Bert 只吃峰 | 架构事实成立 |
| lattice 头与 atom 图耦合 | 架构事实成立；A2 定量证实 **预训练权重依赖该耦合** |
| A0 的 67% | 是「峰 + 组成条件生成」的 oracle Top-20，**不是** indexing 基线 |
| A2 的 0% ew | 破坏性消融下 peaks-only 失效；**不等于**「架构去掉 atom 再训也一定为 0」 |
| 产品/科研口径 | Without-L 继续标为结构探索；indexing 坚持 **仅峰 + λ** |

**一句话**：我们验证了——在不动权重、只砍 formula/坐标的前提下，RealPXRD Without-L 的 lattice 几乎全靠原子图条件；纯 XRD→lattice 在 MP100 严口径下 **elementwise 为 0**。

---

## 5. 明确没做的事

- 未跑 A0c（仅冻坐标）、A1（保留真 N 仅 emb×0）——PM 要求只做最干净 A2；
- 未重训 peaks-only RealPXRD / 未改 Bert；
- 未把 A2 结果写入 indexing 引擎对照表（无效基线）。

若后续要问「架构在正确训练目标下 peaks-only 上限是多少」，需要 **去掉 atom 条件后重新训练**，那是另一项实验，与本次 ckpt 消融不同。

---

## 6. 摘要表

| 项 | 内容 |
|---|---|
| 做了什么 | A2 消融：预训练 Without-L，N=1 + emb×0 + 冻坐标，MP100 lattice 评测 |
| 尺子 | 0.05 / 3°，Top-1 & Top-20，mapping + elementwise |
| 主结果 | A2 elementwise **0% / 0%**；A0 为 32% / 62% |
| 学到什么 | 该 ckpt 的 lattice **依赖 formula 通路**；不能当 peaks-only indexing |
| 下一步（可选） | 若关心表示能力上限 → peaks-only 重训；indexing 主线仍跟 Mc 式搜索 + raw 几何 |
