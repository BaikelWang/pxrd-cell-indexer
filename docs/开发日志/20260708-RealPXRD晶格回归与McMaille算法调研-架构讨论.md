# 2026-07-08 — RealPXRD 晶格回归策略 / McMaille 算法调研 + 架构讨论

> **背景**：M2.0（100k scaling）完成、D29 决策门通过后，PM 提出重新审视模型输出架构是否过度复杂（误以为有 7 个回归头），进而引出"是否需要晶系分类头"的讨论。为此调研 RealPXRD-Solver 的晶格回归策略与经典指标化算法 McMaille 的工作原理，作为架构决策的参考依据。
> **性质**：只读调研 + 讨论，**未改代码**。是否采纳文末提案需 PM 决策。

---

## 1. RealPXRD-Solver 晶格回归策略调研

### 1.1 核心结论

| 问题 | 结论 |
|---|---|
| 回归目标形式 | **3×3 晶格矩阵**（非直接 6 参数），消息传递中用度量张量 `L·Lᵀ`（9维 flatten） |
| 回归目标 cell | **primitive cell**（`conventional: False`），与本任务一致 |
| 是否预测晶系 | **完全不预测**。无分类头，晶系信息不参与训练/推理，只在生成后用 pymatgen 做后验空间群分析（`2_check_spg.py`），与模型无关 |
| 回归范式 | **不是**监督回归（`MSE(pred, true)`），而是 **Flow Matching**：200 步 ODE 迭代去噪，loss 是 `MSE(pred_l, rand_l)`（噪声重建） |

### 1.2 关键代码依据

模型两个输出头（`app/model/cspnet_xrd.py:126-171`）：

```126:171:/nanolab/users/wyx/archive/RealPXRD-Solver/app/model/cspnet_xrd.py
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias = False)
        ...
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
        return lattice_out, coord_out
```

Flow Matching 训练循环（`app/model/flow.py:135-174`）：

```135:174:/nanolab/users/wyx/archive/RealPXRD-Solver/app/model/flow.py
        rand_l, rand_x = torch.randn_like(lattices), torch.rand_like(frac_coords)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x = self.decoder(...)
        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss = self.hparams.cost_lattice * loss_lattice + self.hparams.cost_coord * loss_coord
```

### 1.3 "with L" 模式含义（修正此前认知）

**"L" = Lattice 作为已知先验提供，与晶系无关。** 机制：`cost_lattice=0` → `keep_lattice=True`（`app/model/flow.py:61`）。

| 维度 | without L (`pxrd-all`) | with L (`pxrd-all-L`) |
|---|---|---|
| 训练 | 晶格参与 flow loss，`cost_lattice=1` | 晶格固定用真值，`cost_lattice=0` |
| 推理输入 | PXRD + 化学式 | PXRD + 化学式 + **6 个晶格参数**（转 3×3） |
| 推理时晶格 | 从随机噪声 ODE 演化 | 固定不变，只解坐标 |
| 晶系 | 不使用 | 不使用 |

`sample_flow_L.py:170` 命令行 `--cell` 要求 6 个 float，无晶系参数：

```170:170:/nanolab/users/wyx/archive/RealPXRD-Solver/scripts/sample_flow_L.py
    parser.add_argument('--cell', required=True, nargs=6, type=float)
```

**与本任务的关系**：RealPXRD with-L 假设"indexing 已完成"，只解决结构解析（原子坐标）子问题——是本任务下游的互补环节，不是竞争方案。它印证了"晶系在几何生成任务中不是必需的显式条件"这一点。

---

## 2. McMaille 算法本质调研

来源：Le Bail, A. (2004). *Monte Carlo indexing with McMaille*. Powder Diffraction, 19(3), 249-254.

**本质：对 6 大晶系分别独立做 Monte Carlo 参数空间随机搜索（穷举试探），不是"先算 lattice 再判晶系"，也不是"先判晶系再算 lattice"，而是晶系框架下的参数空间搜索 + 拟合优度筛选，两者天然耦合。**

核心流程：
1. **黑盒自动模式**：对 cubic / hex-trig-rhomb / tetragonal / orthorhombic / monoclinic / triclinic **逐一独立**跑随机搜索。每个晶系自由参数数不同（立方 1 个 → 三斜 6 个），因此在某晶系框架下采样时，晶系约束（如立方 a=b=c）已经隐式满足——晶系与晶格从一开始就是绑定采样，不存在"先后"关系。
2. 随机晶格参数 → 布拉格公式算理论峰位 → 与观测伪谱比较 → Rp 型 R 因子
3. R 足够小的候选进入局部 Monte Carlo 微调（200–5000 步，P≈15% 概率接受不改善的扰动以跳出假极小值）
4. 汇总全部晶系候选，按 **R 值最小 + 对称性最高 + 体积最小 + 索引峰数最多**（Occam's razor）排序选择最终解

**关键要点**：晶系不是被"预测"出来的，而是被"枚举"的搜索维度；最终解的可信度判据是纯几何拟合优度，而非分类概率。

---

## 3. 当前模型架构复核（澄清此前误解）

调研代码确认（详见 `src/pxrd_cell_indexing/model/heads.py`、`losses.py`、`topk.py`、`eval.py`）：

| 事实 | 代码依据 |
|---|---|
| **回归头本就不条件于分类头** —— 二者并行读同一 embedding，forward 中无耦合 | `heads.py:93-104`：`crystal_system_logits` 和 `lattice_norm` 分别独立由 `self.crystal_system_head(embedding)` / `self.lattice_head(embedding)` 产生 |
| Regression 头输出 6 个数（非 7 个头，此前 PM 认知有误） | `heads.py:39-52`，`test_heads.py:26` |
| Loss 组合：默认 `1.0*CE + 1.0*SmoothL1(6参数)` | `losses.py:41-56` |
| `top1_lattice_match_rate` 用 pymatgen `Lattice.find_mapping`，**ltol=0.3（30%长度容差！）、atol=10°** —— 相当宽松，不检验晶系对称性是否精确成立 | `eval.py:27-28, 100-110` |
| Top-K 候选生成（primary/secondary/supercell 变体）**强依赖**分类头输出（晶系标签来源、confidence 排序） | `topk.py:116-230` |
| 归一化仅针对 6 参数（log+zscore 长度，zscore 角度），无 matrix/Gram tensor 归一化实现 | `normalization.py:26-98` |
| 模型输入纯 PXRD 峰表，无化学式/组成信息（与 RealPXRD 不同） | `dataset.py:306-313`，`bert.py:103-109` |

**重要发现**：`top1_lattice_match_rate` 的容差本就很宽松（ltol=0.3），意味着"预测晶格是否精确满足某晶系的对称约束"从评测口径上看**本来就不是必要条件**——这在一定程度上削弱了"必须先精确判晶系再按约束回归"的原始动机。

---

## 4. PM 提案与讨论

**PM 提案**：参考 RealPXRD，放弃 6 参数直接回归，转为预测 3×3 晶格矩阵；完全不做晶系分类；回归目标为 primitive cell；benchmark 时把矩阵转 6 参数，用 pymatgen 后验算晶系与 lattice match rate。

### 4.1 这个提案实际包含两个可分离的决策

**决策 A：是否去掉晶系分类头（分类损失 + Top-K 中的分类依赖）**

支持理由：
- 代码事实已证明，回归头从来没有条件于分类头的输出——去掉分类头不会损失"条件回归"的任何既有收益
- match 容差宽松（ltol=0.3），说明精确对称性不是匹配成功的必要条件
- 用 pymatgen 从预测晶格**几何地**推导晶系，比"两个独立头各自训练、可能互相矛盾"更自洽（不会出现"晶格像四方但分类说单斜"的内部不一致）
- RealPXRD 的先例支持"几何回归任务本身不需要显式晶系监督"

代价/风险：
- **Top-K 候选生成机制目前强依赖分类头**（primary 标签来源、7 个 secondary 晶系假设、confidence 排序全部来自 softmax logits）。去掉分类头后，这部分逻辑必须重新设计——不能是简单的"删掉一个头"。
- 需要一个新的候选生成策略。这里 McMaille 的思路恰好提供了一个更本质的替代方案：**对每个预测晶格，尝试"snap"到 7 个晶系（或更精确的 Bravais 类型）各自的约束表，用几何拟合优度（snap 前后偏差大小）排序生成候选**，而不是用分类概率排序。这与 McMaille"穷举 + 拟合优度筛选"的哲学完全一致，且比当前"同一组参数只换标签"的 secondary hypothesis 做法更合理（现在的 secondary 候选实际上不满足对应晶系的约束，纯粹是换个标签蒙对 pymatgen 的宽松匹配）。
- 失去 `crystal_system_accuracy` 这个直接可读的分类诊断指标（但可以用"pymatgen 从预测晶格推导的晶系 vs. 真实晶系"重新定义一个等价指标，不算真正丢失）

**决策 B：6 参数 → 3×3 矩阵表示**

这是一个相对独立的数值/表示层面的决策，和是否去掉分类头没有必然关系：
- RealPXRD 的 3×3 矩阵在其标准构造约定下（a 沿 x 轴，b 在 xy 平面）其实只有 6 个非零自由参数，本质是 (a,b,c,α,β,γ) 的另一种参数化（用 cos/sin 组合），不是真正多出 3 个自由度
- 可能的好处：避免角度归一化的周期性/尺度问题，损失的几何性质可能更均匀，但这是需要实验验证的数值细节，不是架构层面的根本改变
- 需要重新设计 `LatticeNormalizer`（现在只支持 6 参数 log/zscore），且这几个矩阵分量的统计分布（含正负号、混合 sin/cos 项）需要重新摸底才能选归一化方式
- eval 端改动很小：预测矩阵 → 用 pymatgen `Lattice(matrix).parameters` 转回 6 参数即可接入现有 `top1_lattice_match_rate`

### 4.2 建议

不建议把决策 A 和决策 B 捆绑成一次改动，原因：
1. 二者收益机制不同（A 是任务范式简化 + 自洽性提升；B 是数值表示细节），混在一起做实验，如果结果变差/变好，无法归因是哪个变量起作用
2. 刚刚完成 M2.0 的 100k scaling 实验并做出"数据量是主要瓶颈"的决策（D29），如果紧接着做一次大架构改动，应该保证能在同一个 100k 基线上做严格的 ablation 对比，而不是又混入新变量

**建议的验证顺序**（如果 PM 认可方向）：
1. **先做决策 A 的最小改动实验**：保留 6 参数回归头不变，去掉分类头/CE loss，Top-K 用"snap 到各晶系约束表 + 几何拟合优度排序"重新实现，在 100k 基线上对比 `top1_lattice_match_rate`（这个不需要晶系）和"post-hoc 晶系 + lattice 联合匹配率"（新定义的等价 joint 指标），与当前 100k 基线的 `top1_joint_match_rate` 对比
2. **决策 B 作为独立的后续消融**：在决策 A 定下来之后（或者甚至不依赖决策 A），单独测试 3×3矩阵/6参数 两种表示对回归精度（`lattice_mae`/`length_mape`）的影响

3. 无论 A 是否采纳，**Bravais 类型约束表**（此前讨论的"snap 到约束"依据）需要先做数据验证——即之前提过的"统计晶系类型分布和约束表在 primitive cell 下的真实成立情况"验证脚本，这个验证对 A、B 两个决策都有意义，可以先做。

---

## 5. 待 PM 决策的问题

- [ ] 是否认可"决策 A（去掉分类头，Top-K 改为 McMaille 式 snap+拟合优度排序）"作为下一步优先验证方向？
- [ ] 决策 B（3×3矩阵表示）是否作为独立、次要的消融实验，晚于 A 再做？
- [ ] 是否先跑 Bravais 约束表验证脚本，作为 A/B 决策的共同前置证据？

---

## 附录：本次调研的 Agent 记录

- RealPXRD 晶格回归/with-L 调研：agent id `3bde084b-c117-40f9-b63d-8a065e4d33c7`
- 当前模型架构/eval 容差复核：agent id `b87e80ed-7a8b-4683-afbc-9291eac7d25d`
- McMaille 论文：Le Bail, A. (2004), *Powder Diffraction* 19(3):249-254, DOI: 10.1154/1.1763152
