# 实验设计 — RealPXRD Without-L：formula / 坐标对 lattice 的消融（MP100）

> **日期**：2026-07-14  
> **状态**：✅ A2（纯 XRD→lattice）已在 MP100 @ 0.05/3° 跑完；A0c/A1 本轮未做  
> **目的**：检验预训练 `pxrd-all` 的 lattice 是否依赖 `atom_types` / 坐标演化，还是主要吃 XRD emb。  
> **性质**：对**现成 checkpoint 的破坏性消融**，不是 peaks-only 重训，也不是 indexing 新产品基线。  
> **关联**：[`20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md`](../开发日志/20260713-RealPXRD-WithoutL-MP100评测与indexing对标综合报告.md)

---

## 0. A2 实测结果（2026-07-14）

协议：`N=1`、atom emb×0、坐标冻结 @0.5、只更新 lattice；`pxrd-all`；MP100；K=20；`ltol=0.05` / `atol=3°`。

| 臂 | Top-1 map | Top-20 map | Top-1 ew | Top-20 ew |
|---|---:|---:|---:|---:|
| **A0**（峰+formula，联合采样） | 34% | 67% | 32% | 62% |
| **A2**（纯 XRD→lattice） | **2%** | **35%** | **0%** | **0%** |

产物：
- 脚本：`archive/RealPXRD-Solver/scripts/eval_mp100_xrd_only_lattice.py`
- JSON：`archive/RealPXRD-Solver/实验/mp100_without_l_lattice/ablation_A2_xrd_only_ltol0.05_atol3.json`

**解读**：elementwise 归零 → 预训练 lattice **强依赖**原子图/组成条件；Top-20 mapping 35% 而 ew=0，多为 `find_mapping` 宽松伪命中，不能当作有效 indexing。  
本 ckpt **不能**在掐断 formula 后当 peaks-only indexing 用。

---

## 1. 假设与要回答的问题

**代码事实**（已核实）：

- Bert **只**吃 `(pxrd_x, pxrd_y, peak_num)`，不看 formula。
- `lattice_out` 与 `coord_out` 共用 CSPNet 节点表征；`atom_types` 进 `Embedding`，与 XRD emb、time concat 后做消息传递，再 mean-pool → lattice。

**待检验假设**：

| ID | 假设 | 若成立，预期现象 |
|---|---|---|
| H1 | 预训练 lattice **强依赖** `atom_types` | 掐断元素语义后 Top-1/Top-20 相对 A0 **大幅下降** |
| H2 | 坐标联合扩散对 lattice **有实质耦合** | 仅冻坐标（仍真 formula）相对 A0 有可见掉点 |
| H3 | XRD emb  alone 在**未重训**权重下仍带可观晶胞信息 | A2（N=1 哑原子）仍显著高于随机/历史 ~5% 误用基线 |

**本实验不声称**：消融后的分数 = peaks-only indexing 能力上限（那需要去掉 atom 条件后重训）。

---

## 2. 固定条件（所有臂共用）

| 项 | 设定 |
|---|---|
| 模型 | `pretrained/weight/2501/pxrd-all/last_one.ckpt`（Without-L） |
| 数据 | MP-100 CIF，与既有评测同一目录 |
| 峰 | conventional → reduced → XRDCalculator，`y>5`（同 `eval_mp100_without_l_lattice.py`） |
| 真值 | primitive `(a,b,c,α,β,γ)` |
| K | `num_evals=20`（独立噪声采样，oracle Top-20） |
| 时间步 | `infer_timesteps=200`，`seed=42` |
| 尺子 | `ltol=0.05`，`atol=3°`；同时报 **mapping** + **elementwise** |
| 主表指标 | Top-1 / Top-20 × mapping / elementwise（4 个数） |

对照基准（已跑，作 A0）：

| | Top-1 map | Top-20 map | Top-1 ew | Top-20 ew |
|---|---:|---:|---:|---:|
| A0 全量 | 34% | 67% | 32% | 62% |

---

## 3. 实验臂设计

### 3.1 变量正交表

真正要拆开的是两件事：**元素语义**、**坐标耦合**；另加 **图规模（num_atoms）** 是否泄漏组成。

| 臂 | atom 语义 | num_atoms | 坐标 | 意图 |
|---|---|---|---|---|
| **A0** | 真 primitive `atom_types` | 真 N | 正常更新 `x_t` | 基线（已完成） |
| **A0c** | 真 `atom_types` | 真 N | **冻结**（不更新） | 单拆坐标耦合（H2） |
| **A1** | **embedding 置零** | 真 N | **冻结** | 去元素语义，仍保留「原子个数」图规模 |
| **A2** | 哑原子（常数 type） | **N=1** | **冻结** | 最接近「纯 XRD→单节点→lattice」 |

> 不采用「`atom_types=0`」：代码为 `Embedding(atom_types - 1)`，0 → -1 非法。

### 3.2 各臂实现定义（评审用，精确到行为）

#### A0 — Baseline（已有）

- `atom_types` = primitive formula 展开  
- `sample()` 默认：`x` 与 `l` 联合更新  

#### A0c — Freeze coords, keep formula

- 输入同 A0  
- 采样循环内：`x_t_minus_1 = x_t`（等价于忽略 `coord_out`），**只更新 lattice**  
- `x_T` 初始化：建议固定为 **全 0.5**（可复现；避免随机坐标噪声干扰臂间对比）  
  - 备选：与 A0 同分布 `U(0,1)` 但逐步冻结——评审倾向 **固定 0.5**，减少额外随机源  

#### A1 — Zero atom embedding, keep N, freeze coords

- 构图仍用真 `num_atoms`（节点数 = 真 primitive 原子数）  
- `atom_types` 可填任意合法值（如全 `1`=H），但在 `CSPNet.forward` 中：  
  `node_features = self.node_embedding(...)` 之后立刻 **`node_features = node_features * 0`**  
  （比「换一种假元素」更干净：不注入错误化学语义）  
- 坐标：同 A0c，冻结于全 0.5  
- XRD / time / lattice MP / `lattice_out`：**照常**  

> 注意：A1 **仍泄漏 N=num_atoms**（图大小、mean-pool 分母）。若 A1 ≫ A2，说明「个数」仍是重要组成代理。

#### A2 — Single dummy node, freeze coords

- `num_atoms=1`，`atom_types=[Z_dummy]`（建议 `Z=1`，仅占位；随后同样 **emb×0**，与 A1 一致地去掉元素语义）  
- 坐标：单点冻结 0.5  
- 全连接图退化为 1 个自环；XRD emb 打到唯一节点 → MP → pool（恒等）→ `lattice_out`  
- **这是对「formula 整条支路」最彻底的切断**（无元素、无 N）

### 3.3 明确不做的变体（避免评审发散）

| 变体 | 为何本轮不做 |
|---|---|
| 随机打乱 formula | 与「置零」信息不同，留作可选 A1b |
| 只置零 `coord_out` 但仍步进随机 `x_t` | 坐标仍进边特征，消融不干净 |
| 改 Bert / 重训 | 超出「预训练 ckpt 依赖」问题 |
| 喂错化学式但正确 N | 另题；可作 follow-up |

---

## 4. 采样伪代码（A1/A2/A0c 共用骨架）

```text
xrd = Bert(pxrd)                         # 不变
l = randn(3,3)
x = ones(N,3) * 0.5                      # 冻结初值
for t in reverse_timesteps:
    node = AtomEmb(atom_types)           # A1/A2: 随后 node = 0
    node = concat(node, time, xrd[nodes])
    node = CSPNet_MP(node, x, l)         # x 不更新，仍参与边（常数）
    pred_l, pred_x = lattice_head(node), coord_head(node)
    l = update(l, pred_l)                # 只动 lattice
    # x 保持不变
return l → (a,b,c,α,β,γ)
```

实现落点（评审通过后再改）：

- 优先：**推理脚本内包装 `sample_ablation(...)`**，不改动训练用 `CSPFlow.sample` 默认行为  
- 或给 `sample(..., freeze_coords=False, zero_atom_emb=False, force_num_atoms=None)`  

---

## 5. 结果解读矩阵（先定判决规则）

设 \(R\) = Top-20 elementwise（主判决；mapping 并列）。相对 A0 的掉点 \(\Delta = R_{A0}-R\)。

| 结果模式 | 解读 |
|---|---|
| A0c ≈ A0，A1/A2 大崩 | lattice 主要靠 **atom 语义（+N）**；坐标耦合弱；XRD-only 在本 ckpt 上不可用 |
| A0c 明显掉，A1/A2 更掉 | 坐标与 atom **都重要** |
| A1 ≈ A0，A2 崩 | 关键泄漏是 **num_atoms（图规模）**，不是元素种类 |
| A2 仍接近 A0 | 本 ckpt 的 lattice **主要吃 XRD emb**（意外；需复查实现是否未真正置零） |
| A2 ≫ 随机，但 ≪ A0 | XRD 有贡献，但预训练 lattice **仍依赖** 原子图条件 |

**随机下限（可选 sanity）**：同一脚本对 lattice 输出高斯噪声或打乱峰，确认 A2 不是评测 bug。本轮可不跑，若 A2 异常高再补。

---

## 6. 工程与成本

| 项 | 估计 |
|---|---|
| 臂数 | A0 复用已有 JSON；新跑 **A0c + A1 + A2** = 3×100×K20 |
| 单臂时间 | ≈ A0，约 **3–4 min**（4090）；合计 ~10–15 min |
| 产出 | `实验/mp100_without_l_lattice/ablation_{A0c,A1,A2}_ltol0.05_atol3.json` + 汇总表 |
| Smoke | 每臂 `--limit 2` 先过，再全量 |

---

## 7. 成功标准（实验本身，不是模型达标）

本设计「做对了」当且仅当：

1. A0 数字与已存 JSON **可复现**（同 seed，允许极小浮点差）  
2. A1/A2 确认 **atom emb 为 0**（脚本内 assert 或 debug 打印一次范数）  
3. A0c/A1/A2 确认 **坐标轨迹恒定**  
4. 四指标齐全，写入实验记录并回写综合报告一小节  

---

## 8. 请 PM 确认的点

1. **臂集合**：是否同意 **A0（复用）+ A0c + A1 + A2**？是否砍掉 A0c 以省一臂？  
2. **冻坐标初值**：固定 **0.5** vs 随机但冻结？  
3. **A1 置零方式**：`emb * 0`（推荐）vs 全体同一假元素且不置零？  
4. **A2 的 N=1 + emb×0**：是否作为「切断 formula」的主结论臂？  
5. **主指标**：是否同意以 **Top-20 / Top-1 elementwise** 为主、mapping 为辅？  

确认后按本设计改推理脚本并开跑。
