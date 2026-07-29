# 2026-07-29 — Arm A 消融结论与 PXRD-indexer 重训协议

> **性质**：路线决策 + 工程改造 + 零成本探针  
> **状态**：消融与训练改造 ✅；ArmA 6M 对照 🟡 进行中；PXRD-indexer 4 卡重训 ⏳ 待启动  
> **前序**：[`20260728-路线偏离说明与纠正.md`](20260728-路线偏离说明与纠正.md)、  
> [`20260728-McMaille照搬到瓶颈定位-完整复盘.md`](20260728-McMaille照搬到瓶颈定位-完整复盘.md)、  
> [`20260728-种子池失败分析与体积先验偏小.md`](../实验记录/20260728-种子池失败分析与体积先验偏小.md)

---

## 0. 命名

自研 `PeakGeometryTransformerEncoder + ConditionalFlowHead` 这条线正式命名为
**PXRD-indexer**（代码里 `--backbone peaktf` 保留兼容）。  
它不是「RealPXRD Bert − CSPNet」，而是从零训的峰几何 token + Fourier(g=1/d²) encoder。

---

## 1. Arm A 100k 消融：RealPXRD 迁移学不到优势

协议对齐 PXRD-indexer 100k：60ep / bs=256 / lr=1e-3 / equiv=off / eval@100 每 3ep。  
主指标：MP100 primitive L4-strict library@100。

| Run | 配置 | best library@100 | 备注 |
|---|---|---:|---|
| **PXRD-indexer** | peaktf scratch | **51%** @ep57 | 对照 |
| ArmA freeze | Bert+CSPNet 冻，训 flow | 43% | 原方案 A |
| A both unfreeze | Bert+CSPNet 解冻 | 41% | |
| C unfreeze CSPNet | 仅解冻 decoder | 44% | |
| D no CSPNet | 砍掉 CSPNet，Bert emb → flow | 42% | ≈ freeze |
| E Fourier pos | CSPNet 保留；`pos_proj` 随机 Fourier | 43% | 需解冻 encoder |
| F scratch d256L6 | 无 CSPNet；Bert-style 扩到 d256/L6 从零 | **37%** @ep48 | bs256 OOM → bs64×accum4 |

**未做**：B（只解冻 encoder）——Bert 仅 36,928 参、且 2θ `.long()` 锁死 1° 分辨率，单独解冻没有信息可学。

### 结论（已拍板）

1. Arm A 各变体统计上挤在 **40–44%**，彼此无显著差异；全部明显低于 PXRD-indexer 的 **51%**。
2. **CSPNet 是死重**：砍掉（D）≈ 全冻（A）。A2 壳在 peaks-only 下把原子特征置零，图通路贡献接近零。
3. **解冻无用**：encoder 太小 + 1° 量化；decoder 在 atom-feature=0 时学不到 indexing 几何。
4. **Fourier 位置编码 alone 救不了 37k Bert**（E=43%）。
5. **加宽 Bert-style encoder 从零训更差**（F=37%）——不是「容量不够」，是这条 encoder 形态本身不适合本任务。

因此：**继续抠 Arm A 架构没有预期收益。**  
主线回到 PXRD-indexer；ArmA 6M 以全冻对照跑完即可（或视 GPU 需要中止）。

实测细节仍见路线偏离文档 §「需要同时记录的技术事实」：Bert 36,928 参；+0.05° 平移 ‖Δemb‖=0。

---

## 2. 旧 PXRD-indexer 6M（`full6m_equiv_off`）的三个问题

### 2.1 没训完，且 resume 把后半段搞坏

计划 34ep，原始 run 只到 **ep18**，loss 仍在降（train 0.65→0.20，valid 0.64→0.25）。  
ep17+ 是 `--resume best.pt` 重启的：当时 `best.pt` **不存 `optimizer_state`**，AdamW 动量清零。

| | ep017 train / @1 | ep018 train / @1 |
|---|---|---|
| 原始 run | 0.2096 / 42% | 0.1995 / **44%** |
| resume 后 | 0.2929 / 29% | 0.2837 / 33% |

history 里 ep17–20「掉下来」的曲线是假的，不能读成过拟合。真实情况是 **ep18 还在涨**。

旧 run 用 MP100 选的 best：**library@100 = 69% @ep16**（@1≈39%；邻近 epoch @1 可到 ~44%）。

### 2.2 选模指标在选噪声

选模用 MP100（n=100）上 K=100 随机采样的 library@100。  
p≈0.6 时二项 std ≈4.9pp，再叠加 flow 采样噪声。曲线相邻 epoch 抖 10pp
（ep5=60% → ep6=50% → ep7=60%）。69% 与 62% **不显著**。  
相对单调的信号是 @1 与 loss。

### 2.3 评测容差比下游需求松一个数量级

L4-strict：`ltol=0.05`（5%）。McMaille Rp 门实际要约 **0.2%**。  
K=1000 池按对齐后最大边长相对误差重计：

| best 相对误差 | 覆盖 |
|---|---:|
| <0.2% | 38% |
| <1%（堪用） | **61%** |
| <5%（≈现口径） | 73% |

e2e Top-1 46% 落在 0.2%–1% 之间。LSQ 救不动：τ=0.005 时精修接受率仅 7.4%，
精修前后 `len_relerr` 仍 ≈1.59%。  
**把 library@100 从 69% 推到 75% 可能对 e2e 无用**，若新增命中都是 3–5% 误差。

---

## 3. 零成本探针：`sample_steps` 不是瓶颈

脚本：`scripts/probe_sample_steps.py`  
ckpt：`full6m_equiv_off/best.pt`（ep16）  
协议：固定 z0（common random numbers），MP100，K=100，只改 Euler 步数。  
产物：`results/flow_seedgen/probe_sample_steps.json`，日志 `logs/probe_sample_steps.log`。

| steps | L4-strict | **<1%** | <0.5% | <0.2% | 中位误差 |
|---:|---:|---:|---:|---:|---:|
| 25 | 67% | **51%** | 37% | 23% | 0.384% |
| 50 | 67% | **50%** | 39% | 26% | 0.368% |
| 100 | 67% | 50% | 38% | 28% | 0.346% |
| 200 | 67% | 50% | 40% | 27% | 0.316% |
| 400 | 67% | 50% | 40% | 27% | 0.337% |

25→400 步，**<1% 命中纹丝不动**；L4-strict 也不动。  
只有最深尾部有一点：`<0.2%` 23%→27%，中位误差 0.384%→0.316%，且 **200 步饱和**。

**推论**：离散化误差约在 0.05–0.07% 量级；主导误差是模型条件分布，不是积分。  
重训时选模用 `sample_steps=50` 足够；最终报告可再以 200 步重采一次白拿尾部。

---

## 4. PM 拍板（重训边界）

| # | 决策 |
|---|---|
| 1 | 跑 `sample_steps` 探针（已完成，见 §3） |
| 2 | 选模口径：**hit@K，对齐边长相对误差 < 1%** |
| 3 | 4 卡机可直接读 `/nanolab/...` 数据路径 |
| 4 | **从零重训**（不 resume 旧 full6m） |
| — | **忽略** 失败分析里的 P1（MAX_PEAKS / 体积先验 / 低对称上采样）与 P3 |
| — | 评测协议保持 **primitive** L4-strict；conventional 不作选模 |
| — | ArmA 6M 对照继续跑（本机单卡）；重训放到火山 4 卡机 |

---

## 5. 重训协议（PXRD-indexer full6m v2）

### 5.1 不变

- 模型：`backbone=peaktf`，encoder/flow 配置同旧 full6m  
- 数据：`train_full_niggli_seed42` ~5.96M，`equiv=off`，augment on  
- 优化：全局 batch **512**，lr=1e-3，warmup=1ep，cosine over **34ep**，wd=0.05  
- 推理采样：训练中 `sample_steps=50`，`eval_k=100`

### 5.2 变更

| 项 | 旧 | 新 |
|---|---|---|
| 选模 | MP100 library@100（5% 口径） | **valid 前缀 `valid_1pct`（<1%）** |
| 选模集 | n=100 MP100 | 固定 valid 前缀 n=300（`--valid-eval-n`） |
| 采样噪声 | 每 epoch 重抽 z0 | **固定 seed（CRN）**，epoch 间差值反映模型 |
| MP100 | 每 epoch，且驱动 best.pt | 默认仅 **最终 epoch** 报告（`--mp100-every 0`） |
| 日志 | 仅 history.json | + **`metrics.csv`**（四档命中率 + lr + select_score） |
| resume | 可从无 optimizer 的 best.pt 冷启 | **禁止**；`best.pt` 现也存 optimizer_state |
| 并行 | 单卡 | **torchrun DDP**，global batch 固定 512 |

四档并行记录：`<0.2% / 0.5% / 1% / 5%`，主选模用 1%。

### 5.3 4 卡启动（火山习惯）

入口：`scripts/train_pxrd_indexer_full6m.sh`  
换算同 `pxrd_mof` 的 `train_large_v2_expert.sh`：

```text
target_global_batch=512, n_gpu=4 → batch_size=128, grad_accum=1
```

```bash
cd /path/to/PRXD-Cell-indexing-model-0706
# 默认：global batch 2048（4×512）、lr=4e-3（相对旧 512/1e-3 线性缩放）、amp=bf16
n_gpu=4 MASTER_PORT=16520 bash scripts/train_pxrd_indexer_full6m.sh
```

输出默认：`results/flow_seedgen/pxrd_indexer_full6m_v2_gbs2048_bf16/`  
若要对齐旧单卡日程：`target_global_batch=512 lr=1e-3 amp=off out_dir=.../pxrd_indexer_full6m_v2`。  
单卡 ~0.9h/ep × 34 ≈ 31h；4 卡 + gbs2048 + bf16 预期明显快于最初的 gbs512/fp32 估算（~8–10h）。

### 5.4 期望收益排序（探针后修订）

1. **训满 34ep 到 cosine 末端**——上次恰好在 LR 仍高时停；这是最便宜的一档。  
2. **可信选模**——不再用 n=100 的 5% 噪声指标挑 ckpt。  
3. （已否决作为杠杆）加 `sample_steps`。  
4. （本轮不做）MAX_PEAKS / 体积先验 / 容量放大——等 v2 曲线出后再议。

---

## 6. 工程改动清单

| 文件 | 变更 |
|---|---|
| `scripts/train_flow_seedgen.py` | DDP；`forward()` 供 reducer；`--select-metric valid_1pct`；valid 精度评测；`metrics.csv`；resume 护栏；`--mp100-every` / `--valid-eval-n` / `--limit-train-batches` |
| `scripts/ft_realpxrd/models.py` | `ArmAFlowModel.forward`（DDP 同理） |
| `src/.../peak_transformer.py` | `pool_query` 在非 `attn` 池化模式下 `requires_grad=False`（否则 DDP 报 unused param；state_dict 形状不变） |
| `scripts/probe_sample_steps.py` | 新增：步数 × 精度阈值探针 |
| `scripts/train_pxrd_indexer_full6m.sh` | 火山 4 卡入口 |

验证：

- 单卡冒烟：选模路径 + `metrics.csv` + `best.pt` 含 `optimizer_state`  
- `torchrun --nproc_per_node=2` CPU 冒烟：2 epoch 跑通  
- 3-rank 梯度同步检查：`max param drift = 0`（确认 `forward` 真正打通 DDP）  
- launcher 端到端：读到 full jsonl → `186112` batch @ bs32 = 5.95M ✓

---

## 7. ArmA 6M 对照进度（本机，非阻塞）

`results/flow_seedgen/armA_full6m/`，`--arma-unfreeze none`，同超参。  
单 epoch ~2–2.5h（CSPNet 图构建偏 CPU，GPU util ~10–15%）。

| | ep1 library@100 | @1 | @20 |
|---|---:|---:|---:|
| ArmA freeze 6M | 30% | 12% | 24% |
| PXRD-indexer 旧 full6m ep1 | 35% | 12% | 29% |

与 100k 消融方向一致。34ep 按此速度约需 **3+ 天**；不挡 4 卡机上的 indexer 重训。

---

## 8. 与「路线偏离」文档的关系

2026-07-28 纠正决定是「按方案 A 重做」。  
**2026-07-29 消融给出上限证据后，主线再修正为：**

- 方案 A（RealPXRD 迁移）在本任务上 **到顶约 40–44%（100k）**，不值得继续架构调参；  
- **主线 = PXRD-indexer**，用严口径选模 + 训满 + 多卡从零重开；  
- ArmA 6M 仅作规模对照，完成后写入对比表即可。

路线偏离文档本身仍保留「过失定性」；**技术路线的最终裁决以本文为准**。
