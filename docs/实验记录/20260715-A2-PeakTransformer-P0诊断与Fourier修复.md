# A2 P0 诊断与 Fourier 修复（2026-07-15）

## 现象

首轮 `overfit700_a2_peak_transformer_t3`（Linear 投影 raw `g` + CLS-only）：

| 指标 | 结果 |
|------|------|
| best strict elem | **5.0%** @ep237 |
| loose lattice match | **45.6%** |
| 对照 R9 hist Niggli overfit | strict 11.6% / loose **82.1%** |

短探针（同 batch、恒定 LR）：hist 200 step → loss 0.09；PT → 卡在 ~0.42（mean-floor）。

## 根因

不是 mask/packing bug（token/梯度均正常）。**标量 `g` 的 Linear 投影无法提供足够的高频位置分辨**，Transformer 早期 embedding 无区分度，FiLM 退化为晶系均值预测。

## 修复

1. **Fourier features on `g`**：16 个 log-spaced 频率，sin/cos → 再拼 `Δg/I/rank`
2. **Pooling**：`cls_mean = 0.5*(CLS + masked mean)`
3. P0 预算：`lr=3e-3`，`max_epochs=1200`（bs=700 时约需 ≥800 step）

探针（恒定 LR 800 step）：Fourier+mean → loose **93%** / strict 17%；hist → 100% strict（更快）。

## Gate 口径说明

历史 hist P0 在 cosine 调度下 loose ~82%（非 95%）。v3「≥95% elementwise」按 **loose lattice match（ltol=0.3）** 对齐 hist 过拟合能力；strict 为 north-star 观察项。本轮目标：loose ≥85%（对齐/超过 R9 hist），争取 ≥95%。

## 对照要求（用户）

100k T1/T2/T3 必须同时对比：

- **R10-slim baseline**：valid1400 elem **21.29%**
- **A1-M1 E1c**：训练中（见 `scale_100k_r11b_e1c_niggli_seed42`）

## P0 结果（Fourier 修复后）

`overfit700_a2_peak_transformer_t3_fourier_seed42`：

| 指标 | 结果 |
|------|------|
| best loose lattice match | **100%** |
| best strict elem | **97.00%** |
| Gate | **PASS**（loose≥95% 且 strict≥95%）|

旧 Linear(g) 臂：strict 5% / loose 45% → 已作废。

## 100k 消融（已撤销）

> **纠正**：消融应在 **3500**，100k 仅用于胜出者锁模。  
> 误启的 `scale_100k_a2_t{1,2,3}` 已停止，**不作裁决**。  
> 正确流程见 `20260715-A2-PeakTransformer-3500消融.md`。

对照锚点（100k 锁模阶段才用）：

| 臂 | valid1400 strict elem | 状态 |
|----|----------------------|------|
| **R10-slim** (baseline) | **21.29%** @ep32 | 冻结 |
| **A1-M1 E1c** | 训练中 | 100k 集成 |
| A2 winner | — | 3500 消融后再上 |
