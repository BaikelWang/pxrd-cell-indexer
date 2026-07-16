# A2 Peak Transformer 第二轮优化 · Wave-2（H-sym）+ 3500→100k 迁移性发现

日期：2026-07-15
基座：T1（T20-pos，linear-g Fourier，cls_mean，FiLM，matrix6，baseline）；对照 T1=10.14%。

## 背景：原始 H-sym 在 Niggli 基下不可行（实证）

标签是 **Niggli 约化原胞**。实测 3500 训练集每晶系的约束模式：

- **cubic**：a=b=c 恒成立（500/500），角全相等 ∈ {60,90,109.47}（对应 P/F/I）✅ 干净
- **tetragonal**：相等长度对不固定（a=b 345 / b=c 127 / a=b=c 28）；45/500 角非 90
- **hexagonal**：a=b(258) 或 b=c(241) 交替；哪个角=120 也不固定
- **orthorhombic/trigonal/monoclinic/triclinic**：模式高度可变

原因：Niggli 按长度排序会把「唯一轴」放到不同位置（prolate/oblate）。→ **按 CS 硬编码自由度会污染大半标签**，只对顺序无关约束安全。代码里 `bravais_angle_prior_loss` 已印证（注释：CS_PHYS_PARAM_MASK 对原胞是错的）。

## 实施的两条约化基安全臂

| 臂 | 改动 | 性质 |
|----|------|------|
| Hs1 cubic-exact | cubic 头只回归 1 个 log 长度 + 3-way setting，硬构造 a=b=c、α=β=γ；eval 用 setting 分类器（无 oracle 泄漏） | 硬 DOF 缩减 |
| Hs2 angle_prior | 打开已有 `angle_prior`（cubic min-over-{90,60,109.47}，hex/trig β=90），weight=0.25 | 软对称先验 |

## 结果（best epoch）

| 臂 | strictElem | strictMap | loose | clfCS | oracleSE | predSE | csSubElem | cubic proxy |
|----|-----------:|----------:|------:|------:|---------:|-------:|----------:|------------:|
| **T1_base** | **10.14%** | 10.43% | 47.43% | 47.0% | 10.57% | 10.14% | 21.58% | 77.0 |
| Hs1_cubex | 8.86% | 9.14% | 47.29% | 49.7% | 9.57% | 8.86% | 17.82% | 75.0 |
| Hs2_angpri | 1.29% | 1.71% | 40.86% | 48.4% | 1.29% | 1.29% | 2.65% | 18.0 |

## 结论

1. **Hs1 cubic-exact 未超过 T1（8.86 vs 10.14），cubic proxy 甚至略降（75 vs 77）。** 诊断：硬 setting 选择很脆——setting 分类器错一档，角度就跳 30–49°，立刻 strict fail；而 FiLM 连续回归即便 setting 没定准也能落在正确角附近。硬正确性换掉了连续鲁棒性，得不偿失。且单标量长度头未必比 FiLM 的 matrix6 回归更准。
2. **Hs2 angle_prior 崩了（1.29%），但结论被污染**：angle_prior 的 Huber 作用在**度数**量级（角误差 ~30° → Huber ~145），weight=0.25 后（~36）碾压了 matrix6 SmoothL1（~0.2）。这是我的权重量纲 bug，不是对角度先验本身的干净检验。即便修好，鉴于 wave-1 的 H1 物理损失已中性，预期收益也低。

**保持 T1，H-sym 两臂均不采纳。**

## 更重要的方法论发现：3500 排名不迁移到 100k

100k lock-in 进行中（ep~46）：

| | 3500 strict | 100k strict (ep46) |
|---|---:|---:|
| T1 (T20-pos) | **10.14%**（3500 胜出） | 22.29% |
| T3 (T48-geom) | ~9–10%（略低于 T1） | **25.07%（反超）** |

**3500 上 T1>T3，100k 上 T3>T1——排名反转。** 说明：
- 3500 规模对架构差异是噪声受限的，其排名对 100k 没有可靠预测力；
- 两轮 3500 微消融（P1/P2/P3/H1/H2/Hs1/Hs2）都在 T1 的 ~1.5pp 噪声带内，很可能同样无法预测 100k 行为；
- 精度墙的本质是**数据规模 + 回归精度**，不是这些输入表征/损失/对称微调。

## 建议

1. **以 T3（T48-geom）为默认 encoder**（100k 上明确领先），等 T1/T3 lock-in 跑满确认。
2. 后续优化**直接在 100k 上验证**少数几条最有理论依据的杠杆（贵但有决定性），不再用 3500 做架构级筛选——3500 只适合「淘汰完全不收敛的臂」。
3. 候选可在 100k 复核者：log-Fourier（大晶胞长尾在 100k 有足够样本）、attention pooling（3500 loose 已略优）。cubic-exact / angle_prior 不再追。
