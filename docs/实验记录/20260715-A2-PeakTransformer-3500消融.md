# A2 3500 消融裁决与 100k 双臂锁模

## 3500 结论（修正理解）

1. **Peak Transformer 整体成立**：T1/T2/T3 的 strict elem（9.3–10.1%）均显著高于 R10-H3（4.71%）与 E1c（7.14%），可视为优于 Dense hist 的主线。
2. **T1 vs T3 不宜仅凭 strict 定胜负**：strict 差距 <1pp；T3 在 loose lattice match、classifier CS、cubic/hexagonal 等维度领先。
3. **T2**：严格指标介于两者之间，100k 锁模暂不优先。

| 臂 | strict | loose | clf CS | 备注 |
|----|--------|-------|--------|------|
| T1 T20-pos | **10.14%** | 47.4% | 47.0% | strict 略高 |
| T3 T48-geom | 9.29% | **50.3%** | **54.7%** | loose/CS/高对称更好 |

## 100k 锁模（已启动）

并行挂起，对照 **R10-slim 21.29%** 与 **A1-E1c**（训练中）：

- `scale_100k_a2_t1_t20_pos_seed42` — log: `results/beat_engine/a2/scale_100k_a2_t1_lockin.log`
- `scale_100k_a2_t3_t48_geom_seed42` — log: `results/beat_engine/a2/scale_100k_a2_t3_lockin.log`

裁决：看 valid1400 strict elem，并兼顾 loose / CS / 分晶系；二者都过 Gate 则保留更均衡者或并列进入后续阶段。
