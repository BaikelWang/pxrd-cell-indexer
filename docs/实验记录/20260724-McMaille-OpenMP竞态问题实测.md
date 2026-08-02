# 2026-07-24 — McMaille 开启 OpenMP 并行触发竞态条件（实测证据）

> **背景**：为「大批量 PXRD indexing」探索无损精度的加速手段，第一个被测试的方案是
> 「源码本就有 `$OMP` 并行结构，之前编译漏加 `-fopenmp`，补上就是免费多核加速」。
> **结论：这个假设是错的，直接开并行会丢精度，不能用。**

---

## 结论先行

McMaille v4.00（`McMaille.for`）里的 `$OMP PARALLEL DO` 结构使用了**未加保护的共享标志**
（`NOUT`、`INTEREST`），在真并行（`-fopenmp` + `OMP_NUM_THREADS>1`）下会触发竞态条件：

- 结果**不确定**（同参数重跑，耗时、候选表都不同）
- 会**提前中断搜索**（跳过本该跑的晶系/体积段）
- 会产出**损坏候选行**（`FoM=Infinity`、`NaN` 字段）
- 在其中一次重跑里，**Top-1 直接变成错误答案**

**不能**简单加 `-fopenmp` 上生产；也不能假设官方 `PMcMaille`（未审查其源码差异）天然修好了这个问题。

---

## 复现步骤

```bash
cd third_party/McMaille/run_lab
gfortran -O3 -fopenmp -ffixed-line-length-132 -std=legacy -fallow-argument-mismatch \
  -o mcmaille_omp McMaille_anchored.for

mkdir omp_test_000001 && cd omp_test_000001
cp ../{cub,hex,rho,tet,ort,mon,tri}.hkl ../mcmaille_omp ../run_000001/000001b.dat .

OMP_NUM_THREADS=32 ./mcmaille_omp 000001b   # 跑第 1 次
OMP_NUM_THREADS=32 ./mcmaille_omp 000001b   # 跑第 2 次（同参数）
```

---

## 实测数据

| 跑法 | 墙钟 | 单斜是否跑完 | Top-1 结果 |
|---|---:|---|---|
| 单线程基线（`-fopenmp` 未加，前序实验） | 677 s | 是（完整 666s） | ✅ `a=11.513, b=4.169, c=4.425`，`P Ortho`，命中 CIF 真解 |
| OMP 32 线程，第 1 次 | **1.5 s** | **否**（正交后直接判定 `Rp<Rmin`，跳过单斜） | 候选表首行 `IN=0, FoM=NaN`；第二行才是正确尺度的正交解 |
| OMP 32 线程，第 2 次（同一套参数） | **10.3 s** | 部分（仅跑了 138 s，应为 666 s） | **错误**：`IN=0, FoM=Infinity, a=11.32,b=6.62,c=6.44,β=96.9`；`IN=17, FoM=Infinity, a=12.15,b=11.59,c=3.53,β=93.4`——**均不匹配真解** |

两次相同参数的运行，耗时相差近 7 倍，结果也不同——典型的数据竞态特征。

---

## 根因（源码定位）

`McMaille.for` 的每个晶系搜索循环里：

```fortran
C$OMP PARALLEL DEFAULT(SHARED) ...
C$OMP DO
      DO 196 NCEL=1,NCELLS
      IF(NOUT.GE.1)GO TO 196
      IF(INTEREST.GE.1)GO TO 196
      ...
      IF(NTRIED.GT.TMAX)THEN
      NOUT=NOUT+1
      GO TO 196
      ENDIF
      ...
      INTEREST=INTEREST+1
```

- `NOUT`、`INTEREST` 在 `DEFAULT(SHARED)` 下是**跨线程共享**的
- 读（`IF(...GE.1)`）和写（`=...+1`）**都没有** `OMP ATOMIC`/`CRITICAL` 保护
- 任意一个线程的**局部**重试计数（`NTRIED`，`FIRSTPRIVATE`，各线程独立累积）一旦先撞线，
  会把共享的 `NOUT` 加 1；下一轮迭代里**所有线程**都会看到 `NOUT.GE.1` 而集体退出
- 线程数越多、调度越不确定，越容易撞到——这就是为何两次同参数运行结果都不同

另外还怀疑候选存储路径（`CEL`/`IGC` 等数组，仅部分操作包在 `CRITICAL(STORE1/STORE2/FOUND)` 里）
也可能存在类似未完全保护的路径，导致 `NaN`/`Infinity` 的候选行——未继续深挖，但已足够作为
「不能信任现状并行」的证据。

---

## 对官方 `PMcMaille`（并行发行版）的态度

`PMcMaille.zip` 里的 `McMaille.for` 确实把 `IPROCS=1`（硬编码）替换成了真正的
`IPROCS=OMP_GET_NUM_PROCS()`，说明官方并行版**确实打算**启用多线程。但：

- 我们**没有**审查过它是否修了 `NOUT`/`INTEREST` 的竞态
- 在没有独立验证之前，**不能假设**它是安全的，只能假设它「打算」并行

---

## 结论对「大批量 indexing 加速」的影响

1. **不要**用「开 OpenMP / 上 PMcMaille」作为默认加速手段，除非先做独立正确性审计
2. 批量场景下更安全、零风险的杠杆是**样本级多进程并行**（每个样本单线程跑完整算法，
   不同样本分布到不同核/进程），这条路径不触碰任何共享状态问题
3. 任何涉及并发/共享状态的优化，必须先建立「已知真值样本回归集」再验证，
   且要**多次重跑**确认结果稳定（本记录已证明单次运行「看起来对」不能说明安全）

---

## 产物

- 编译产物：`run_lab/mcmaille_omp`
- 复现目录：`run_lab/omp_test_000001/`（`000001b.imp`、两次 console 日志）
