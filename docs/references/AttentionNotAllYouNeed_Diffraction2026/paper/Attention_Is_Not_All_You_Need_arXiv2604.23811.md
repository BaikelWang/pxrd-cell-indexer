Attention Is Not All You Need for Diffraction

arXiv is now an independent nonprofit! Learn more×

# Attention Is Not All You Need for Diffraction

Elizabeth J. Baggett Department of Physics, Boston College, Chestnut Hill, MA 02467, USA NIST Center for Neutron Research, National Institute of Standards and Technology, Gaithersburg, MD 20899, USA Edward G. Friedman Computer Science Department, School of Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213, USA NIST Center for Neutron Research, National Institute of Standards and Technology, Gaithersburg, MD 20899, USA Abhishek Shetty Derrick Chan-Sew Vanellsa Acha Harshita Dwarcherla School of Information, University of California, Berkeley, CA 94720, USA Paul Kienzle NIST Center for Neutron Research, National Institute of Standards and Technology, Gaithersburg, MD 20899, USA William Ratcliff william.ratcliff@nist.gov NIST Center for Neutron Research, National Institute of Standards and Technology, Gaithersburg, MD 20899, USA Department of Materials Science and Engineering, University of Maryland, College Park, MD 20742, USA Department of Physics, University of Maryland, College Park, MD 20742, USA

###### Abstract

Determining crystal symmetry from powder X-ray diffraction is a central problem in materials characterization, yet multiple space groups can produce indistinguishable patterns posing a significant challenge for automated classification. We show that attention-based architectures, while superior to convolutional networks for this task, are insufficient on their own: reliable symmetry extraction requires encoding crystallographic knowledge into both the network architecture and the training curriculum. We introduce a physics-informed transformer that classifies powder patterns into 99 extinction groups, the most specific symmetry classification accessible from diffraction data alone, using an explicit $\sin^{2}\!\theta$ coordinate channel, physics-aware positional encoding, and a structured multi-task decoder that separates geometric rule learning from holistic pattern recognition. A three-stage curriculum of balanced synthetic pretraining, realistic fine-tuning with explicit preferred-orientation modeling, and Bayesian prior injection proves essential for bridging the synthetic-to-real domain gap, while post-hoc temperature scaling—rather than additional training—is the key remaining ingredient for robust real-data transfer. By mapping predictions onto the directed acyclic graph of maximal translationengleiche subgroups, we show that the calibrated model’s errors are not random but physically structured: they remain local on the subgroup hierarchy and flow predominantly toward lower-symmetry descendants, consistent with the physical erasure of systematic-absence cues by real-world noise. We further identify a “catastrophic paradox” in which classical Rietveld fit quality does not cleanly predict neural classification difficulty, because lower label-space entropy partly offsets worse profile fits. These results establish that physics-informed target design, curriculum, and calibrated inference matter as much as model capacity for scientific machine learning on diffraction data.

## I Introduction

Powder X-ray diffraction (PXRD) is one of the most widely used probes for determining crystal structure, yet extracting symmetry information from a powder pattern remains a notoriously ill-posed inverse problem. Segal et al. have shown that the PXRD-to-structure loss landscape is highly non-convex: commonly used similarity metrics produce rugged optimization surfaces, and gradient-based refinement can converge to incorrect local minima even from moderately distorted initial states [16]. Constraining the search to the correct crystal family helps, but does not eliminate the problem. Traditionally, symmetry determination requires manual analysis by an expert crystallographer—a process poorly suited to the massive data volumes produced by modern high-throughput experiments and autonomous beamlines.

Machine learning offers a path toward automating this analysis. Convolutional neural networks (CNNs) were first applied to space-group classification from powder patterns by Park et al. [14], and subsequent work by Lolla et al. [12] and Schopmans et al. [15] achieved strong performance using ResNet architectures trained on synthetic data. More recently, transformer architectures [19] have been applied to diffraction: Chen et al. [3] used a Vision Transformer (ViT) [4] to classify metal–organic frameworks from PXRD patterns, and Simonnet et al. [17] reported gains over CNNs for mineral identification from synthetic single-phase data.

However, these prior studies share two limitations. First, they frame the task as space-group classification, even though diffraction cannot distinguish space groups that share identical reflection conditions. Second, they treat the diffraction pattern as a generic one-dimensional signal, applying standard image-classification architectures without encoding the physics of diffraction geometry. In this work, we address both limitations.

We find that reliable symmetry extraction requires physics at every level of the pipeline, not just the model architecture. Reframing the classification target from 230 space groups to 99 extinction groups—the information-theoretically identifiable equivalence classes under powder diffraction—more than doubles Top-1 accuracy on held-out synthetic data in controlled comparisons. A single post-hoc calibration parameter, applied without retraining, triples Top-1 accuracy on degraded real minerals by decoupling the model’s learned evidence from the geological prior absorbed during fine-tuning. And when we map the calibrated model’s residual errors onto the crystallographic subgroup hierarchy, we find they are not random: the model systematically falls to nearby lower-symmetry descendants, behaving like a conservative crystallographer under uncertainty. These results suggest that the $\sim$ 10% Top-1 ceiling on highly degraded real mixtures reflects a fundamental information-theoretic constraint of 1D powder diffraction rather than a limitation of the model.

In this work, we argue that attention mechanisms, while necessary, are not sufficient for reliable symmetry extraction from powder data. We make three contributions:

1.

We demonstrate that extinction groups—the 99 equivalence classes of space groups sharing identical systematic absences—are the information-theoretically correct classification targets for powder diffraction, and that reframing the task at this level substantially improves classification accuracy in synthetic benchmarks. A matched 230-space-group control confirms this point: even after collapsing its predictions post-hoc into extinction-group space, the space-group route remains well below direct extinction-group training on the same held-out regime.

2.

We introduce a physics-informed transformer architecture that incorporates an explicit $\sin^{2}\!\theta$ coordinate channel, physics-aware positional encoding, and a structured multi-task decoder separating crystallographic rule learning from holistic pattern recognition.

3.

We show that a three-stage training curriculum—balanced synthetic pretraining, RRUFF-conditioned realistic fine-tuning, and Bayesian prior injection at inference—is essential for bridging the synthetic-to-real domain gap. We further demonstrate that post-hoc temperature scaling (here, temperature is a mathematical scalar controlling the sharpness of the predicted probability distribution, not a physical quantity) resolves target-domain overconfidence, producing physically interpretable error structure on real-world mixtures.

We evaluate all models on a new algorithmically curated benchmark of 473 real RRUFF mineral patterns and report a detailed analysis of failure modes, including a “catastrophic paradox” in which classical Rietveld fit quality does not cleanly predict neural classification difficulty.

## II The Case for Extinction Groups

An extinction group is the set of space groups that produce identical systematic absences in reciprocal space. Symmetry elements such as lattice centering, glide planes, and screw axes determine which Miller indices $(hkl)$ can give rise to diffraction peaks. Because different space groups can share identical reflection conditions, they are experimentally indistinguishable by standard powder diffraction. For example, Friedel’s law further eliminates the distinction between centrosymmetric and non-centrosymmetric structures when anomalous scattering is negligible. These overlaps reduce the 230 crystallographic space groups to 99 unique extinction groups. (We merge hexagonal $P$ – $c$ – and trigonal $P$ —- $c$ because they produce identical systematic absences.)

The collapse from 230 space groups to 99 extinction groups has a direct consequence for machine learning: any model trained to predict space groups is penalized by the loss function for failing to distinguish physically indistinguishable patterns. Table 1 illustrates the effect: switching from space-group to extinction-group classification with the same ResNet-18 architecture substantially improves Top-1 accuracy (37% to 80%). This first comparison was deliberately illustrative, because it was confounded by both class count and dataset size. An additional, cleaner control matched 2.0M uniform corpus (see Supplemental Material for full details). Once both models are scored in extinction-group space, the post-hoc SG $\rightarrow$ EG collapse reaches 8.61% Top-1 versus 19.32% for direct EG training—a factor-of-two advantage that confirms the target-design effect is not merely a class-count artifact.

This mirrors standard crystallographic practice, in which an experimentalist first determines a pool of candidate space groups from extinction conditions, then narrows that pool using non-diffraction constraints such as polarity, centrosymmetry, and formula-unit compatibility with the unit-cell volume.

Table 1: ResNet-18 classification accuracy on balanced synthetic reflection data. The two runs differ in class count and dataset size (2.3M across 230 space groups vs. 990k across 99 extinction groups), so this table should be read as an illustrative rather than controlled comparison. The matched 2.0M SG $\rightarrow$ EG control discussed in the text provides that cleaner comparison and still favors direct extinction-group training.

| Target | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- |
| Space Groups (230 classes) | 37% | 74% | 88% |
| Extinction Groups (99 classes) | 80% | 95% | 97% |

## III Data

### III.1 The Distribution Problem

Well-populated crystal structure databases such as the ICSD exhibit steep label imbalance (Fig. 1). Table 2 illustrates the consequences: when a CNN is evaluated on the RRUFF test set, models trained on biased distributions outperform the frequency baseline at Top-1, indicating genuine feature learning, but by $k\approx 5$ their rankings increasingly recapitulate the training prior rather than discriminating among candidates on the basis of the pattern of systematic absences.

Figure 1: Extinction-group distribution in the Inorganic Crystal Structure Database (ICSD), showing severe geological class imbalance.

Table 2: Top- $k$ accuracy on the RRUFF test set when sampling training labels from different distributions. Above $k\approx 5$ , models trained on biased distributions are largely recapitulating the prior.

| $k$ | RRUFF | ICSD | Uniform |
| --- | --- | --- | --- |
| 1 | 12.3% | 9.7% | 1.0% |
| 5 | 47.4% | 27.2% | 5.1% |
| 10 | 67.0% | 53.7% | 10.1% |
| 20 | 83.7% | 76.4% | 20.2% |

Cross-evaluation confirms that balanced training produces more robust models: when a biased model is tested on a balanced test set, Top-1 drops to 3%, whereas a balanced model tested on the biased set retains 8% Top-1 (Table S1, Supplemental Material).

### III.2 Synthetic Data Generation

We employ two complementary generation strategies, both producing extinction-group-balanced datasets.

Reflection-Based Generation. For each target space group we generate random lattice parameters consistent with its metric constraints and use the Computational Crystallography Toolbox (cctbx) [6] to compute the allowed reflections. Each reflection is represented by a Gaussian peak at its $2\theta$ position with width given by the instrument resolution function.

Crystal-Structure-Based Generation. We use PyXtal [5] to build synthetic crystal structures, verified with ASE [11] to ensure the intended symmetry is preserved. Powder patterns are then simulated with PyCrysFML [8], which computes full structure-factor intensities together with Lorentz-type intensity weighting and instrument-specific resolution broadening, producing more realistic intensity envelopes at the cost of higher computational expense.

Full details of the database architecture, parallel generation pipeline, and HDF5 storage scheme are given in the Supplemental Material.

## IV Architecture

Powder diffraction is not a translation-invariant signal: absolute peak position matters because it encodes reciprocal-space geometry, and small lattice changes shift the entire pattern nonlinearly through Bragg’s law. The position-dependence of the signal is the core reason CNNs plateau early on this task, while attention-based models perform better on both synthetic scaling studies and downstream real-data transfer. Beyond any single peak’s absolute location, what carries diagnostic weight is the relational structure among peaks – their pairwise distances, intensity ratios, and co-occurrence patterns – dependencies that attention mechanisms are explicitly designed to capture, but that convolutional filters, constrained by local receptive fields, cannot model without significant depth. Our model therefore starts from a transformer backbone, extending the standard encoder with three physics-motivated modifications.

### IV.1 Coordinate Channel

Instead of feeding the transformer a single intensity channel, we provide a two-channel input: the intensity profile and an explicit $\sin^{2}\!\theta$ coordinate grid. This acts as a physical ruler that encodes the metric tensor relationship between peak position and $d$ -spacing, sparing the model from having to infer diffraction geometry from positional encoding alone.

### IV.2 Physics-Aware Positional Encoding

For the ViT models used in the real-data curriculum, we retain a learned absolute positional embedding over patch tokens and add a physics-derived positional term aligned to diffraction geometry. Concretely, each patch is assigned its mean $2\theta$ coordinate, transformed to $\sin^{2}\!\theta$ , scaled, and passed through a small MLP to produce a patch-level embedding that is added to the learned positional embedding. This injects reciprocal-space structure directly into the token positions while preserving the flexibility of a learned absolute embedding. In the trained real-data model, this learned physics term is nearly one-dimensional and monotone in the input $\sin^{2}\!\theta$ coordinate, indicating that it functions primarily as a learned reciprocal-space ruler rather than as an arbitrary positional code (see Supplemental Material).

This mechanism is distinct from the coordinate channel above. The coordinate channel exposes pointwise $\sin^{2}\!\theta$ values as an additional input feature before patchification, allowing early layers to reason jointly over intensity and physical location at the raw-trace level. The physics-aware positional encoding instead acts at the patch-token level, biasing self-attention with diffraction-aware token positions. In practice, the two components play complementary roles: one is an input-side physical ruler, while the other is a token-level positional prior.

### IV.3 Dual-Head Decoder

The shared transformer backbone feeds into two decoders that are trained jointly.

Split Head (Crystallographic Rule Decoder). This head predicts a structured set of crystallographic bits: crystal system, lattice centering, and glide-plane/screw-axis features, decoded through a deterministic lookup into extinction groups. The split head outputs a 37-bit target vector; crystal system and centering are trained with cross-entropy losses, while the sparse operator bits use binary cross-entropy with positive-class weighting (pos_weight). Not all 37-bit combinations correspond to valid extinction groups: the lookup table maps valid combinations to one or more extinction groups, while invalid bit patterns have no crystallographic interpretation. In practice, this loss acts as a structural regularizer: it forces the backbone to attend to weak present/absent peaks—the same features a crystallographer would examine—rather than relying solely on dominant fingerprint peaks.

Auxiliary Head (Direct Extinction-Group Classifier). This head predicts the 99 extinction groups directly via softmax. It learns the continuous joint distribution and is more robust to real-world noise than strict Boolean rule decoding.

Fusion Decoder. At inference time, we fuse the predictions:

| $$\mathbf{p}_{\mathrm{fused}}=\alpha\,\mathbf{p}_{\mathrm{split}}+(1-\alpha)\,\mathbf{p}_{\mathrm{aux}}$$ | (1) |
| --- | --- |

When rule evidence is weak, the split-head logits remain near zero and the induced extinction-group distribution becomes diffuse after lookup. Fusion therefore acts as a practical uncertainty-balancing mechanism: the auxiliary head dominates when the rule path is indecisive, while the split path sharpens predictions when physically distinctive absence cues are present.

### IV.4 Model Specifications

Regular Transformer (RT). $d_{\mathrm{model}}=256$ , 4 attention heads, RoPE [18], adaptive average pooling, 2.73M parameters. AdamW optimizer [13] with step scheduler.

Vision Transformer (ViT). Patch size 25, learnable CLS token, 8 heads, $d_{\mathrm{model}}=256$ , 9.52M parameters. Adam optimizer [9], fixed learning rate.

## V Training Curriculum

The central lesson of this work is that architecture and data design must be co-optimized. The main curriculum uses 1.38M uniform stage-1 samples followed by 2.35M corrected samples conditioned on RRUFF, an experimental mineral database, in stage 2. Late ablations then extend stage 2 with exact preferred orientation and mixed standard+preferred-orientation curricula (see Supplemental Material for full lineage).

### V.1 Phase 1: Uniform Synthetic Pretraining

To prevent the model from collapsing toward common low-symmetry classes, we pretrain on a large synthetic dataset balanced uniformly across all 99 extinction groups. Because empirical mineral distributions are strongly skewed toward lower-symmetry classes, training directly on geological frequencies risks learning the prior before learning the weak diffraction signatures of rare centering, glide, and screw rules. Uniform pretraining forces the model to represent all extinction groups evenly, building a symmetry-balanced geometric engine.

### V.2 Phase 2: RRUFF-Style Synthetic Fine-Tuning

Purely uniform synthetic data is too clean and chemically random. The second phase fine-tunes on synthetic patterns that incorporate realistic backgrounds, multiplicative and additive noise, peak broadening, and impurity phases. Crucially, main-phase stoichiometries are constructed from Wyckoff-compatible multiplicities, and structures are accepted only when the realized extinction group detected by spglib matches the intended label. This stage was the single largest driver of real-data improvement in our ablations.

### V.3 Phase 3: Bayesian Inference and Calibration

Because Phase 1 deliberately removes geological priors, we restore them at inference time using an Empirical Bayes prior—concretely, the log-frequency of each extinction group in the training corpus, added to the auxiliary logits before softmax—and no evaluation labels are used (see Supplemental Material for provenance). However, the stage-2 model already absorbs part of the geological prior through fine-tuning on millions of RRUFF-conditioned patterns. Adding the external prior without calibration therefore effectively double-counts this bias. To resolve this, we apply post-hoc temperature scaling [7]—dividing the auxiliary logits by $T$ before adding the external log-prior—which proved critical for real-benchmark performance.

Figure 2 shows the impact of each curriculum stage on a real RRUFF holdout.

Figure 2: Effect of curriculum stages on real RRUFF holdout Top-1 accuracy. RRUFF domain fine-tuning (Phase 2) is the dominant factor in bridging the sim-to-real gap.

## VI Results

We report results on three evaluation tiers: (i) balanced synthetic data for architecture comparison and scaling, (ii) the broader RRUFF-473 real benchmark for decoder comparison, and (iii) the more challenging RRUFF-325 real benchmark for calibration and topological analysis.

### VI.1 Scaling Studies

Table 3 presents the full scaling comparison on balanced reflection-based data. Despite having 3.5 $\times$ fewer parameters, the regular transformer consistently outperforms the ViT on idealized synthetic benchmarks, suggesting that patch tokenization is a liability when the input consists of perfectly sharp peak profiles. By contrast, the ResNet-18 baseline plateaus at approximately 80% Top-1 regardless of training set size (Fig. S1, Supplemental Material).

However, the real-data system is built on the physics-informed ViT backbone. We interpret this as a transfer/generalization tradeoff: the ViT’s patch-based representation is coarser, but that coarseness accommodates real-world peak broadening, background, and sample displacement. The ViT also provides a natural interface for the coordinate channel and physics-aware positional encoding, and its CLS token yields a global representation whose attention distribution can be visualized directly. Ablation studies (Table S3, Supplemental Material) confirm that the ViT is less sensitive to distribution shifts in $2\theta$ range and generation method. A separate positional ablation on real data confirms that the physics-aware PE carries the stronger positional prior; removing it collapses strict split-head validity to zero on both RRUFF benchmarks (Supplemental Material). Supplemental analysis further shows that the learned physics-aware positional term behaves as an almost one-dimensional $Q^{2}$ -like reciprocal-space ruler, supporting the interpretation that its gain comes from imposing the correct diffraction geometry rather than from adding arbitrary representational complexity.

Table 3: Top- $k$ accuracy (%) on balanced synthetic reflection data.

| Size | Regular Transformer | | | Vision Transformer | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Top-1 | Top-3 | Top-5 | Top-1 | Top-3 | Top-5 |
| 99k | 57.5 | 78.8 | 86.8 | 21.7 | 38.1 | 46.8 |
| 396K | 80.4 | 94.2 | 97.1 | 30.9 | 49.4 | 59.6 |
| 990K | 89.8 | 97.9 | 99.0 | 42.5 | 65.3 | 76.3 |
| 2.17M | 93.7 | 99.2 | 99.7 | 60.9 | 85.1 | 91.9 |
| 5.14M | 94.4 | 99.5 | 99.9 | 71.8 | 91.7 | 96.0 |

### VI.2 Real-Data Evaluation: Two Benchmarks

We use two complementary real-data benchmarks. The first is RRUFF-473, a 473-pattern benchmark built from upstream RRUFF scans [10] using algorithmic family-consistency filters and nuisance-fit stratification (172 recoverable, 153 usable, 74 poor, 74 catastrophic). This benchmark is algorithmically curated, not hand-cleaned, making the evaluation harder but more reproducible than sanitized sets used in prior work [15]. The second is RRUFF-325, a deterministic downstream subset from the same curation pipeline that retains only the usable and recoverable nuisance-fit strata. We use this stricter slice for calibration and topological analysis so that the most extreme nuisance regimes do not dominate those controlled measurements.

### VI.3 Decoder Comparison (RRUFF-473)

Table 4 and Fig. 3 summarize the decoder comparison. The system has two distinct operating points. Best Top-1: Stage-2b (larger uniform pretraining plus fine-tuning) with the fused decoder ( $\alpha=0.50$ ) at 16.70%, benefiting from larger uniform pretraining that increased geometric rigidity in the split path. Best Top-5: Stage-2a (the earlier fine-tuned checkpoint) with the Bayesian auxiliary head at 52.22%, retaining the softer joint ranking that the larger uniform stage partially traded away. Fusion improves Top-1 for both checkpoints, confirming that the split and auxiliary paths provide complementary information.

Figure 3: Decoder comparison on the RRUFF-473 benchmark.

Table 4: Top- $k$ accuracy (%) on the full RRUFF-473 benchmark.

| Model | Decoder | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- |
| Stage-2a | Split | 14.80 | 28.96 | 34.04 |
| | Bayesian aux | 14.59 | 36.15 | 52.22 |
| | Fused ( $\alpha\!=\!0.25$ ) | 15.64 | 34.88 | 45.24 |
| Stage-2b | Split | 15.01 | 27.91 | 32.77 |
| | Bayesian aux | 14.80 | 33.19 | 50.95 |
| | Fused ( $\alpha\!=\!0.50$ ) | 16.70 | 32.98 | 41.01 |

### VI.4 Why the Split Head Is Still Valuable

It may seem that the auxiliary head’s dominance makes the split head redundant. This is incorrect, for three reasons. First, the split head acts as a structural regularizer for the shared backbone, preventing lazy shortcuts that overemphasize tallest peaks while ignoring weak symmetry-defining features. Second, the split and auxiliary heads fail in different ways, which is precisely why fusion improves Top-1. Third, the split head provides a geometry-driven backup when geological priors are misleading. The strongest evidence is the fusion result itself: if the split head contributed no information beyond what the auxiliary head already captured, fusion could not improve Top-1.

### VI.5 Calibration on the Harsh 325-Pattern Benchmark

On RRUFF-325, the Stage-2c model’s raw auxiliary logits are strongly overconfident: the uncalibrated auxiliary path reaches only 2.15% Top-1 and 14.46% Top-5 on the 325-pattern benchmark. Applying temperature scaling ( $T=5$ ) to the auxiliary logits before Bayesian fusion changed the picture completely: the same checkpoint reached 9.54% Top-1, 27.38% Top-3, and 43.08% Top-5 without changing any model weights (Fig. 4). Standard calibration diagnostics confirm this improvement: the expected calibration error (ECE) falls from 0.457 to 0.049, the negative log-likelihood (NLL) from 6.58 to 3.70, and the multiclass Brier score [1] from 1.283 to 0.958.

This result materially changes the interpretation of stage 2. The fine-tuned model did learn useful real-domain structure, but its logits became saturated by the target-domain class imbalance. The dominant failure mode was not loss of diffraction physics; it was overconfident coupling of structural evidence to the geological prior. Temperature scaling restored usable uncertainty, allowing the external prior to act as intended rather than reinforcing the model’s internal bias.

Figure 4: Post-hoc calibration restores ranking accuracy on RRUFF-325. Results shown for the final Stage-2c checkpoint. Softening the auxiliary logits ( $T=5$ ) before adding the geological prior decouples structural evidence from geological overconfidence.

### VI.6 Preferred Orientation and Mixed-Curriculum Ablations

The stage-2c calibration result above established that post-hoc temperature scaling is essential, but the topological analysis (Section VI.7) also exposed a physically interpretable failure mode: the calibrated model remained strongly biased toward lower-symmetry descendant predictions when experimental nuisance effects erased weak systematic-absence cues. To test whether subtractive experimental noise was underrepresented in the simulator, we added exact March–Dollase preferred orientation (PO), which models systematic reflection suppression due to non-random crystallite orientation, to the stage-2 realism curriculum while leaving stage 1 unchanged.

The pure-PO ablation produced the best exact ranking result on the harsh RRUFF-325 benchmark: after one epoch of PO-conditioned fine-tuning, the calibrated Bayesian auxiliary decoder reached 13.54% Top-1 and 40.92% Top-5 on RRUFF-325, and 13.53% Top-1 and 49.89% Top-5 on RRUFF-473. This confirms that explicit modeling of texture-suppressed reflections improves exact ranking on real diffraction patterns. However, it also narrowed the broader ranking distribution: on RRUFF-325, Top-5 remained below the earlier non-PO Stage-2c value of 43.08%, and strict split-head validity collapsed to below 1%, indicating that a pure texture-heavy curriculum destabilizes exact Boolean rule extraction.

We therefore trained a larger mixed stage-2 curriculum combining approximately 2.346M standard RRUFF-conditioned samples with 500k PO samples in a single one-epoch continuation from the same uniform base checkpoint. This large mixed model is the best balanced final model. On RRUFF-325 it achieved calibrated Top-1 / Top-5 = 10.46% / 43.69%, slightly exceeding the earlier Stage-2c Top-5 while still improving Top-1 over the non-PO baseline. On RRUFF-473 it achieved 9.94% / 50.74%. Most strikingly, split-head validity recovered from 0.92% in the pure-PO run to 47.38% on RRUFF-325 and 49.26% on RRUFF-473, indicating that the mixed curriculum restores symbolic self-consistency while retaining the benefits of explicit PO exposure.

Table 5: Real-data performance and topological behavior on RRUFF-325 under calibrated Bayesian auxiliary decoding. All entries are measured on the same 325 patterns; “Split valid” reports the fraction for which the split head decodes to exactly one legal extinction-group template. Topological metrics (Desc./Anc./Branch, $\leq$ 2 hops, mean DAG distance) are defined in Section VI.7; branch jumps denote errors that cross between distinct crystal-system families in the DAG rather than moving within a single family’s hierarchy.

| Model curriculum | Top-1 / Top-5 | Split valid | Desc./Anc./Branch | $\leq$ 2 hops | Mean DAG dist. |
| --- | --- | --- | --- | --- | --- |
| Stage-2c (no PO) | 9.54 / 43.08 | $\sim$ 0.6% | 191 / 27 / 76 | 38.4% | 2.72 |
| PO-only, 1 epoch | 13.54 / 40.92 | 0.92% | 157 / 28 / 96 | 53.0% | 2.51 |
| Mixed 200k pilot | 13.23 / 40.92 | 1.54% | 153 / 30 / 99 | 49.65% | 2.59 |
| Final large mixed run | 10.46 / 43.69 | 47.38% | 165 / 31 / 95 | 51.55% | 2.46 |

### VI.7 Topological Evaluation of Symmetry Degradation

Standard Top-1 accuracy treats every misclassification as an equal categorical failure, but crystallographically some errors are far more reasonable than others. We therefore mapped predictions onto the condensed directed acyclic graph (DAG) of maximal translationengleiche ( $t$ -) subgroups—subgroups that preserve the translation lattice while removing point-group operations—linking the 99 extinction groups. One trigonal cycle is merged into a single node in this condensed graph (see Supplemental Material). In this graph, an edge represents the loss of a symmetry operation without changing the underlying translation lattice.

This analysis reveals a stark behavioral contrast. The older uncalibrated replay-era baseline (Legacy) has a mean topological error distance of 3.50 and is heavily biased toward ancestor hallucinations (158 higher-symmetry errors vs. 64 descendant errors), indicating reliance on chemical priors to guess high-symmetry structures while ignoring contradictory spectral evidence. By contrast, the calibrated non-PO Stage-2c model is much more local and conservative (Table 5): 38.4% of its wrong Top-1 predictions lie within graph distance $\leq 2$ of the ground truth (mean distance 2.72), and its directed errors flow predominantly downward—191 descendant vs. 27 ancestor errors. The PO-informed ablations then reveal a more nuanced picture. Pure PO training sharply improves local accuracy (53.0% of errors within $\leq 2$ hops) and reduces descendant errors to 157, but it also drives branch jumps upward and destabilizes split-head validity. The final large mixed model preserves most of that locality gain while recovering broad candidate coverage: 51.55% of wrong predictions lie within graph distance $\leq 2$ , the mean distance drops to 2.46, descendant errors fall to 165, and ancestor predictions rise modestly to 31 (Table 5).

This behavior is consistent with the physics of the problem. Higher-symmetry extinction groups mandate strict systematic absences (regions of zero intensity). Real-world background noise and impurity phases add intensity to reciprocal space, while preferred orientation can suppress whole reflection families. The non-PO model acts like a cautious crystallographer: observing intensity where an absence should be, it rejects the unsupported higher symmetry and conservatively defaults to a lower-symmetry descendant. Once explicit PO is introduced, the model learns that some missing reflections are not true absences but texture-suppressed observations. This weakens the descendant bias and tightens the local graph neighborhood of the remaining errors. At the same time, branch jumps remain elevated, which we interpret as a texture aliasing effect: once preferred orientation erases a diagnostic family of reflections, the surviving 1D barcode can become locally ambiguous between nearby crystallographic cousins rather than merely lower-symmetry descendants.

This descendant-biased error pattern also mirrors human crystallographic workflow, in which practitioners often fall back to lower-symmetry extinction groups when the data do not unambiguously support a higher-symmetry assignment.

Figure 5: Topological structure of Top-1 errors on the RRUFF-325 benchmark. Left: error-distance distribution on the condensed extinction-group subgroup graph. Right: directionality. Calibration and preferred-orientation-aware training shift errors toward shorter graph distances, reduce the strong descendant bias of the non-PO model, and reveal that many residual failures are local lateral hops rather than distant hallucinations.

### VI.8 The Catastrophic Paradox

One of the most striking findings is that classical Rietveld fit quality does not cleanly predict neural-network classification difficulty (Fig. 6). The 74 catastrophic-fit patterns span only 13 unique extinction groups and 38 minerals, whereas the 172 recoverable patterns span 37 extinction groups and 111 minerals. Importantly, the effect is not driven by a few dominant minerals: the most frequent mineral in the catastrophic stratum accounts for only 4 of 74 scans. On the final mixed checkpoint, raw Top-1 across the recoverable, usable-or-better, poor, and catastrophic strata is 11.05%, 9.80%, 10.81%, and 6.76%, respectively. Reweighting each stratum to a common extinction-group distribution changes these values to 13.18%, 10.61%, 10.71%, and 10.80%, showing that label-space concentration explains about 45% of the recoverable-versus-catastrophic gap, but not all of it.

There is also a materials explanation. Many catastrophic patterns arise from strongly textured or highly cleavable minerals, where preferred orientation severely distorts relative intensities without removing the underlying Bragg-angle topology. While Rietveld-style nuisance fits collapse because the profile model expects powder-averaged intensities, our neural models appear to rely less on exact relative peak heights and can still recognize the invariant geometric barcode when the intensity envelope is badly distorted.

The same entropic logic governs the model’s predictive “sinks.” The final RRUFF-325 comparison shows that the dominant measured sinks are EG 4 and EG 99 rather than a universal collapse into a single fallback group. Throughout, EG numbers refer to our internal extinction-group lookup-table indices, released with the code and Supplemental Material; they are identifiers rather than a separate international standard. When noise obscures fine peak-splitting, predictions concentrate into nearby low-constraint groups within the correct Bravais neighborhood—a structured, physically interpretable fallback.

Figure 6: The catastrophic paradox. Classical profile-fit quality and neural classification difficulty do not align monotonically; extinction-group reweighting removes about 45% of the recoverable-versus-catastrophic gap.

## VII Comparison with Prior Work

Having established the calibrated model’s performance and failure modes on our curated benchmarks, we now situate these results relative to prior work.

The most direct comparison is with Schopmans et al. [15], who trained ResNet models on large on-the-fly synthetic data (up to 261M diffractograms) for space-group classification, reporting approximately 25% Top-1 on a hand-filtered set of 942 RRUFF patterns restricted to 145 space groups.

The two studies differ on nearly every axis, making direct numeric comparison misleading:

•

Classification target: they predict space groups (145 retained classes); we predict extinction groups (all 99 classes, no exclusions).

•

Evaluation set: their 942-pattern set was hand-filtered; our RRUFF-473 benchmark is algorithmically curated and retains difficult patterns.

•

Data generation: they use ICSD-informed synthesis with strong chemical priors; we use a physics-first curriculum beginning from uniform, chemically random data.

•

Training scale: up to 261M diffractograms vs. our 5.14M.

On absolute Top-1, we do not yet match their reported $\sim$ 25%. On ranking breadth, our best Top-5 of 52.22% on RRUFF-473 is competitive given the data-scale difference. On our harshest RRUFF-325 benchmark, the final large mixed model reaches calibrated 10.46% Top-1 and 43.69% Top-5, reflecting the additional difficulty of strongly degraded mixtures without hand-filtering while still yielding broad candidate coverage.

A separate low-background 1222-pattern RRUFF holdout tells a complementary story. On this cleaner real-data set, the final large mixed model reached calibrated 10.07% Top-1 and 45.34% Top-5 in extinction-group space, while exact split-head validity rose to 65.30%, compared with 47.38% on the harsher RRUFF-325 benchmark. This indicates that the rule-based pathway is not simply broken on experimental data; rather, its strict symbolic decoding becomes substantially more usable as nuisance severity decreases, while the auxiliary head remains the more reliable path under stronger texture, overlap, and impurity corruption.

Cao et al. [2] recently introduced SimXRD-4M, a large-scale multiphysical simulator producing 4.07M diffractograms across 230 space groups, and reported strong transfer to experimental RRUFF data. Direct comparison is not straightforward for several reasons: the two studies use different classification targets (space groups vs. extinction groups), different RRUFF subsets (their $\sim$ 3000-pattern evaluation set vs. our algorithmically curated RRUFF-473/325), different intensity representations ( $d$ – $I$ vs. fixed $2\theta$ grid), and different normalization conventions. An exploratory cross-evaluation of our Stage-2c model on a locally reconstructed Cu-like RRUFF subset of 2738 patterns (see Supplemental Material) produced calibrated performance consistent with our main benchmarks (10.04% Top-1, 45.73% Top-5 in extinction-group space), but this is not a controlled head-to-head comparison. Because the two systems differ in benchmark definition, input representation ( $d$ – $I$ versus a fixed $2\theta$ grid), normalization conventions, and target taxonomy, a controlled comparison would require nontrivial harmonization and retraining that is beyond the scope of the present study.

Simple distribution-only baselines help contextualize these numbers. An ICSD-frequency prior—which never inspects a diffraction pattern—reaches about 9.7% Top-1 but only 27.2% Top-5 on RRUFF-like extinction labels. Our final large mixed model is only modestly above that prior on Top-1, but reaches 43.69% Top-5 on RRUFF-325 and 50.74% Top-5 on RRUFF-473, indicating that it is not merely reproducing a static class histogram: the model uses diffraction evidence to re-rank candidates on a per-pattern basis even when the strongest symmetry cues are partially destroyed by noise (see Supplemental Material for full analysis). Direct head-to-head comparison against recent larger-scale simulated-pretraining studies and multi-instrument experimental benchmarks remains future work, because the current study uses a different target taxonomy, harder algorithmically curated RRUFF subsets, and a single fixed Cu-K $\alpha$ laboratory geometry.

Recent generative approaches to PXRD analysis aim at the harder problem of full structure recovery. These methods are not directly comparable, but an extinction-group predictor of the type developed here could serve as an upstream constraint for such pipelines, reducing the combinatorial search space before structure generation.

## VIII Discussion

### VIII.1 Architecture Matters, But So Does Data

The scaling study demonstrates that transformers substantially outperform CNNs on this task. Yet even the best transformer fails on real data without the right training curriculum. The RRUFF fine-tuning stage alone accounts for the majority of the real-data performance gain (Fig. 2), underscoring that the synthetic-to-real domain gap is not merely a matter of model capacity.

### VIII.2 The Multi-Task Design Principle

The dual-head architecture embodies a general principle for scientific machine learning: separate the interpretable physics-driven prediction from the robust statistical prediction, train them jointly so each regularizes the other, and fuse at inference. This is preferable to either a pure physics-rules approach (too brittle for noisy real data) or a pure black-box classifier (no interpretability, no protection against prior collapse).

### VIII.3 Curriculum as Domain Adaptation

The training pathway can be understood as curriculum domain adaptation: uniform pretraining learns symmetry cues without geological frequency bias; RRUFF-conditioned fine-tuning adapts those representations to realistic nuisance structure; late preferred-orientation ablations show that subtractive noise must be represented explicitly; and calibrated Bayesian decoding restores empirical mineralogical prevalence at inference time without reintroducing uncontrolled overconfidence. The final large mixed run suggests that nuisance realism must itself be balanced: a small amount of exact PO sharpens ranking, but the best overall deployment model retains a majority of ordinary powder-like mixtures.

### VIII.4 Classical Baselines

Alongside the neural-network work, we developed a classical statistical baseline using sparse Pawley fitting for extinction-group ranking. Even with physically motivated information criteria (AIC/BIC), the unconstrained full-sweep classical selector was unstable: lower-symmetry models overfit profile imperfections, and accidental chemical absences mimic symmetry absences. A topology-guided conditional benchmark—ranking only candidates within the neural model’s bounded subgroup neighborhood—is both better posed and much faster in practice, converting a brittle multi-hour global sweep into bounded local verification that completes in tens of seconds to minutes on the supported branches.

In a broader supported non-monoclinic follow-up on 34 RRUFF-325 cases, all 34 bounded runs completed; 8/34 recovered the exact top-ranked space group, all in the rhombohedral branch. Primitive monoclinic remained the hardest regime, but after low-angle truncation, reflection clustering, and a nonnegative ridge/NNLS-style inner solve, a broader 61-case monoclinic follow-up completed 51 cases, showing that the bounded backend can be made practical even in the most difficult branch, albeit with local rather than yet exact rankings (see Supplemental Material).

### VIII.5 Limitations and Next Steps

Our best balanced harsh-benchmark result is the final large mixed model at 10.46% Top-1 and 43.69% Top-5 on the real RRUFF-325 benchmark, while a pure-PO ablation reaches the sharper but narrower 13.54% / 40.92% operating point. This is still far from solved symmetry assignment, and three factors appear to bound the current system. First, there is substantial label ambiguity. Even perfectly simulated patterns from some extinction groups become nearly indistinguishable after peak overlap, preferred orientation, and impurity structure compress the available evidence; the prior-only baseline’s 9.7% Top-1 suggests that achievable Top-1 on this distribution may be substantially constrained even for a strong model. The model’s main value on this benchmark is therefore not only exact Top-1, but also the large Top-5 gain over that frequency baseline.

Second, there is persistent simulator mismatch. Our synthetic generator now includes exact preferred orientation in the late ablation stage, and that intervention clearly changes the real-data error structure, but it still does not capture the full nuisance realism of field data, particularly specimen displacement, fluorescence, and related instrument-specific artifacts. The subgroup-DAG analysis shows that many of the remaining errors are local rather than random, suggesting that the model usually learns the correct crystallographic neighborhood but lacks sufficient nuisance fidelity to resolve it. In particular, the persistence of local branch jumps even after PO-aware training is consistent with a 1D “texture aliasing” limit, where suppression of whole reflection families makes nearby crystallographic cousins difficult to distinguish from a powder trace alone.

The empirical ceiling of $\sim$ 10% Top-1 for extinction-group classification on degraded real mixtures has implications beyond the present task. Since extinction-group determination requires only the binary presence or absence of reflections, whereas full structure solution requires accurate continuous intensities, the difficulty we observe even at the binary level suggests that end-to-end structure recovery from 1D powder data under comparable nuisance conditions faces at least comparable difficulty. These observations motivate modular pipelines in which a calibrated symmetry predictor constrains the search space for downstream structure-solution methods, whether classical (Rietveld) or generative (diffusion models, GNNs), rather than attempting unconstrained end-to-end inversion.

The most immediate next steps are therefore bounded rather than purely scaling-focused: (i) targeted improvements to nuisance realism rather than brute-force data scaling; (ii) broader real-data validation of the matched SG $\rightarrow$ EG control and related target-taxonomy comparisons; and (iii) tighter benchmark-level analyses of how calibration and structured decoding fail under specific nuisance regimes. The main lesson of the stage-2 experiments is not simply “more data helps” but rather that calibration and nuisance realism determine whether the information already learned by the model becomes usable.

A late control supports this interpretation. Simple weight-space interpolation (WiSE-FT) [21] between the broader Stage-2c checkpoint and the sharper PO-only checkpoint improved Top-1 relative to Stage-2c but did not recover the desired hybrid of PO-level Top-1 with Stage-2c-level Top-5. This suggests that the preferred-orientation effect is not merely a late linear interpolation in weight space, but instead depends on jointly learning texture and non-texture regimes during fine-tuning.

## IX Conclusions

We have shown that reliable symmetry classification from powder diffraction requires three co-designed ingredients: the measurement-compatible target (extinction groups), a physics-informed architecture that separates crystallographic rule learning from statistical pattern matching, and a training curriculum that explicitly manages the synthetic-to-real domain gap.

Two results carry implications beyond this specific PXRD task. First, post-hoc calibration is not a cosmetic final step but a central part of the scientific inference pipeline: on degraded real data, temperature-scaled Bayesian decoding substantially improves deployment performance without retraining by decoupling learned structural evidence from the geological prior absorbed during fine-tuning. Second, the subgroup-DAG analysis shows that the calibrated model’s residual errors are not random but physically structured. Preferred-orientation-aware training weakens the strong descendant bias of the non-PO model, while the final large mixed curriculum—combining standard and texture-augmented data—yields the tightest local topological error radius we observed. This graceful topological degradation emerges without explicit topological supervision, suggesting that calibrated inference can recover interpretable physical structure even from flexible neural classifiers.

These findings point toward a broader design pattern for scientific machine learning on spectroscopic and scattering data: encode the measurement physics in the architecture, let the training curriculum manage domain shift, and treat calibrated inference as a first-class design parameter rather than an afterthought. For powder diffraction specifically, the extinction-group framework and the RRUFF-473/RRUFF-325 benchmarks provide a foundation for community comparison, while the topology-guided classical verification pipeline shows how neural priors can convert an intractable global search into a bounded local problem. Attention is necessary but not sufficient: the physics must be in the architecture, the care must be in the data, and the final inference step must be calibrated to the deployment distribution.

## Data availability

Reproducibility materials for this study—including trained checkpoints, training and evaluation configurations, canonical evaluation wrappers, and the compact JSON artifacts underlying the benchmark, positional-ablation, calibration, and topological analyses reported here—are released through the public GitHub repository https://github.com/scattering/paper-ai-diffraction together with the Zenodo archival package https://doi.org/10.5281/zenodo.19558452. The release documents the required benchmark file names, expected local paths, and a reconstruction-oriented workflow for constructing the curated RRUFF-325 and RRUFF-473 benchmarks from the upstream RRUFF-derived sources.

###### Acknowledgements.

The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for resources made available to NIST under contract number 1333ND25PNB180410 that have contributed to the research results reported within this paper. URL: Texas Advanced Computing Center. Computational resources were also provided through ACCESS allocation PHY250007, “Applications of AI to Diffraction.” Support for Edward G. Friedman and Elizabeth Baggett was provided by the Center for High Resolution Neutron Scattering, a partnership between the National Institute of Standards and Technology and the National Science Foundation under Agreement No. DMR-2010792. We thank Brian DeCost, Austin McDannald, Craig Brown, and Hui Wu for useful conversations. We thank the RRUFF project at the University of Arizona for making their mineral diffraction database publicly available.

## Supplemental Material

## S1 CNN Baseline Scaling

Figure S1: Scaling ResNet-18 with synthetic reflection data. Performance plateaus beyond approximately 1M samples, in contrast to the continued improvement observed for transformers at the same data scales (see main-text Table III).

## S2 Bias and Cross-Distribution Studies

Table S1: Cross-evaluation of biased (ICSD-distributed) and balanced ResNet-18 models. The balanced model generalizes substantially better to the opposite distribution.

| Evaluation | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- |
| Biased model on biased test | 73.45% | 92.16% | 95.86% |
| Balanced model on balanced test | 57.09% | 81.31% | 89.30% |
| Biased model on balanced test | 38.68% | 62.85% | 76.19% |
| Balanced model on biased test | 54.00% | 81.26% | 89.75% |

## S3 Extinction-Group Mapping Table

For reproducibility, Table LABEL:tab:full_eg_map lists the full 99-extinction-group mapping used throughout this work. Each extinction group is identified by the same integer label used in training and evaluation, together with a canonical Hermann–Mauguin-style representative and the associated crystallographic space-group numbers.

Table S2: Full extinction-group lookup used in the 99-class formulation. Canonical labels are taken from the code-level lookup table, and the associated space-group numbers are the complete crystallographically indistinguishable sets under the powder-diffraction extinction conditions used here.

| EG | Canonical extinction group | Space-group numbers |
| --- | --- | --- |
| 1 | P - 1 1 (equiv: P 1 - 1, P 1 1 -) | 3, 6, 10 |
| 2 | P 21 1 1 (equiv: P 1 21 1, P 1 1 21) | 4, 11 |
| 3 | P b 1 1 (equiv: P c 1 1, P n 1 1, P 1 a 1, P 1 c 1, P 1 n 1, P 1 1 a, P 1 1 b, P 1 1 n) | 7, 13 |
| 4 | P 21/b 1 1 (equiv: P 21/c 1 1, P 21/n 1 1, P 1 21/a 1, P 1 21/c 1, P 1 21/n 1, P 1 1 21/a, P 1 1 21/b, P 1 1 21/n) | 14 |
| 5 | C - 1 1 (equiv: B - 1 1, I - 1 1, C 1 - 1, A 1 - 1, I 1 - 1, B 1 1 -, A 1 1 -, I 1 1 -) | 5, 8, 12 |
| 6 | C n 1 1 (equiv: B b 1 1, I c 1 1, C 1 c 1, A 1 n 1, I 1 a 1, B 1 1 n, A 1 1 a, I 1 1 b) | 9, 15 |
| 7 | P - - - | 16, 25, 47 |
| 8 | P - - 21 (equiv: P - 21 -, P 21 - -) | 17 |
| 9 | P - 21 21 (equiv: P 21 - 21, P 21 21 -) | 18 |
| 10 | P 21 21 21 | 19 |
| 11 | P - - a (equiv: P - - b, P - a -, P - c -, P b - -, P c - -) | 26, 28, 51 |
| 12 | P - - n (equiv: P - n -, P n - -) | 31, 59 |
| 13 | P - a a (equiv: P b - b, P c c -) | 27, 49 |
| 14 | P - a b (equiv: P - c a, P b - a, P b c -, P c - b, P c a -) | 29, 57 |
| 15 | P - a n (equiv: P - n a, P b - n, P c n -, P n - b, P n c -) | 30, 53 |
| 16 | P - c b (equiv: P b a -, P c - a) | 32, 55 |
| 17 | P - c n (equiv: P - n b, P b n -, P c - n, P n - a, P n a -) | 33, 62 |
| 18 | P - n n (equiv: P n - n, P n n -) | 34, 58 |
| 19 | P b a a (equiv: P b a b, P b c b, P c a a, P c c a, P c c b) | 54 |
| 20 | P b a n (equiv: P c n a, P n c b) | 50 |
| 21 | P b c a (equiv: P c a b) | 61 |
| 22 | P b c n (equiv: P b n a, P c a n, P c n b, P n a b, P n c a) | 60 |
| 23 | P b n b (equiv: P c c n, P n a a) | 56 |
| 24 | P b n n (equiv: P c n n, P n a n, P n c n, P n n a, P n n b) | 52 |
| 25 | P n n n | 48 |
| 26 | C - - - (equiv: B - - -, A - - -) | 21, 35, 38, 65 |
| 27 | C - - 21 (equiv: B - 21 -, A 21 - -) | 20 |
| 28 | C - - (ab) (equiv: B - (ac)-, A(bc)- -) | 39, 67 |
| 29 | C - c - (equiv: C c - -, B - - b, B b - -, A - - a, A - a -) | 36, 40, 63 |
| 30 | C - c (ab) (equiv: C c - (ab), B - (ac)b, B b (ac)-, A(bc)- a, A(bc)a -) | 41, 64 |
| 31 | C c c - (equiv: B b - b, A - a a) | 37, 66 |
| 32 | C c c (ab) (equiv: B b (ac)b, A(bc)a a) | 68 |
| 33 | I - - - | 23, 24, 44, 71 |
| 34 | I - - (ab) (equiv: I - (ac)-, I(bc)- -) | 46, 74 |
| 35 | I - c b (equiv: I c - a, I b a -) | 45, 72 |
| 36 | I b c a | 73 |
| 37 | F - - - | 22, 42, 69 |
| 38 | F - d d (equiv: F d - d, F d d -) | 43 |
| 39 | F d d d | 70 |
| 40 | P - - - | 75, 81, 83, 89, 99, 111, 115, 123 |
| 41 | P - 21 - | 90, 113 |
| 42 | P 42 - - | 77, 84, 93 |
| 43 | P 42 21 - | 94 |
| 44 | P 41 - - | 76, 78, 91, 95 |
| 45 | P 41 21 - | 92, 96 |
| 46 | P - - c | 105, 112, 131 |
| 47 | P - 21 c | 114 |
| 48 | P - b - | 100, 117, 127 |
| 49 | P - b c | 106, 135 |
| 50 | P - c - | 101, 116, 132 |
| 51 | P - c c | 103, 124 |
| 52 | P - n - | 102, 118, 136 |
| 53 | P - n c | 104, 128 |
| 54 | P n - - | 85, 129 |
| 55 | P 42/n - - | 86 |
| 56 | P n - c | 137 |
| 57 | P n b - | 125 |
| 58 | P n b c | 133 |
| 59 | P n c - | 138 |
| 60 | P n c c | 130 |
| 61 | P n n - | 134 |
| 62 | P n n c | 126 |
| 63 | I - - - | 79, 82, 87, 97, 107, 119, 121, 139 |
| 64 | I 41 - - | 80, 98 |
| 65 | I - - d | 109, 122 |
| 66 | I - c - | 108, 120, 140 |
| 67 | I - c d | 110 |
| 68 | I 41/a - - | 88 |
| 69 | I a - d | 141 |
| 70 | I a c d | 142 |
| 71 | P - - - | 143, 147, 149, 150, 156, 157, 162, 164, 168, 174, 175, 177, 183, 187, 189, 191 |
| 72 | P 31 - - | 144, 145, 151, 152, 153, 154 |
| 73 | P - - c | 159, 163, 186, 190, 194 |
| 74 | P - c - | 158, 165, 185, 188, 193 |
| 75 | R (obv) - - (equiv: R (rev) - -, R - - -) | 146, 148, 155, 160, 166 |
| 76 | R (obv)- - c (equiv: R (rev)- - c, R - - c) | 161, 167 |
| 77 | P 63 - - | 173, 176, 182 |
| 78 | P 62 - - | 171, 172, 180, 181 |
| 79 | P 61 - - | 169, 170, 178, 179 |
| 80 | P - c c | 184, 192 |
| 81 | P - - - | 195, 200, 207, 215, 221 |
| 82 | P 21 - - | 198 |
| 83 | P 42 - - | 208 |
| 84 | P 41 - - | 212, 213 |
| 85 | P - - n | 218, 223 |
| 86 | P a - - | 205 |
| 87 | P n - - | 201, 224 |
| 88 | P n - n | 222 |
| 89 | I - - - | 197, 199, 204, 211, 217, 229 |
| 90 | I 41 - - | 214 |
| 91 | I - - d | 220 |
| 92 | I a - - | 206 |
| 93 | I a - d | 230 |
| 94 | F - - - | 196, 202, 209, 216, 225 |
| 95 | F 41 - - | 210 |
| 96 | F - - c | 219, 226 |
| 97 | F d - - | 203, 227 |
| 98 | F d - c | 228 |
| 99 | P - | 1, 2 |

Table S3: Bias cross-evaluation for the vision transformer. The pattern is consistent: balanced training generalizes better to biased evaluation than vice versa.

| Model | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- |
| VT – Biased Reflection | 78.73% | 92.79% | 96.44% |
| VT – Biased PyXtal | 53.78% | 76.66% | 84.45% |
| VT – Balanced Refl. on Biased Test | 87.06% | 97.93% | 99.25% |
| VT – Biased Refl. on Balanced Test | 44.35% | 68.18% | 78.35% |
| VT – Balanced PyXtal on Biased Test | 50.49% | 75.44% | 84.51% |
| VT – Biased PyXtal on Balanced Test | 22.01% | 37.68% | 47.41% |

## S4 Ablation: Data Generation Method and $2\theta$ Range


<!-- truncated local extract; full PDF in same folder -->
