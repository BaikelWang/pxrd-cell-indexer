"""Training configuration (native yaml + dataclass, D18)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pxrd_cell_indexing.data.normalization import LatticeRepresentation
from pxrd_cell_indexing.losses import LossWeights

BestMetricName = Literal[
    "top1_lattice_match_proxy",
    "top1_lattice_match_rate",
    "top1_joint_match_rate",
    "crystal_system_accuracy",
    "composite",
    "strict_raw_top1_lattice_match_rate",
    "strict_raw_top1_elementwise_rate",
    "strict_composite",
]


@dataclass
class DataConfig:
    train_lmdb: str
    valid_lmdb: str
    train_jsonl: str
    valid_jsonl: str
    lattice_stats: str
    representation: LatticeRepresentation = "angles"
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    train_augment: bool = True
    valid_augment: bool = False
    # B3: 2θ shift range for spectrum augment (degrees); 0 disables peak-position jitter.
    augment_shift_range: float = 0.1
    # B2: upsample hard crystal systems in the train jsonl (1.0 = no upsample).
    hard_cs_upsample: float = 1.0
    hard_cs_names: tuple[str, ...] = ("hexagonal", "trigonal", "monoclinic", "triclinic")
    # A4 (v3 §8 / v4 §7): "legacy" = original shift+noise augment; "robust" =
    # auditable clean/perturb mix (global zero-shift, per-peak jitter, dropout,
    # impurity peaks, intensity noise, preferred orientation).
    augment_mode: Literal["legacy", "robust"] = "legacy"
    robust_clean_probability: float = 0.8
    robust_global_zero_shift_deg: float = 0.3
    robust_jitter_sigma_deg: float = 0.05
    robust_jitter_clip_deg: float = 0.15
    robust_dropout_max_count: int = 4
    robust_impurity_max_count: int = 2
    robust_impurity_intensity_frac_max: float = 0.2
    robust_intensity_noise_frac_min: float = 0.05
    robust_intensity_noise_frac_max: float = 0.10
    robust_preferred_orientation_max_peak_frac: float = 0.3
    robust_preferred_orientation_max_suppress_frac: float = 0.3


@dataclass
class ModelConfig:
    encoder_checkpoint: str | None = None
    freeze_encoder: bool = True
    normalize_embedding: bool = True
    hidden_dim: int = 256
    embedding_dim: int = 512
    """Encoder output / head input width (histogram ``output_dim``)."""
    dropout: float = 0.1
    position_encoding: Literal["discrete", "continuous", "physical"] = "discrete"
    """discrete = RealPXRD 2θ→long Embedding; continuous = float 2θ MLP (A3);
    physical = multi-channel peak tokens (inverse_d2/I), no separate pos embed."""
    encoder_type: Literal[
        "bert",
        "histogram",
        "histogram_fusion",
        "spectrum_fusion",
        "spectrum_only",
        "spectrum_cnn",
        "peak_transformer",
        "peak_hist_fusion",
        "peak_transformer_hist_fusion",
    ] = "bert"
    """bert / histogram / spectrum variants / peak_transformer / peak+hist fusion (A2-T4)."""
    peak_feature_mode: Literal[
        "legacy",
        "continuous_2theta_i",
        "reciprocal_d_i",
        "inverse_d2_i",
        "inverse_d2_logi",
        "inverse_d2_only",
    ] = "legacy"
    wavelength_angstrom: float = 1.54184
    intensity_transform: Literal["none", "linear", "sqrt", "log"] = "linear"
    hist_bins: int = 256
    sorted_peak_count: int = 24
    hist_pool: Literal["max", "sum"] = "max"
    histogram_hidden_dim: int = 512
    histogram_dropout: float = 0.0
    histogram_num_blocks: int = 0
    """R8: residual MLP blocks in histogram encoder. 0 = legacy shallow MLP."""
    spectrum_bins: int = 1024
    """R11b-E3: 2θ grid length for peak→profile reconstruction."""
    spectrum_sigma_deg: float = 0.15
    """Gaussian stick width (degrees) for spectrum reconstruction."""
    spectrum_cnn_channels: tuple[int, ...] = (64, 128, 256)
    spectrum_cnn_kernel: int = 7
    fusion_mode: Literal["gate", "concat"] = "gate"
    fusion_branch_dim: int = 256
    """A2-T4: per-branch embedding width before concat/gate fusion."""
    peak_transformer_max_peaks: int = 20
    peak_transformer_token_mode: Literal["pos", "pos_i", "geom"] = "geom"
    peak_transformer_d_model: int = 256
    peak_transformer_num_layers: int = 4
    peak_transformer_num_heads: int = 8
    peak_transformer_ffn_dim: int = 1024
    peak_transformer_dropout: float = 0.1
    peak_transformer_fourier_freqs: int = 16
    """Fourier frequencies on g=1/d²; 0 = legacy Linear-on-raw (P0-failing)."""
    peak_transformer_fourier_mode: Literal["linear", "log", "loglinear"] = "linear"
    """Fourier band on g: linear (abs res) | log (relative res) | loglinear (split)."""
    peak_transformer_g_floor: float = 1e-3
    """Lower clamp on normalized g for the log-Fourier band."""
    peak_transformer_pool: Literal["cls", "mean", "cls_mean", "attn", "cls_mean_max"] = "cls_mean"
    """Pooling after Transformer: cls | mean | 0.5*(cls+mean) | attn | cls⊕mean⊕max."""
    peak_transformer_rel_attn: Literal["none", "scalar", "mlp"] = "none"
    """A2.5-B1: relative attn bias φ(g_i−g_j); none|scalar|mlp."""
    peak_transformer_rel_freqs: int = 8
    """Fourier frequencies on Δg for rel_attn=mlp."""
    intensity_min: float = 5.0
    max_peaks: int | None = None
    warm_start_checkpoint: str | None = None
    """Optional indexing checkpoint for partial weight init (skips embed_positions)."""
    head_type: Literal["shared", "cs_conditional", "film"] = "shared"
    """shared = one lattice head; cs_conditional = 7 per-CS heads; film = FiLM shared (R10-H3)."""
    use_cs_classifier: bool = False
    """Train a light CS classifier for predicted routing at inference."""
    cs_route: Literal["oracle", "predicted"] = "oracle"
    """Valid/eval routing: oracle uses GT CS; predicted uses classifier argmax."""
    train_cs_route: Literal["oracle", "predicted"] = "oracle"
    """Train-time routing; P0/R2 default is teacher-forced oracle."""
    cubic_bravais_split: bool = False
    """R3: cubic → 3 setting heads (90/60/109.47)."""
    setting_route: Literal["oracle", "predicted", "classifier"] = "oracle"
    train_setting_route: Literal["oracle", "predicted", "classifier"] = "oracle"
    use_cubic_setting_classifier: bool = False
    """Train 3-way cubic setting classifier for deployable P/F/I routing."""
    hard_cs_finetune_epoch: int | None = None
    """If set, from this epoch freeze encoder+cubic heads; only train non-cubic heads."""
    multi_hypothesis: bool = False
    """R10: non-cubic cs_heads output num_hypotheses candidates (MCL training)."""
    num_hypotheses: int = 3
    """K for multi_hypothesis heads."""
    head_num_layers: int = 2
    """R8: MLP depth for lattice/CS/setting heads (2 = legacy)."""
    cubic_exact: bool = False
    """Hs1: exact a=b=c 1-DOF construction for cubic-routed samples (film head only)."""


@dataclass
class OptimConfig:
    head_lr: float = 1e-3
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: float = 1.0
    max_epochs: int = 5
    grad_clip: float = 1.0
    early_stop_patience: int | None = None
    accumulate_grad_batches: int = 1
    profile_timing: bool = False
    min_epochs: int = 1
    """Do not early-stop before this epoch (R9/R10 long fine-convergence)."""
    save_best_loss: bool = True
    """Also save checkpoints/best_valid_loss.pt by lowest valid loss."""


@dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "results/experiments"
    data: DataConfig = field(default_factory=DataConfig)  # type: ignore[arg-type]
    model: ModelConfig = field(default_factory=ModelConfig)  # type: ignore[arg-type]
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    best_metric: BestMetricName = "top1_lattice_match_rate"
    # Loose funnel (historical) vs strict north-star tolerances for valid metrics / selection.
    eval_ltol: float = 0.3
    eval_atol_deg: float = 10.0
    strict_ltol: float = 0.05
    strict_atol_deg: float = 3.0
    log_every: int = 20
    eval_every: int = 1
    # When False, skip pymatgen find_mapping (expensive / can hang on degenerate
    # cells). Funnel mapping rates fall back to elementwise; north-star elementwise
    # Gate is unaffected. Recommended for P0-700 overfits.
    eval_pymatgen_match: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        loss_raw = dict(raw.get("loss", {}))
        data_raw = dict(raw["data"])
        optim_raw = dict(raw.get("optim", {}))
        hard_cs = data_raw.pop("hard_cs_names", None)
        data_kwargs = {
            **data_raw,
            "representation": data_raw.get("representation", "angles"),
            "prefetch_factor": data_raw.get("prefetch_factor", 2),
            "persistent_workers": data_raw.get("persistent_workers", True),
            "augment_shift_range": float(data_raw.get("augment_shift_range", 0.1)),
            "hard_cs_upsample": float(data_raw.get("hard_cs_upsample", 1.0)),
        }
        if hard_cs is not None:
            data_kwargs["hard_cs_names"] = tuple(hard_cs)
        model_raw = dict(raw["model"])
        max_peaks = model_raw.get("max_peaks", None)
        if max_peaks is not None:
            max_peaks = int(max_peaks)
        model_kwargs = {
            **model_raw,
            "encoder_checkpoint": model_raw.get("encoder_checkpoint"),
            "position_encoding": model_raw.get("position_encoding", "discrete"),
            "encoder_type": model_raw.get("encoder_type", "bert"),
            "peak_feature_mode": model_raw.get("peak_feature_mode", "legacy"),
            "wavelength_angstrom": float(model_raw.get("wavelength_angstrom", 1.54184)),
            "intensity_transform": model_raw.get("intensity_transform", "linear"),
            "hist_bins": int(model_raw.get("hist_bins", 256)),
            "sorted_peak_count": int(model_raw.get("sorted_peak_count", 24)),
            "hist_pool": model_raw.get("hist_pool", "max"),
            "histogram_hidden_dim": int(model_raw.get("histogram_hidden_dim", 512)),
            "histogram_dropout": float(model_raw.get("histogram_dropout", 0.0)),
            "histogram_num_blocks": int(model_raw.get("histogram_num_blocks", 0)),
            "spectrum_bins": int(model_raw.get("spectrum_bins", 1024)),
            "spectrum_sigma_deg": float(model_raw.get("spectrum_sigma_deg", 0.15)),
            "spectrum_cnn_channels": tuple(
                int(c) for c in model_raw.get("spectrum_cnn_channels", (64, 128, 256))
            ),
            "spectrum_cnn_kernel": int(model_raw.get("spectrum_cnn_kernel", 7)),
            "fusion_mode": model_raw.get("fusion_mode", "gate"),
            "fusion_branch_dim": int(model_raw.get("fusion_branch_dim", 256)),
            "peak_transformer_max_peaks": int(model_raw.get("peak_transformer_max_peaks", 20)),
            "peak_transformer_token_mode": model_raw.get("peak_transformer_token_mode", "geom"),
            "peak_transformer_d_model": int(model_raw.get("peak_transformer_d_model", 256)),
            "peak_transformer_num_layers": int(model_raw.get("peak_transformer_num_layers", 4)),
            "peak_transformer_num_heads": int(model_raw.get("peak_transformer_num_heads", 8)),
            "peak_transformer_ffn_dim": int(model_raw.get("peak_transformer_ffn_dim", 1024)),
            "peak_transformer_dropout": float(model_raw.get("peak_transformer_dropout", 0.1)),
            "peak_transformer_fourier_freqs": int(model_raw.get("peak_transformer_fourier_freqs", 16)),
            "peak_transformer_fourier_mode": model_raw.get("peak_transformer_fourier_mode", "linear"),
            "peak_transformer_g_floor": float(model_raw.get("peak_transformer_g_floor", 1e-3)),
            "peak_transformer_pool": model_raw.get("peak_transformer_pool", "cls_mean"),
            "peak_transformer_rel_attn": model_raw.get("peak_transformer_rel_attn", "none"),
            "peak_transformer_rel_freqs": int(model_raw.get("peak_transformer_rel_freqs", 8)),
            "intensity_min": float(model_raw.get("intensity_min", 5.0)),
            "max_peaks": max_peaks,
            "warm_start_checkpoint": model_raw.get("warm_start_checkpoint"),
            "head_type": model_raw.get("head_type", "shared"),
            "use_cs_classifier": bool(model_raw.get("use_cs_classifier", False)),
            "cs_route": model_raw.get("cs_route", "oracle"),
            "train_cs_route": model_raw.get("train_cs_route", "oracle"),
            "cubic_bravais_split": bool(model_raw.get("cubic_bravais_split", False)),
            "setting_route": model_raw.get("setting_route", "oracle"),
            "train_setting_route": model_raw.get("train_setting_route", "oracle"),
            "use_cubic_setting_classifier": bool(
                model_raw.get("use_cubic_setting_classifier", False)
            ),
            "hard_cs_finetune_epoch": (
                None
                if model_raw.get("hard_cs_finetune_epoch") is None
                else int(model_raw["hard_cs_finetune_epoch"])
            ),
            "multi_hypothesis": bool(model_raw.get("multi_hypothesis", False)),
            "num_hypotheses": int(model_raw.get("num_hypotheses", 3)),
            "head_num_layers": int(model_raw.get("head_num_layers", 2)),
            "cubic_exact": bool(model_raw.get("cubic_exact", False)),
            "embedding_dim": int(model_raw.get("embedding_dim", 512)),
            "hidden_dim": int(model_raw.get("hidden_dim", 256)),
            "dropout": float(model_raw.get("dropout", 0.1)),
        }
        return cls(
            experiment_name=raw["experiment_name"],
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
            output_dir=raw.get("output_dir", "results/experiments"),
            data=DataConfig(**data_kwargs),
            model=ModelConfig(**model_kwargs),
            optim=OptimConfig(
                **{
                    **optim_raw,
                    "accumulate_grad_batches": optim_raw.get("accumulate_grad_batches", 1),
                    "profile_timing": optim_raw.get("profile_timing", False),
                    "min_epochs": int(optim_raw.get("min_epochs", 1)),
                    "save_best_loss": bool(optim_raw.get("save_best_loss", True)),
                }
            ),
            loss=LossWeights(
                regression=loss_raw.get("regression", 1.0),
                mode=loss_raw.get("mode", "baseline"),
                length_weight=loss_raw.get("length_weight", 1.0),
                angle_weight=loss_raw.get("angle_weight", 1.0),
                physical_weight=loss_raw.get("physical_weight", 1.0),
                huber_delta=loss_raw.get("huber_delta", 5.0),
                hinge_ltol=float(loss_raw.get("hinge_ltol", 0.05)),
                hinge_atol_deg=float(loss_raw.get("hinge_atol_deg", 3.0)),
                hinge_length_weight=float(loss_raw.get("hinge_length_weight", 1.0)),
                hinge_angle_weight=float(loss_raw.get("hinge_angle_weight", 1.0)),
                angle_prior_weight=float(loss_raw.get("angle_prior_weight", 0.25)),
                classification=float(loss_raw.get("classification", 0.0)),
                setting_classification=float(loss_raw.get("setting_classification", 0.0)),
                manifold_consistency_weight=float(
                    loss_raw.get("manifold_consistency_weight", 0.1)
                ),
                peak_consistency_weight=float(
                    loss_raw.get("peak_consistency_weight", 0.15)
                ),
                peak_consistency_scale=float(
                    loss_raw.get("peak_consistency_scale", 1000.0)
                ),
                peak_consistency_max_hkl=int(
                    loss_raw.get("peak_consistency_max_hkl", 4)
                ),
                peak_consistency_n_lines=int(
                    loss_raw.get("peak_consistency_n_lines", 20)
                ),
                wavelength_angstrom=float(
                    loss_raw.get(
                        "wavelength_angstrom",
                        model_raw.get("wavelength_angstrom", 1.54184),
                    )
                ),
                soft_strict_tau=float(loss_raw.get("soft_strict_tau", 0.5)),
            ),
            best_metric=raw.get("best_metric", "top1_lattice_match_rate"),
            eval_ltol=float(raw.get("eval_ltol", 0.3)),
            eval_atol_deg=float(raw.get("eval_atol_deg", 10.0)),
            strict_ltol=float(raw.get("strict_ltol", 0.05)),
            strict_atol_deg=float(raw.get("strict_atol_deg", 3.0)),
            log_every=raw.get("log_every", 20),
            eval_every=raw.get("eval_every", 1),
            eval_pymatgen_match=bool(raw.get("eval_pymatgen_match", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    def resolve_paths(self, project_root: str | Path) -> TrainConfig:
        """Resolve relative data/model/output paths against project root."""
        root = Path(project_root)
        data = self.data
        for field_name in (
            "train_lmdb",
            "valid_lmdb",
            "train_jsonl",
            "valid_jsonl",
            "lattice_stats",
        ):
            value = getattr(data, field_name)
            path = Path(value)
            if not path.is_absolute():
                setattr(data, field_name, str((root / path).resolve()))
        if self.model.encoder_checkpoint and not Path(self.model.encoder_checkpoint).is_absolute():
            self.model.encoder_checkpoint = str(
                (root / self.model.encoder_checkpoint).resolve()
            )
        if self.model.warm_start_checkpoint and not Path(
            self.model.warm_start_checkpoint
        ).is_absolute():
            self.model.warm_start_checkpoint = str(
                (root / self.model.warm_start_checkpoint).resolve()
            )
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str((root / self.output_dir).resolve())
        return self
