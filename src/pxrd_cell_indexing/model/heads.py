"""Model heads and full indexing model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from pxrd_cell_indexing.model.encoder.bert import BertModel
from pxrd_cell_indexing.model.encoder.histogram import InverseD2HistogramEncoder
from pxrd_cell_indexing.model.encoder.peak_transformer import PeakGeometryTransformerEncoder
from pxrd_cell_indexing.model.encoder.peak_hist_fusion import PeakTransformerHistogramFusionEncoder
from pxrd_cell_indexing.model.encoder.spectrum_fusion import (
    HistogramSpectrumFusionEncoder,
    SpectrumOnlyEncoder,
)
from pxrd_cell_indexing.model.encoder.loader import (
    DEFAULT_ENCODER_CONFIG,
    build_bert_model,
    load_xrd_encoder_from_checkpoint,
)
from pxrd_cell_indexing.model.bravais_setting import (
    CUBIC_SETTING_TARGETS_DEG,
    N_CUBIC_SETTINGS,
    cubic_setting_idx_from_phys,
    pick_cubic_setting_self_consistent,
)
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

HeadType = Literal["shared", "cs_conditional", "film"]
CsRoute = Literal["oracle", "predicted"]
SettingRoute = Literal["oracle", "predicted", "classifier"]
N_CRYSTAL_SYSTEMS = len(CRYSTAL_SYSTEMS)
_CS_CUBIC = 0


@dataclass(frozen=True)
class HeadConfig:
    embedding_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    output_dim: int = 6
    """Regression target width: 6 for (a,b,c,alpha,beta,gamma)/matrix6 canonical
    components, 9 for unconstrained full 3x3 matrix regression (Decision B, true
    matrix9 variant)."""
    head_type: HeadType = "shared"
    """shared = one lattice head; cs_conditional = 7 per-CS heads;
    film = shared deep lattice MLP with FiLM from CS index (R10-H3)."""
    use_cs_classifier: bool = False
    """If True, predict CS logits (routing for cs_conditional/film; aux CE for shared)."""
    default_cs_route: CsRoute = "oracle"
    """Default routing when forward(..., cs_route=None). Train P0 uses oracle."""
    cubic_bravais_split: bool = False
    """R3: replace cubic head with 3 setting heads (90/60/109.47)."""
    default_setting_route: SettingRoute = "oracle"
    use_cubic_setting_classifier: bool = False
    """If True with cubic_bravais_split, predict P/F/I setting for deployable routing."""
    multi_hypothesis: bool = False
    """R10: non-cubic cs_heads output num_hypotheses candidates instead of one
    (Multiple-Choice-Learning). Cubic keeps its existing setting-classifier split
    (K=1) so the 89% cubic elementwise ceiling isn't disturbed."""
    num_hypotheses: int = 3
    """K for multi_hypothesis heads (ignored when multi_hypothesis=False)."""
    head_num_layers: int = 2
    """R8: depth of lattice / CS / setting MLPs. 2 = legacy (Linear→GELU→Dropout→Linear);
    ≥3 inserts extra hidden GELU layers of width ``hidden_dim``."""
    cubic_exact: bool = False
    """Hs1: for cubic-routed samples, replace the free matrix6 regression with an
    exact 1-DOF construction a=b=c, α=β=γ∈{90,60,109.47}. In the Niggli-reduced
    basis a=b=c holds for 100% of cubic samples, so tying the three lengths to a
    single predicted scalar guarantees they pass/fail the strict length check
    together. Only valid with head_type='film' and matrix6 representation."""


def _mlp(in_dim: int, hidden: int, out_dim: int, *, dropout: float, num_layers: int) -> nn.Sequential:
    """Build a small MLP with ``num_layers`` linear maps (min 2)."""
    n = max(int(num_layers), 2)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(n - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class LatticeRegressionHead(nn.Module):
    """Predict lattice regression targets in normalized space (width = config.output_dim)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            config.output_dim,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class FiLMLatticeHead(nn.Module):
    """Shared deep lattice MLP with FiLM conditioning on crystal-system index (R10-H3).

    Stem maps embedding → hidden; CS embedding produces (γ, β); modulated features
    go through remaining depth then to ``output_dim``. One set of regression weights
    for all crystal systems.
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        dropout = config.dropout
        n = max(int(config.head_num_layers), 2)
        self.stem = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cs_embed = nn.Embedding(N_CRYSTAL_SYSTEMS, hidden)
        self.film = nn.Linear(hidden, 2 * hidden)
        body: list[nn.Module] = []
        for _ in range(n - 2):
            body.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        self.body = nn.Sequential(*body) if body else nn.Identity()
        self.out = nn.Linear(hidden, config.output_dim)

    def forward(self, embedding: torch.Tensor, cs_idx: torch.Tensor) -> torch.Tensor:
        h = self.stem(embedding)
        gamma, beta = self.film(self.cs_embed(cs_idx.long())).chunk(2, dim=-1)
        h = gamma * h + beta
        h = self.body(h)
        return self.out(h)


def _cubic_unit_matrix6() -> torch.Tensor:
    """Canonical matrix6 components for a=b=c=1 at each cubic setting angle.

    Returns ``[N_CUBIC_SETTINGS, 6]``. A cubic reduced cell scales linearly with
    the edge length L, so the physical matrix6 for any cubic sample is simply
    ``L * unit[setting]`` — this is what makes the exact 1-DOF construction valid.
    """
    cells = torch.tensor(
        [[1.0, 1.0, 1.0, t, t, t] for t in CUBIC_SETTING_TARGETS_DEG],
        dtype=torch.float64,
    )
    m = lattice_params_to_matrix(cells)  # [N, 3, 3]
    comp = torch.stack(
        [m[..., 0, 0], m[..., 0, 2], m[..., 1, 0], m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]],
        dim=-1,
    )
    return comp.to(torch.float32)  # [N, 6]


class CubicExactHead(nn.Module):
    """Hs1: predict cubic edge length (log-scale) + 3-way setting logits.

    The actual matrix6 is assembled in ``IndexingModel.forward`` as
    ``exp(log_L) * unit_matrix6[setting]`` so a=b=c and the discrete angle are
    exact by construction.
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.len_net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            1,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )
        self.setting_net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            N_CUBIC_SETTINGS,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )
        # Bias the length toward log(~5 Å) so early training starts near the
        # typical cubic edge instead of L≈1 Å.
        nn.init.constant_(self.len_net[-1].bias, 1.6)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_l = self.len_net(embedding).squeeze(-1)  # [B]
        setting_logits = self.setting_net(embedding)  # [B, N_CUBIC_SETTINGS]
        return log_l, setting_logits


class MultiHypothesisLatticeHead(nn.Module):
    """R10: predict K candidate lattice regression targets per embedding.

    Output shape ``[..., num_hypotheses, output_dim]``. Used for the non-cubic
    ``cs_heads`` slots when ``HeadConfig.multi_hypothesis`` is set, trained with
    Multiple-Choice-Learning (``losses.mcl_min_loss``): only the closest-to-truth
    hypothesis gets gradient per sample, so the K heads can specialize on the
    different plausible local solutions a single PXRD pattern admits.
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.num_hypotheses = max(int(config.num_hypotheses), 1)
        self.output_dim = config.output_dim
        self.net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            self.output_dim * self.num_hypotheses,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        out = self.net(embedding)
        return out.view(*out.shape[:-1], self.num_hypotheses, self.output_dim)


class CrystalSystemHead(nn.Module):
    """Light 7-way crystal-system classifier for head routing (not Top-K)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            N_CRYSTAL_SYSTEMS,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class CubicSettingHead(nn.Module):
    """3-way cubic Bravais setting classifier (90 / 60 / 109.47)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = _mlp(
            config.embedding_dim,
            config.hidden_dim,
            N_CUBIC_SETTINGS,
            dropout=config.dropout,
            num_layers=config.head_num_layers,
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class IndexingModel(nn.Module):
    """Encoder + lattice regression (shared or CS-conditional)."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        head_config: HeadConfig | None = None,
        freeze_encoder: bool = False,
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head_config = head_config or HeadConfig()
        self.normalize_embedding = normalize_embedding
        self.head_type: HeadType = self.head_config.head_type
        self.default_cs_route: CsRoute = self.head_config.default_cs_route
        self.cubic_bravais_split = bool(self.head_config.cubic_bravais_split)
        self.default_setting_route: SettingRoute = self.head_config.default_setting_route
        self.multi_hypothesis = bool(self.head_config.multi_hypothesis)
        self._normalizer = None  # optional; required for predicted cubic setting route
        self.cubic_exact_head: nn.Module | None = None

        if self.head_type == "shared":
            self.lattice_head = LatticeRegressionHead(self.head_config)
            self.cs_heads = None
            self.cubic_setting_heads = None
            self.cubic_setting_classifier = None
            self.cs_classifier = (
                CrystalSystemHead(self.head_config)
                if self.head_config.use_cs_classifier
                else None
            )
        elif self.head_type == "film":
            self.lattice_head = FiLMLatticeHead(self.head_config)
            self.cs_heads = None
            self.cubic_setting_heads = None
            self.cubic_setting_classifier = None
            self.cs_classifier = (
                CrystalSystemHead(self.head_config)
                if self.head_config.use_cs_classifier
                else None
            )
            if self.head_config.cubic_exact:
                self.cubic_exact_head = CubicExactHead(self.head_config)
                self.register_buffer(
                    "_cubic_unit_m6", _cubic_unit_matrix6(), persistent=False
                )
        else:
            self.lattice_head = None
            if self.multi_hypothesis:
                # Cubic (idx 0) keeps its existing single-point head + setting
                # split; every other crystal system gets K trainable hypotheses
                # (R10).
                self.cs_heads = nn.ModuleList(
                    [
                        LatticeRegressionHead(self.head_config)
                        if idx == _CS_CUBIC
                        else MultiHypothesisLatticeHead(self.head_config)
                        for idx in range(N_CRYSTAL_SYSTEMS)
                    ]
                )
            else:
                self.cs_heads = nn.ModuleList(
                    [LatticeRegressionHead(self.head_config) for _ in range(N_CRYSTAL_SYSTEMS)]
                )
            self.cs_classifier = (
                CrystalSystemHead(self.head_config)
                if self.head_config.use_cs_classifier
                else None
            )
            self.cubic_setting_heads = (
                nn.ModuleList(
                    [LatticeRegressionHead(self.head_config) for _ in range(N_CUBIC_SETTINGS)]
                )
                if self.cubic_bravais_split
                else None
            )
            self.cubic_setting_classifier = (
                CubicSettingHead(self.head_config)
                if (
                    self.cubic_bravais_split
                    and self.head_config.use_cubic_setting_classifier
                )
                else None
            )
        self.set_encoder_trainable(not freeze_encoder)

    def set_normalizer(self, normalizer) -> None:
        """Attach lattice normalizer for predicted cubic-setting self-consistency."""
        self._normalizer = normalizer

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def set_cubic_heads_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze cubic CS head and cubic setting heads (hard-CS finetune)."""
        if self.cs_heads is not None:
            for p in self.cs_heads[_CS_CUBIC].parameters():
                p.requires_grad = trainable
        if self.cubic_setting_heads is not None:
            for head in self.cubic_setting_heads:
                for p in head.parameters():
                    p.requires_grad = trainable

    def set_non_cubic_heads_trainable(self, trainable: bool) -> None:
        if self.cs_heads is None:
            return
        for idx, head in enumerate(self.cs_heads):
            if idx == _CS_CUBIC:
                continue
            for p in head.parameters():
                p.requires_grad = trainable

    @property
    def encoder_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.lattice_head is not None:
            params.extend(self.lattice_head.parameters())
        if self.cs_heads is not None:
            params.extend(self.cs_heads.parameters())
        if self.cs_classifier is not None:
            params.extend(self.cs_classifier.parameters())
        if self.cubic_setting_heads is not None:
            params.extend(self.cubic_setting_heads.parameters())
        if self.cubic_setting_classifier is not None:
            params.extend(self.cubic_setting_classifier.parameters())
        if self.cubic_exact_head is not None:
            params.extend(self.cubic_exact_head.parameters())
        return [p for p in params if p.requires_grad]

    def encode(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encoder(pxrd_x, pxrd_y, peak_num)
        if self.normalize_embedding:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def _route_cs_idx(
        self,
        embedding: torch.Tensor,
        *,
        crystal_system_idx: torch.Tensor | None,
        cs_route: CsRoute,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.cs_classifier is not None:
            logits = self.cs_classifier(embedding)
            outputs["crystal_system_logits"] = logits
        if cs_route == "oracle":
            if crystal_system_idx is None:
                raise ValueError("cs_route='oracle' requires crystal_system_idx")
            return crystal_system_idx.long()
        if self.cs_classifier is None:
            raise ValueError("cs_route='predicted' requires use_cs_classifier=True")
        return outputs["crystal_system_logits"].argmax(dim=-1)

    def _apply_cubic_exact(
        self,
        embedding: torch.Tensor,
        lattice_norm: torch.Tensor,
        *,
        routed_idx: torch.Tensor,
        lattice_phys: torch.Tensor | None,
        setting_route: SettingRoute | None,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Hs1: overwrite cubic-routed rows with an exact a=b=c construction."""
        assert self.cubic_exact_head is not None
        if self._normalizer is None or not hasattr(self._normalizer, "component_mean"):
            raise ValueError("cubic_exact requires a matrix6 normalizer via set_normalizer(...)")
        log_l, setting_logits = self.cubic_exact_head(embedding)
        outputs["cubic_setting_logits"] = setting_logits
        cubic_mask = routed_idx == _CS_CUBIC
        if not bool(cubic_mask.any()):
            return lattice_norm
        s_route = setting_route or self.default_setting_route
        if s_route == "oracle":
            if lattice_phys is None:
                raise ValueError("cubic_exact setting_route='oracle' requires lattice_phys")
            setting_idx = cubic_setting_idx_from_phys(lattice_phys)
        else:
            setting_idx = setting_logits.argmax(dim=-1)
        unit = self._cubic_unit_m6.to(dtype=lattice_norm.dtype)  # [S, 6]
        length = torch.exp(log_l).unsqueeze(-1)  # [B, 1]
        m6_phys = length * unit[setting_idx]  # [B, 6]
        mean = self._normalizer._mean_tensor(device=m6_phys.device, dtype=m6_phys.dtype)
        std = self._normalizer._std_tensor(device=m6_phys.device, dtype=m6_phys.dtype)
        m6_norm = (m6_phys - mean) / std
        lattice_norm = lattice_norm.clone()
        lattice_norm[cubic_mask] = m6_norm[cubic_mask]
        return lattice_norm

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
        crystal_system_idx: torch.Tensor | None = None,
        cs_route: CsRoute | None = None,
        lattice_phys: torch.Tensor | None = None,
        setting_route: SettingRoute | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding = self.encode(pxrd_x, pxrd_y, peak_num)
        outputs: dict[str, torch.Tensor] = {"embedding": embedding}

        if self.head_type == "shared":
            assert self.lattice_head is not None
            outputs["lattice_norm"] = self.lattice_head(embedding)
            if self.cs_classifier is not None:
                outputs["crystal_system_logits"] = self.cs_classifier(embedding)
            return outputs

        if self.head_type == "film":
            assert isinstance(self.lattice_head, FiLMLatticeHead)
            route = cs_route or self.default_cs_route
            routed_idx = self._route_cs_idx(
                embedding,
                crystal_system_idx=crystal_system_idx,
                cs_route=route,
                outputs=outputs,
            )
            lattice_norm = self.lattice_head(embedding, routed_idx)
            if self.cubic_exact_head is not None:
                lattice_norm = self._apply_cubic_exact(
                    embedding,
                    lattice_norm,
                    routed_idx=routed_idx,
                    lattice_phys=lattice_phys,
                    setting_route=setting_route,
                    outputs=outputs,
                )
            outputs["lattice_norm"] = lattice_norm
            outputs["routed_cs_idx"] = routed_idx
            return outputs

        assert self.cs_heads is not None
        if self.multi_hypothesis:
            # Heterogeneous K per CS (cubic K=1, others K=num_hypotheses): pad the
            # single-hypothesis heads by repeating hyp-0 so everything stacks into
            # one dense [B, 7, max_K, D] tensor. Padding with a duplicate is a
            # no-op for MCL (min over identical copies == the single value).
            raw = [head(embedding) for head in self.cs_heads]
            raw = [r if r.ndim == 3 else r.unsqueeze(1) for r in raw]  # [B, K_i, D]
            max_k = max(r.shape[1] for r in raw)
            padded = []
            for r in raw:
                k = r.shape[1]
                if k < max_k:
                    r = torch.cat([r, r[:, :1, :].expand(-1, max_k - k, -1)], dim=1)
                padded.append(r)
            hyp_all = torch.stack(padded, dim=1)  # [B, 7, max_K, D]
            outputs["lattice_hyp_all"] = hyp_all
            all_preds = hyp_all[:, :, 0, :]  # primary (hyp-0) per CS, backward-compatible [B, 7, D]
        else:
            # [B, 7, D]
            all_preds = torch.stack([head(embedding) for head in self.cs_heads], dim=1)
        outputs["lattice_norm_all"] = all_preds
        route = cs_route or self.default_cs_route
        routed_idx = self._route_cs_idx(
            embedding,
            crystal_system_idx=crystal_system_idx,
            cs_route=route,
            outputs=outputs,
        )
        batch_idx = torch.arange(all_preds.shape[0], device=all_preds.device)
        lattice_norm = all_preds[batch_idx, routed_idx]
        if self.multi_hypothesis:
            # Own-(routed)-CS hypothesis set: at train time cs_route="oracle" so
            # this is exactly the ground-truth CS's K candidates (what the MCL
            # loss needs); at eval it follows whatever route is configured.
            outputs["lattice_hyp_selected"] = outputs["lattice_hyp_all"][batch_idx, routed_idx]

        if self.cubic_setting_heads is not None:
            cubic_all = torch.stack(
                [head(embedding) for head in self.cubic_setting_heads], dim=1
            )
            outputs["cubic_setting_norm_all"] = cubic_all
            if self.cubic_setting_classifier is not None:
                outputs["cubic_setting_logits"] = self.cubic_setting_classifier(embedding)
            cubic_mask = routed_idx == _CS_CUBIC
            if cubic_mask.any():
                s_route = setting_route or self.default_setting_route
                if s_route == "oracle":
                    if lattice_phys is None:
                        raise ValueError(
                            "setting_route='oracle' requires lattice_phys for cubic split"
                        )
                    setting_idx = cubic_setting_idx_from_phys(lattice_phys)
                elif s_route == "classifier":
                    if "cubic_setting_logits" not in outputs:
                        raise ValueError(
                            "setting_route='classifier' requires use_cubic_setting_classifier"
                        )
                    setting_idx = outputs["cubic_setting_logits"].argmax(dim=-1)
                else:
                    if self._normalizer is None:
                        raise ValueError(
                            "predicted cubic setting route requires model.set_normalizer(...)"
                        )
                    cubic_phys = self._normalizer.denormalize(
                        cubic_all.reshape(-1, cubic_all.shape[-1])
                    ).reshape(cubic_all.shape[0], N_CUBIC_SETTINGS, -1)
                    setting_idx = pick_cubic_setting_self_consistent(cubic_phys)
                outputs["routed_cubic_setting_idx"] = setting_idx
                lattice_norm = lattice_norm.clone()
                lattice_norm[cubic_mask] = cubic_all[batch_idx[cubic_mask], setting_idx[cubic_mask]]

        outputs["lattice_norm"] = lattice_norm
        outputs["routed_cs_idx"] = routed_idx
        return outputs


def _encoder_config_from_kwargs(encoder_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_ENCODER_CONFIG)
    if encoder_config:
        cfg.update(encoder_config)
    return cfg


def build_indexing_model(
    *,
    checkpoint_path: str | None = None,
    encoder_config: dict[str, Any] | None = None,
    head_config: HeadConfig | None = None,
    freeze_encoder: bool = False,
    normalize_embedding: bool = True,
) -> IndexingModel:
    """Construct IndexingModel with optional pretrained encoder weights."""
    cfg = _encoder_config_from_kwargs(encoder_config)
    encoder_type = str(cfg.pop("encoder_type", "bert"))
    if encoder_type == "histogram":
        # Histogram encoder is always randomly initialized (no RealPXRD weights).
        encoder = InverseD2HistogramEncoder(cfg)
    elif encoder_type == "peak_transformer":
        encoder = PeakGeometryTransformerEncoder(cfg)
    elif encoder_type in ("peak_hist_fusion", "peak_transformer_hist_fusion"):
        # A2-T4: Peak Transformer ⊕ inverse_d2 histogram (concat/gate).
        encoder = PeakTransformerHistogramFusionEncoder(cfg)
    elif encoder_type in ("histogram_fusion", "spectrum_fusion"):
        # R11b-E3: peak histogram + peak-reconstructed spectrum CNN fusion.
        encoder = HistogramSpectrumFusionEncoder(cfg)
    elif encoder_type in ("spectrum_only", "spectrum_cnn"):
        # Diagnostic: spectrum CNN only (no peak histogram branch).
        encoder = SpectrumOnlyEncoder(cfg)
    elif checkpoint_path is None:
        encoder = build_bert_model(cfg)
    else:
        # Physical peak tokens change embed_tokens input dim; load non-strict.
        encoder, _ = load_xrd_encoder_from_checkpoint(
            checkpoint_path,
            config=cfg,
            strict=False,
        )
    return IndexingModel(
        encoder,
        head_config=head_config,
        freeze_encoder=freeze_encoder,
        normalize_embedding=normalize_embedding,
    )


def load_warm_start_state_dict(
    model: IndexingModel,
    checkpoint_path: str | Path,
    *,
    skip_key_substrings: tuple[str, ...] = ("embed_positions",),
    map_location: str | torch.device = "cpu",
) -> dict[str, int]:
    """Load overlapping weights from an indexing checkpoint, skipping renamed modules."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"No model_state_dict in {checkpoint_path}")
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped = 0
    shape_mismatch = 0
    for key, value in state.items():
        if any(part in key for part in skip_key_substrings):
            skipped += 1
            continue
        if key not in model_state:
            skipped += 1
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch += 1
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped": skipped,
        "shape_mismatch": shape_mismatch,
        "missing_after": len(missing),
        "unexpected_after": len(unexpected),
    }
