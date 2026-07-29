"""Arm A / B / C model definitions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from pxrd_cell_indexing.model.flow_head import ConditionalFlowHead


class LatticeMLP(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BertFourierPos(nn.Module):
    """BertModel clone with continuous Fourier(2θ/g) position encoding instead of int Embedding."""

    def __init__(self, pretrained=None, *arg, **kwargs):
        super().__init__()
        from app.model.bert import DotDict
        from app.model.transformer.transformer_encoder import (
            TransformerEncoder,
            init_bert_params,
        )

        args = DotDict(**kwargs)
        self.padding_idx = 0
        self.embed_dim = args.encoder_embed_dim

        self.embed_tokens = nn.Sequential(
            nn.Linear(1, args.encoder_embed_dim),
            nn.LayerNorm(args.encoder_embed_dim),
            nn.ReLU(True),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.vnode_encoder = nn.Embedding(1, args.encoder_embed_dim)
        # Keep attribute for state_dict compatibility (unused)
        self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)

        n_fourier = 16
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(64.0), n_fourier))
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        self.pos_proj = nn.Sequential(
            nn.Linear(2 * n_fourier, args.encoder_embed_dim),
            nn.LayerNorm(args.encoder_embed_dim),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )

        self.sentence_encoder = TransformerEncoder(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            rel_pos=False,
            rel_pos_bins=320,
            max_rel_pos=1280,
            post_ln=args.post_ln,
        )
        self.apply(init_bert_params)
        self.out = nn.Linear(args.encoder_embed_dim, args.output_dim)

    def batch_input(self, pxrd_x, pxrd_y, peak_num):
        """Keep continuous 2θ (float), intensity float. No .long() on angles."""
        max_peak_num = int(peak_num.max().item())
        batch_peak_x = torch.zeros(
            (peak_num.shape[0], max_peak_num, 1), device=pxrd_x.device, dtype=pxrd_x.dtype
        )
        batch_peak_y = torch.zeros(
            (peak_num.shape[0], max_peak_num, 1), device=pxrd_y.device, dtype=pxrd_y.dtype
        )
        idx = 0
        for i in range(len(peak_num)):
            n = int(peak_num[i].item())
            batch_peak_x[i, :n] = pxrd_x[idx : idx + n]
            batch_peak_y[i, :n] = pxrd_y[idx : idx + n]
            idx += n
        return batch_peak_x, batch_peak_y

    def _fourier_pos(self, two_theta: torch.Tensor) -> torch.Tensor:
        # two_theta: [B, L, 1] degrees → g ∝ sin²(θ)
        theta = two_theta.squeeze(-1) * (math.pi / 360.0)  # half-angle in rad
        g = torch.sin(theta).clamp(min=1e-6).pow(2)
        # normalize roughly to [0,1] for CuKa ~ up to 90°
        g_norm = (g / (math.sin(math.pi / 4) ** 2)).clamp(1e-4, 4.0)
        ang = g_norm.unsqueeze(-1) * self.fourier_freqs.view(1, 1, -1) * (2 * math.pi)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.pos_proj(feat)

    def forward(self, pxrd_x, pxrd_y, peak_num):
        src_pos, src_tokens = self.batch_input(pxrd_x, pxrd_y, peak_num)
        x = self.embed_tokens(src_tokens).squeeze(-2)
        pos_embed = self._fourier_pos(src_pos)
        x = x + pos_embed

        cls_token = self.vnode_encoder.weight.unsqueeze(0).repeat(src_tokens.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x.type(self.sentence_encoder.emb_layer_norm.weight.dtype)

        # padding where intensity==0 and pos==0 (empty slots)
        padding_mask = (src_tokens.squeeze(-1) == 0) & (src_pos.squeeze(-1) == 0)
        padding_mask = torch.cat(
            [
                torch.zeros(padding_mask.size(0), 1, dtype=torch.bool, device=x.device),
                padding_mask,
            ],
            dim=1,
        )
        if not padding_mask.any():
            padding_mask = None

        x = self.sentence_encoder(x, padding_mask=padding_mask)
        out_ = x[:, 0, :]
        return self.out(out_)


class ArmBModel(nn.Module):
    """Frozen Bert + lattice MLP (quantized 2θ positions)."""

    def __init__(self, encoder: nn.Module, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = LatticeMLP(512, 6)

    def encode(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.encoder.parameters())):
            z = self.encoder(pxrd_x, pxrd_y, peak_num)
        return F.normalize(z.float(), dim=-1)

    def forward(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        return self.head(self.encode(pxrd_x, pxrd_y, peak_num))


class ArmCModel(nn.Module):
    """Fourier-pos Bert (partial unfreeze) + lattice MLP."""

    def __init__(self, encoder: BertFourierPos):
        super().__init__()
        self.encoder = encoder
        # freeze all, then unfreeze Fourier path + last transformer layer + out
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder.pos_proj.parameters():
            p.requires_grad = True
        # last encoder layer
        layers = self.encoder.sentence_encoder.layers
        for p in layers[-1].parameters():
            p.requires_grad = True
        for p in self.encoder.out.parameters():
            p.requires_grad = True
        for p in self.encoder.sentence_encoder.final_layer_norm.parameters():
            p.requires_grad = True
        self.head = LatticeMLP(512, 6)

    def encode(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        z = self.encoder(pxrd_x, pxrd_y, peak_num)
        return F.normalize(z.float(), dim=-1)

    def forward(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        return self.head(self.encode(pxrd_x, pxrd_y, peak_num))


class LatticeOutAdapter(nn.Module):
    """Residual MLP inserted before a fresh Linear(512→9) lattice head (Arm A)."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden, 9, bias=False)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        h = graph_features + self.mlp(graph_features)
        return self.out(h).view(-1, 3, 3)


def a2_decoder_features(
    decoder: nn.Module,
    time_emb: torch.Tensor,
    lattices: torch.Tensor,
    xrd: torch.Tensor,
) -> torch.Tensor:
    """Run frozen CSPNet under A2 conditions; return graph features before lattice_out."""
    B = lattices.size(0)
    device = lattices.device
    num_atoms = torch.ones(B, dtype=torch.long, device=device)
    atom_types = torch.ones(B, dtype=torch.long, device=device)  # dummy Z=1
    frac_coords = torch.full((B, 3), 0.5, device=device, dtype=lattices.dtype)
    node2graph = torch.arange(B, device=device)

    # Replicate CSPNet.forward up to mean-pool, with zeroed atom embedding
    edges, frac_diff = decoder.gen_edges(num_atoms, frac_coords, lattices, node2graph)
    edge2graph = node2graph[edges[0]]
    xrd_feature = xrd[node2graph]
    node_features = decoder.node_embedding(atom_types - 1)
    node_features = torch.zeros_like(node_features)  # A2: strip atom semantics
    t_per_atom = time_emb.repeat_interleave(num_atoms, dim=0)
    node_features = torch.cat([node_features, t_per_atom, xrd_feature], dim=1)
    node_features = decoder.atom_latent_emb(node_features)
    for i in range(decoder.num_layers):
        node_features = decoder._modules[f"csp_layer_{i}"](
            node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
        )
    if decoder.ln:
        node_features = decoder.final_layer_norm(node_features)
    graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
    return graph_features


class A2ConditionedFlowHead(ConditionalFlowHead):
    """gstar6 flow head whose condition is augmented by the frozen A2 decoder.

    The pretrained CSPFlow decoder is itself a lattice denoiser, so its graph
    features depend on the *current* iterate, not just the pattern. Feeding them
    in alongside the static XRD embedding is what makes this transfer learning
    rather than "frozen encoder + a brand new model". ``shell_fn`` is a bound
    method of the owning module, so the frozen weights are registered once there.
    """

    def __init__(self, shell_fn, *, xrd_dim: int = 512, shell_dim: int = 512, **kwargs):
        super().__init__(embedding_dim=xrd_dim + shell_dim, **kwargs)
        self.shell_fn = shell_fn

    def forward(self, embedding, z_t, t):
        return super().forward(torch.cat([embedding, self.shell_fn(z_t, t)], dim=-1), z_t, t)


class ArmAFlowModel(nn.Module):
    """RealPXRD Bert (+ optional A2 CSPNet shell) driving a gstar6 flow head.

    ``unfreeze`` controls which pretrained modules train with the flow head:
    ``none`` (default Arm A), ``encoder``, ``decoder`` (CSPNet + time emb), or ``both``.

    ``use_cspnet=False`` (scheme D): drop the CSPNet shell entirely — condition the
    flow on the Bert XRD embedding alone. Decoder weights are discarded.
    """

    def __init__(
        self,
        bundle: nn.Module,
        *,
        normalizer,
        timesteps: int = 1000,
        flow_hidden: int = 512,
        flow_layers: int = 6,
        unfreeze: str = "none",
        use_cspnet: bool = True,
    ):
        super().__init__()
        from pxrd_cell_indexing.geometry import gstar6_to_lattice, lattice_params_to_matrix
        from pxrd_cell_indexing.model.flow_head import ConditionalFlowHead

        if unfreeze not in ("none", "encoder", "decoder", "both"):
            raise ValueError(f"unfreeze must be none|encoder|decoder|both, got {unfreeze}")
        self.unfreeze = unfreeze
        self.use_cspnet = bool(use_cspnet)
        self.train_encoder = unfreeze in ("encoder", "both")
        # No CSPNet path ⇒ decoder unfreeze is a no-op.
        self.train_decoder = self.use_cspnet and unfreeze in ("decoder", "both")

        self._gstar6_to_lattice = gstar6_to_lattice
        self._params_to_matrix = lattice_params_to_matrix

        self.xrd_encoder = bundle.xrd_encoder
        for p in self.xrd_encoder.parameters():
            p.requires_grad = self.train_encoder

        self.timesteps = timesteps
        self.register_buffer(
            "gstar_mean", torch.tensor(normalizer.component_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "gstar_std", torch.tensor(normalizer.component_std, dtype=torch.float32)
        )

        if self.use_cspnet:
            self.decoder = bundle.decoder
            self.time_embedding = bundle.time_embedding
            for p in self.decoder.parameters():
                p.requires_grad = self.train_decoder
            # time_embedding is part of the CSPFlow lattice path; follow decoder.
            for p in self.time_embedding.parameters():
                p.requires_grad = self.train_decoder
            self.flow = A2ConditionedFlowHead(
                self._shell_features,
                xrd_dim=512,
                shell_dim=512,
                hidden=flow_hidden,
                num_layers=flow_layers,
            )
        else:
            # Scheme D: Bert → flow only. Drop CSPNet so it is not moved to GPU.
            self.decoder = None
            self.time_embedding = None
            bundle.decoder = None
            bundle.time_embedding = None
            self.flow = ConditionalFlowHead(
                embedding_dim=512,
                hidden=flow_hidden,
                num_layers=flow_layers,
            )

    def _shell_features(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """A2 decoder features at the current iterate. XRD is cached by ``forward``."""
        if not self.use_cspnet:
            raise RuntimeError("shell features requested but use_cspnet=False")
        # Lattice decode from gstar6 is non-trainable geometry; detach so grads
        # only hit decoder/time when they are unfrozen (not the gstar path).
        with torch.no_grad():
            comp = z_t.float() * self.gstar_std + self.gstar_mean
            lattices = self._params_to_matrix(self._gstar6_to_lattice(comp))
            times = ((1.0 - t.float()).clamp(0, 1) * self.timesteps).clamp(min=1.0)
        ctx = torch.enable_grad() if self.train_decoder and self.training else torch.no_grad()
        with ctx:
            feats = a2_decoder_features(
                self.decoder, self.time_embedding(times), lattices, self._xrd_cache
            )
        return feats.to(z_t.dtype)

    def encode(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        if self.train_encoder and self.training:
            z = self.xrd_encoder(pxrd_x, pxrd_y, peak_num)
        else:
            with torch.no_grad():
                z = self.xrd_encoder(pxrd_x, pxrd_y, peak_num)
        return F.normalize(z.float(), dim=-1)

    def _bind_xrd(self, embedding: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Cache the per-row XRD condition the shell needs, expanded to the flow's batch."""
        if not self.use_cspnet:
            return embedding
        self._xrd_cache = (
            embedding.repeat_interleave(num_samples, dim=0) if num_samples > 1 else embedding
        )
        return embedding

    def flow_loss(self, embedding: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        self._bind_xrd(embedding)
        return self.flow.flow_loss(embedding, z1)

    @torch.no_grad()
    def sample(self, embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        self._bind_xrd(embedding, kwargs.get("num_samples", 100))
        return self.flow.sample(embedding, **kwargs)

    def forward(self, pxrd_x, pxrd_y, peak_num, z1) -> torch.Tensor:
        """Encode + flow loss in one call, so DDP's autograd hooks fire."""
        return self.flow_loss(self.encode(pxrd_x, pxrd_y, peak_num), z1)


class ArmAModel(nn.Module):
    """A2-shelled CSPFlow with frozen encoder/decoder body + trainable lattice adapter."""

    def __init__(self, bundle: nn.Module, timesteps: int = 1000):
        super().__init__()
        self.xrd_encoder = bundle.xrd_encoder
        self.decoder = bundle.decoder
        self.time_embedding = bundle.time_embedding
        self.timesteps = timesteps
        self.adapter = LatticeOutAdapter(512)

        for p in self.xrd_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # adapter is trainable

    def _a2_decoder_features(self, time_emb, lattices, xrd):
        return a2_decoder_features(self.decoder, time_emb, lattices, xrd)

    def predict_lattice_velocity(
        self, time_emb: torch.Tensor, lattices: torch.Tensor, xrd: torch.Tensor
    ) -> torch.Tensor:
        feats = self._a2_decoder_features(time_emb, lattices, xrd)
        lattice_out = self.adapter(feats)
        if self.decoder.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices)
        return lattice_out

    def encode_xrd(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        with torch.no_grad():
            z = self.xrd_encoder(pxrd_x, pxrd_y, peak_num)
        return F.normalize(z.float(), dim=-1)

    def forward_flow_loss(self, batch: dict) -> torch.Tensor:
        B = batch["lattice_matrix"].size(0)
        device = batch["lattice_matrix"].device
        times = torch.randint(1, self.timesteps + 1, (B,), device=device)
        time_emb = self.time_embedding(times.float())
        c1 = times.float() / self.timesteps
        c0 = 1.0 - c1

        lattices = batch["lattice_matrix"]
        rand_l = torch.randn_like(lattices)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        xrd = self.encode_xrd(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
        pred_l = self.predict_lattice_velocity(time_emb, input_lattice, xrd)
        return F.mse_loss(pred_l, rand_l)

    @torch.no_grad()
    def sample_lattices(
        self,
        pxrd_x,
        pxrd_y,
        peak_num,
        num_evals: int = 100,
        infer_timesteps: int = 200,
    ) -> torch.Tensor:
        """Return [K, 3, 3] lattice matrices for a single sample (broadcast peaks)."""
        device = next(self.parameters()).device
        # replicate single spectrum K times
        xrd_one = self.encode_xrd(pxrd_x, pxrd_y, peak_num)  # [1, 512]
        xrd = xrd_one.expand(num_evals, -1).contiguous()

        l_t = torch.randn(num_evals, 3, 3, device=device)
        assert self.timesteps % infer_timesteps == 0
        mult = self.timesteps // infer_timesteps
        time_start = self.timesteps - mult
        step_size = 1.0 / infer_timesteps

        for t in range(time_start, 0, -mult):
            times = torch.full((num_evals,), t, device=device, dtype=torch.float32)
            time_emb = self.time_embedding(times)
            pred_l = self.predict_lattice_velocity(time_emb, l_t, xrd)
            l_t = l_t - step_size * (pred_l - l_t) / (1.0 - t / self.timesteps)
        return l_t
