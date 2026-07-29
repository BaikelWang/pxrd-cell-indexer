#!/usr/bin/env python3
"""Conditional rectified flow over gstar6 as a McMaille-replacement seed generator.

Model selection is on held-out **seed precision**, not on valid loss and not on
MP100. Two reasons, both measured:

* MP100 is n=100 scored by stochastic K-draws, so its library@K has a ~5pp
  standard error -- picking the argmax epoch over a 34-epoch run selects noise.
* L4-strict allows ``ltol=0.05``, but McMaille's Rp gate needs ~0.2%. On the
  K=1000 pool, 73% of samples had a 5%-hit while only 61% had a sub-1% one, so
  the loose metric overstates what the downstream can actually use.

So the default selection metric is ``valid_1pct``: the fraction of a fixed
valid subset with at least one draw whose aligned length error is under
``--select-tol``. MP100 stays a final-report number.

The ``--equiv-target`` switch is the ablation for the anti-mode-collapse
mechanism: when on, each flow target is a random alternative basis of the true
lattice instead of the canonical Niggli representative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pxrd_cell_indexing.data.dataset import (  # noqa: E402
    PeakFilterConfig,
    PXRDDataset,
    PXRDDatasetConfig,
    SpectrumAugmentConfig,
    collate_peak_batch,
)
from pxrd_cell_indexing.data.equivalent_cells import (  # noqa: E402
    sample_equivalent_lattice6,
)
from pxrd_cell_indexing.data.mp100 import load_mp100_dataset  # noqa: E402
from pxrd_cell_indexing.data.normalization import GStar6Normalizer  # noqa: E402
from pxrd_cell_indexing.geometry import gstar6_to_lattice  # noqa: E402
from pxrd_cell_indexing.model.encoder.peak_transformer import (  # noqa: E402
    PeakGeometryTransformerEncoder,
)
from pxrd_cell_indexing.model.flow_head import ConditionalFlowHead  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

ENCODER_CFG = {
    "peak_transformer_max_peaks": 48,
    "peak_transformer_token_mode": "geom",
    "peak_transformer_d_model": 256,
    "peak_transformer_num_layers": 4,
    "peak_transformer_num_heads": 8,
    "peak_transformer_ffn_dim": 1024,
    "peak_transformer_dropout": 0.1,
    "peak_transformer_fourier_freqs": 16,
    "peak_transformer_pool": "cls_mean",
    "output_dim": 512,
    "wavelength_angstrom": 1.54184,
    "intensity_transform": "linear",
    "intensity_min": 5.0,
}


# Scheme F: scaled-up XRD encoders. Pretrained is d32/L2/ffn32 = 36,928 params;
# d256L4 (3.40M) is matched to the self-built PeakTransformer (3.31M).
ARMA_ENCODER_SCALES = {
    "pretrained": None,
    "d128L4": dict(
        encoder_embed_dim=128,
        encoder_layers=4,
        encoder_ffn_embed_dim=512,
        encoder_attention_heads=8,
    ),
    "d256L4": dict(
        encoder_embed_dim=256,
        encoder_layers=4,
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
    ),
    "d256L6": dict(
        encoder_embed_dim=256,
        encoder_layers=6,
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
    ),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/processed/train100k_niggli_seed42.jsonl")
    ap.add_argument("--valid-jsonl", default="data/processed/valid1400_niggli_seed42.jsonl")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_100k_niggli_seed42.json")
    ap.add_argument("--train-lmdb", default="/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb")
    ap.add_argument("--valid-lmdb", default="/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb")
    ap.add_argument("--equiv-target", choices=["on", "off"], default="on")
    ap.add_argument(
        "--backbone",
        choices=["peaktf", "armA"],
        default="peaktf",
        help="peaktf: self-built PeakTransformer trained from scratch. "
        "armA: RealPXRD Bert + A2 CSPNet shell + gstar6 flow head.",
    )
    ap.add_argument(
        "--arma-unfreeze",
        choices=["none", "encoder", "decoder", "both"],
        default="none",
        help="armA only: which pretrained modules train with the flow head. "
        "none=freeze both (original Arm A); encoder=Bert; decoder=CSPNet+time; both.",
    )
    ap.add_argument(
        "--arma-no-cspnet",
        action="store_true",
        help="armA scheme D: drop CSPNet shell; condition flow on Bert embedding only.",
    )
    ap.add_argument(
        "--arma-encoder-scale",
        choices=sorted(ARMA_ENCODER_SCALES),
        default="pretrained",
        help="armA scheme F: resize the XRD encoder. Anything but 'pretrained' makes "
        "every checkpoint tensor shape-incompatible, so the encoder trains FROM "
        "SCRATCH -- a scratch control reusing the architecture, not transfer learning. "
        "Requires --arma-unfreeze encoder|both.",
    )
    ap.add_argument(
        "--arma-pos",
        choices=["int", "fourier"],
        default="int",
        help="armA scheme E: 'int' is the pretrained 1-degree embed_positions lookup; "
        "'fourier' swaps in continuous Fourier(sin^2 theta) positions. "
        "'fourier' requires --arma-unfreeze encoder|both (pos_proj is randomly init).",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before optimizer step "
        "(effective batch = batch-size * grad-accum).",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument(
        "--amp",
        choices=["off", "bf16", "fp16"],
        default="off",
        help="mixed precision. bf16 preferred on Ampere+/Ada (no GradScaler). "
        "fp16 uses GradScaler. Eval sampling also runs under autocast.",
    )
    ap.add_argument("--flow-layers", type=int, default=6)
    ap.add_argument("--flow-hidden", type=int, default=512)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--eval-k", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument(
        "--select-metric",
        choices=["valid_1pct", "mp100_l4"],
        default="valid_1pct",
        help="what best.pt is selected on. valid_1pct: held-out fraction with a "
        "draw under --select-tol aligned length error. mp100_l4: the legacy "
        "MP100 library@K at ltol=0.05 (noisy, n=100 -- kept for reproduction).",
    )
    ap.add_argument("--select-tol", type=float, default=0.01)
    ap.add_argument(
        "--valid-eval-n",
        type=int,
        default=300,
        help="fixed prefix of the valid split used for selection",
    )
    ap.add_argument("--eval-workers", type=int, default=48)
    ap.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="stop each epoch after N batches (0 = full epoch); smoke tests only",
    )
    ap.add_argument(
        "--mp100-every",
        type=int,
        default=0,
        help="run the MP100 report every N epochs (0 = final epoch only). "
        "Never used for selection unless --select-metric mp100_l4.",
    )
    ap.add_argument("--augment", choices=["on", "off"], default="on")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", default="", help="checkpoint to load model weights from")
    ap.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="first epoch index when resuming; the LR cosine continues from here",
    )
    return ap.parse_args()


def ddp_setup(use_cuda: bool) -> tuple[int, int, int]:
    """Join the torchrun process group. Returns (rank, world_size, local_rank)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return rank, world, local


def build_dataset(args, split: str, *, augment: bool):
    return PXRDDataset(
        PXRDDatasetConfig(
            lmdb_path=Path(args.train_lmdb if split == "train" else args.valid_lmdb),
            split=split,
            sample_list_path=Path(args.train_jsonl if split == "train" else args.valid_jsonl),
            peak_filter=PeakFilterConfig(intensity_min=5.0, max_peaks=None),
            xrd_augment=augment,
            augment=SpectrumAugmentConfig(shift_range=0.1),
            strict=False,
            seed_base=args.seed,
        )
    )


def build_loader(args, split: str, *, world_size: int = 1, rank: int = 0):
    is_train = split == "train"
    dataset = build_dataset(args, split, augment=is_train and args.augment == "on")
    sampler = None
    if is_train and world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_peak_batch,
        pin_memory=args.device.startswith("cuda"),
        drop_last=is_train,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


def randomize_targets(lattice: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Replace each cell by a random alternative basis of the same lattice."""
    arr = lattice.numpy()
    out = np.stack([sample_equivalent_lattice6(row, rng) for row in arr], axis=0)
    return torch.from_numpy(out.astype(np.float32))


class SeedGenerator(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.encoder = PeakGeometryTransformerEncoder(dict(ENCODER_CFG))
        self.flow = ConditionalFlowHead(
            embedding_dim=ENCODER_CFG["output_dim"],
            hidden=args.flow_hidden,
            num_layers=args.flow_layers,
        )

    def encode(self, pxrd_x, pxrd_y, peak_num) -> torch.Tensor:
        return self.encoder(pxrd_x, pxrd_y, peak_num)

    def flow_loss(self, emb, z1):
        return self.flow.flow_loss(emb, z1)

    def sample(self, emb, **kwargs):
        return self.flow.sample(emb, **kwargs)

    def forward(self, pxrd_x, pxrd_y, peak_num, z1):
        """Encode + flow loss in one call, so DDP's autograd hooks fire."""
        return self.flow_loss(self.encode(pxrd_x, pxrd_y, peak_num), z1)


def build_model(args, normalizer) -> torch.nn.Module:
    if args.backbone == "peaktf":
        return SeedGenerator(args)
    # Arm A: reuse the RealPXRD checkpoint, swap only the output head.
    from ft_realpxrd.models import ArmAFlowModel
    from ft_realpxrd.common import load_cspflow_from_ckpt

    if args.arma_pos == "fourier" and args.arma_unfreeze not in ("encoder", "both"):
        raise SystemExit(
            "--arma-pos fourier needs --arma-unfreeze encoder|both: pos_proj is "
            "randomly initialized, freezing it would feed the encoder pure noise."
        )
    override = ARMA_ENCODER_SCALES[args.arma_encoder_scale]
    if override is not None and args.arma_unfreeze not in ("encoder", "both"):
        raise SystemExit(
            f"--arma-encoder-scale {args.arma_encoder_scale} needs --arma-unfreeze "
            "encoder|both: the resized encoder is randomly initialized."
        )
    bundle, hp, missing, unexpected = load_cspflow_from_ckpt(
        torch.device(args.device),
        continuous_pos=args.arma_pos == "fourier",
        encoder_override=override,
    )
    print(f"armA ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if override is not None:
        print(
            f"WARNING scheme F: encoder resized to {args.arma_encoder_scale}; "
            "pretrained XRD weights NOT loaded (scratch control, not transfer learning).",
            flush=True,
        )
    if args.arma_no_cspnet and args.arma_unfreeze in ("decoder", "both"):
        print(
            f"warn: --arma-no-cspnet ignores decoder unfreeze "
            f"(requested {args.arma_unfreeze})",
            flush=True,
        )
    return ArmAFlowModel(
        bundle,
        normalizer=normalizer,
        timesteps=hp["timesteps"],
        flow_hidden=args.flow_hidden,
        flow_layers=args.flow_layers,
        unfreeze=args.arma_unfreeze,
        use_cspnet=not args.arma_no_cspnet,
    )


# --------------------------------------------------------------------- eval
SELECT_TOLS = (0.002, 0.005, 0.01, 0.02, 0.05)


def _score_valid_sample(payload):
    """Best aligned length error over K draws. Runs in a worker process.

    A candidate can only be a strict hit if its volume is within the |det-1|<0.25
    band, so a cheap volume filter skips most of the expensive find_mapping calls.
    """
    from pymatgen.core import Lattice

    from diagnose_mcmaille_value import cell_err
    from remeasure_l4_prim_vs_conv import l4

    cells, truth = payload
    try:
        truth_vol = Lattice.from_parameters(*truth).volume
    except Exception:
        return None
    best = None
    for row in cells:
        if not np.isfinite(row).all():
            continue
        try:
            ratio = Lattice.from_parameters(*row.tolist()).volume / truth_vol
        except Exception:
            continue
        if not 0.7 < ratio < 1.4:
            continue
        if not l4(row.tolist(), truth)[1]:
            continue
        err, _ = cell_err(row.tolist(), truth)
        if np.isfinite(err) and (best is None or err < best):
            best = err
    return best


@torch.no_grad()
def evaluate_valid_precision(
    net, normalizer, batches, device, *, k: int, steps: int, seed: int, workers: int
) -> dict:
    """Fraction of the fixed valid subset with a draw inside each error tolerance.

    z0 is redrawn from the same seed every epoch (common random numbers), so
    epoch-to-epoch differences reflect the model, not resampling luck.
    """
    net.eval()
    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)

    payloads = []
    for batch in batches:
        emb = net.encode(
            batch["pxrd_x"].to(device),
            batch["pxrd_y"].to(device),
            batch["peak_num"].to(device),
        )
        gen = torch.Generator(device=device).manual_seed(seed)
        z = net.sample(emb, num_samples=k, steps=steps, generator=gen)  # [B, K, 6]
        bsz = z.shape[0]
        cells = gstar6_to_lattice((std * z + mean).reshape(-1, 6))
        cells = cells.reshape(bsz, k, 6).cpu().numpy()
        truths = batch["lattice"].numpy()
        payloads.extend((cells[i], truths[i].tolist()) for i in range(bsz))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        best = list(pool.map(_score_valid_sample, payloads, chunksize=4))

    n = len(best)
    hits = [e for e in best if e is not None]
    return {
        "n": n,
        "hit_rate": {str(t): sum(1 for e in hits if e < t) / max(n, 1) for t in SELECT_TOLS},
        "median_best_err": float(np.median(hits)) if hits else None,
        "frac_any_hit": len(hits) / max(n, 1),
    }


def load_valid_eval_batches(args, n: int) -> list:
    """Fixed prefix of the valid split, un-augmented, kept in memory."""
    loader = DataLoader(
        build_dataset(args, "valid", augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=min(4, args.num_workers),
        collate_fn=collate_peak_batch,
    )
    batches, seen = [], 0
    for batch in loader:
        batches.append(batch)
        seen += batch["lattice"].shape[0]
        if seen >= n:
            break
    return batches


def load_mp100_eval(device):
    samples = load_mp100_dataset(ROOT / "data/MP-100samples-benchmark")
    items = []
    for s in samples:
        sid = s.sample_id
        truth = truth_cells(CIF_DIR / f"{sid}.cif")
        items.append(
            {
                "sample_id": sid,
                "pxrd_x": torch.tensor(s.two_theta, dtype=torch.float32).view(-1, 1).to(device),
                "pxrd_y": torch.tensor(s.intensity, dtype=torch.float32).view(-1, 1).to(device),
                "peak_num": torch.tensor([s.peak_num], dtype=torch.long).to(device),
                "prim": truth["prim"],
            }
        )
    return items


@torch.no_grad()
def evaluate_coverage(model, normalizer, items, device, *, k: int, steps: int) -> dict:
    model.eval()
    ks = [1, 5, 10, 20, 50, k]
    hits = {kk: 0 for kk in ks}
    dets, vol_ratios, n_valid = [], [], []
    from pymatgen.core import Lattice

    for item in items:
        emb = model.encode(item["pxrd_x"], item["pxrd_y"], item["peak_num"])
        z = model.sample(emb, num_samples=k, steps=steps)[0]  # [K, 6]
        comp = torch.tensor(normalizer.component_std, device=z.device) * z + torch.tensor(
            normalizer.component_mean, device=z.device
        )
        cells = gstar6_to_lattice(comp).cpu().numpy()

        truth = item["prim"]
        tv = Lattice.from_parameters(*truth).volume
        flags, best_det = [], None
        for row in cells:
            lo, st, det = l4(row.tolist(), truth)
            flags.append(st)
            if det is not None and (best_det is None or abs(det - 1) < abs(best_det - 1)):
                best_det = det
        for kk in ks:
            if any(flags[:kk]):
                hits[kk] += 1
        if best_det is not None:
            dets.append(best_det)
        try:
            vols = [Lattice.from_parameters(*row.tolist()).volume for row in cells]
            vol_ratios.append(float(np.median(vols) / tv))
        except Exception:
            pass
        n_valid.append(float(np.isfinite(cells).all(axis=1).mean()))

    n = len(items)
    return {
        "coverage": {str(kk): hits[kk] / n for kk in ks},
        "library_strict": hits[k] / n,
        "best_det_median": float(np.median(dets)) if dets else None,
        "med_vol_ratio": float(np.median(vol_ratios)) if vol_ratios else None,
        "frac_finite": float(np.mean(n_valid)) if n_valid else 0.0,
    }


METRIC_COLUMNS = [
    "epoch",
    "train_loss",
    "valid_loss",
    "lr",
    "valid_hit_0.2pct",
    "valid_hit_0.5pct",
    "valid_hit_1pct",
    "valid_hit_5pct",
    "valid_median_err",
    "select_score",
    "mp100_library_strict",
    "elapsed_s",
]


def append_metrics_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    use_cuda = args.device.startswith("cuda")
    rank, world, local_rank = ddp_setup(use_cuda)
    is_main = rank == 0
    if world > 1 and use_cuda:
        args.device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(
            json.dumps({**vars(args), "world_size": world}, indent=2)
        )

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    normalizer = GStar6Normalizer.from_json(args.stats)
    train_loader, train_sampler = build_loader(args, "train", world_size=world, rank=rank)
    valid_loader, _ = build_loader(args, "valid")

    valid_eval_batches = mp100 = None
    if is_main:
        valid_eval_batches = load_valid_eval_batches(args, args.valid_eval_n)
        n_valid_eval = sum(b["lattice"].shape[0] for b in valid_eval_batches)
        mp100 = load_mp100_eval(device)
        log(
            f"world={world} train batches/rank={len(train_loader)} "
            f"global batch={args.batch_size * world * max(1, args.grad_accum)} "
            f"valid batches={len(valid_loader)} select_set={n_valid_eval} mp100={len(mp100)}"
        )
        log(f"select_metric={args.select_metric} tol={args.select_tol:.1%} k={args.eval_k}")

    model = build_model(args, normalizer).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    if args.backbone == "armA":
        shell = "no_cspnet" if args.arma_no_cspnet else "cspnet"
        extra = (
            f" shell={shell} pos={args.arma_pos} enc={args.arma_encoder_scale}"
            f" unfreeze={args.arma_unfreeze}"
        )
    else:
        extra = ""
    log(f"backbone={args.backbone}{extra} trainable {n_par:,} / total {n_tot:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = int(args.warmup_epochs * len(train_loader))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        if "optimizer_state" not in ck:
            # Resuming without AdamW moments restarts the optimizer cold: the
            # first full6m run lost ~2 epochs of progress this way.
            raise SystemExit(
                f"{args.resume} has no optimizer_state, so resuming from it would "
                "reset AdamW's moments and undo progress. Resume from last.pt."
            )
        opt.load_state_dict(ck["optimizer_state"])
        log(f"resumed from {args.resume} (epoch {ck.get('epoch')})")

    net = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
        net = model.module

    rng = np.random.default_rng(args.seed)
    history, best = [], {"select_score": -1.0, "epoch": -1}
    hist_path = out_dir / "history.json"
    if args.resume and hist_path.exists():
        history = [r for r in json.loads(hist_path.read_text()) if r["epoch"] < args.start_epoch]
        for r in history:
            if r.get("select_score", -1) > best["select_score"]:
                best = {"select_score": r["select_score"], "epoch": r["epoch"]}
    start_epoch = max(1, args.start_epoch)
    step = (start_epoch - 1) * len(train_loader)
    t0 = time.time()

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    amp_enabled = amp_dtype is not None and use_cuda
    # bf16 does not need a scaler; fp16 does.
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and use_cuda))
    if is_main:
        log(f"amp={args.amp} enabled={amp_enabled}")

    def autocast_ctx():
        return torch.autocast(
            device_type="cuda" if use_cuda else "cpu",
            dtype=amp_dtype or torch.float32,
            enabled=amp_enabled,
        )

    accum = max(1, int(args.grad_accum))
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader, 1):
            lattice = batch["lattice"]
            if args.equiv_target == "on":
                lattice = randomize_targets(lattice, rng)
            z1 = normalizer.normalize(lattice.to(device))
            with autocast_ctx():
                loss = (
                    model(
                        batch["pxrd_x"].to(device),
                        batch["pxrd_y"].to(device),
                        batch["peak_num"].to(device),
                        z1,
                    )
                    / accum
                )
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().float().item()) * accum)

            if bi % accum == 0 or bi == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for group in opt.param_groups:
                    group["lr"] = args.lr * lr_at(step)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

            if args.limit_train_batches and bi >= args.limit_train_batches:
                break

        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in valid_loader:
                z1 = normalizer.normalize(batch["lattice"].to(device))
                with autocast_ctx():
                    emb = net.encode(
                        batch["pxrd_x"].to(device),
                        batch["pxrd_y"].to(device),
                        batch["peak_num"].to(device),
                    )
                    vloss = net.flow_loss(emb, z1)
                vlosses.append(float(vloss.float().item()))
        train_loss = float(np.mean(losses))
        if world > 1:
            packed = torch.tensor([train_loss, float(np.mean(vlosses))], device=device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)  # AVG is NCCL-only
            train_loss, valid_loss = (packed / world).tolist()
        else:
            valid_loss = float(np.mean(vlosses))

        if is_main:
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": args.lr * lr_at(step),
                "elapsed_s": time.time() - t0,
            }

            prec = None
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                with autocast_ctx():
                    prec = evaluate_valid_precision(
                        net,
                        normalizer,
                        valid_eval_batches,
                        device,
                        k=args.eval_k,
                        steps=args.sample_steps,
                        seed=args.seed,
                        workers=args.eval_workers,
                    )
                row["valid_precision"] = prec
                row["valid_hit_0.2pct"] = prec["hit_rate"]["0.002"]
                row["valid_hit_0.5pct"] = prec["hit_rate"]["0.005"]
                row["valid_hit_1pct"] = prec["hit_rate"]["0.01"]
                row["valid_hit_5pct"] = prec["hit_rate"]["0.05"]
                row["valid_median_err"] = prec["median_best_err"]

            want_mp100 = epoch == args.epochs or (
                args.mp100_every > 0 and epoch % args.mp100_every == 0
            )
            if want_mp100 or args.select_metric == "mp100_l4":
                with autocast_ctx():
                    cov = evaluate_coverage(
                        net, normalizer, mp100, device, k=args.eval_k, steps=args.sample_steps
                    )
                row["mp100"] = cov
                row["mp100_library_strict"] = cov["library_strict"]
            if args.select_metric == "valid_1pct":
                score = prec["hit_rate"][str(args.select_tol)] if prec else None
            else:
                score = row.get("mp100_library_strict")
            row["select_score"] = score

            parts = [f"ep{epoch:03d} train={train_loss:.4f} valid={valid_loss:.4f}"]
            if prec:
                parts.append(
                    f"| valid <0.2%={prec['hit_rate']['0.002']:.0%} "
                    f"<1%={prec['hit_rate']['0.01']:.0%} "
                    f"<5%={prec['hit_rate']['0.05']:.0%} "
                    f"med={prec['median_best_err'] or float('nan'):.4%}"
                )
            if "mp100_library_strict" in row:
                parts.append(f"| mp100@{args.eval_k}={row['mp100_library_strict']:.0%}")
            print(" ".join(parts), flush=True)

            if score is not None and score > best["select_score"]:
                best = {"select_score": score, "epoch": epoch, **{k: v for k, v in row.items()}}
                torch.save(
                    {
                        "model_state": net.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "args": vars(args),
                        "encoder_cfg": ENCODER_CFG,
                        "normalizer": normalizer.to_dict(),
                        "select_metric": args.select_metric,
                        "select_score": score,
                        "valid_precision": prec,
                        "epoch": epoch,
                    },
                    out_dir / "best.pt",
                )

            history.append(row)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            append_metrics_row(out_dir / "metrics.csv", row)
            torch.save(
                {
                    "model_state": net.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "args": vars(args),
                    "encoder_cfg": ENCODER_CFG,
                    "normalizer": normalizer.to_dict(),
                    "epoch": epoch,
                },
                out_dir / "last.pt",
            )

        if world > 1:
            dist.barrier()

    if is_main:
        (out_dir / "best_meta.json").write_text(json.dumps(best, indent=2))
        print(
            f"done. best {args.select_metric}={best['select_score']:.0%} @ epoch {best['epoch']}",
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
