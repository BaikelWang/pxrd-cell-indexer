#!/usr/bin/env python3
"""R5-A: compare single-anchor vs multi-anchor Top-K pools (strict elementwise).

Modes:
  - single: routed lattice_norm only (baseline)
  - all7: all CS heads from lattice_norm_all
  - top2: routed CS + second-best CS classifier head
  - all7_noscale: all7 with no length/axis scale variants (elementwise-friendly)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pxrd_cell_indexing.data.dataset import (
    PXRDDatasetConfig,
    PeakFilterConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import lattice_match_elementwise, lattice_match_pymatgen
from pxrd_cell_indexing.model.fom_rerank import fom_config_from_args, maybe_rerank_candidates
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_multi_anchor_top_k_candidates,
    build_top_k_candidates,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cand_arr(c) -> np.ndarray:
    return np.asarray([c.a, c.b, c.c, c.alpha, c.beta, c.gamma], dtype=np.float64)


def _rate(xs: list[bool]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _select_anchors(
    mode: str,
    *,
    lattice_norm: torch.Tensor,
    lattice_norm_all: torch.Tensor | None,
    routed_cs_idx: torch.Tensor,
    cs_logits: torch.Tensor | None,
    normalizer,
) -> torch.Tensor:
    """Return physical anchors [B, A, 6]."""
    if mode == "single":
        phys = normalizer.denormalize(lattice_norm)
        return phys.unsqueeze(1)

    if lattice_norm_all is None:
        raise ValueError(f"mode {mode} requires lattice_norm_all")

    if mode in ("all7", "all7_noscale"):
        b, n_cs, d = lattice_norm_all.shape
        return normalizer.denormalize(lattice_norm_all.reshape(-1, d)).reshape(b, n_cs, d)

    if mode == "top2":
        if cs_logits is None:
            raise ValueError("top2 requires crystal_system_logits")
        # Always include routed index; add second-best classifier index if different.
        probs = F.softmax(cs_logits, dim=-1)
        top2 = probs.topk(k=2, dim=-1).indices  # [B, 2]
        batch = lattice_norm_all.shape[0]
        anchors_norm = []
        for i in range(batch):
            idxs = [int(routed_cs_idx[i].item())]
            for j in top2[i].tolist():
                if j not in idxs:
                    idxs.append(j)
            # Cap at 2 anchors.
            idxs = idxs[:2]
            anchors_norm.append(lattice_norm_all[i, idxs, :])
        # Pad to 2 if needed (duplicate routed).
        stacked = []
        for i, a in enumerate(anchors_norm):
            if a.shape[0] == 1:
                a = torch.cat([a, a], dim=0)
            stacked.append(a)
        anchors_n = torch.stack(stacked, dim=0)  # [B, 2, 6]
        b, a, d = anchors_n.shape
        return normalizer.denormalize(anchors_n.reshape(-1, d)).reshape(b, a, d)

    raise ValueError(f"unknown mode {mode}")


def run_mode(
    *,
    mode: str,
    model,
    loader,
    normalizer,
    device,
    config: TrainConfig,
    top_k: int,
    ltol: float,
    atol_deg: float,
) -> dict[str, Any]:
    use_scale = mode != "all7_noscale"
    topk_cfg = TopKConfig(
        k=top_k,
        length_scale_factors=() if not use_scale else TopKConfig().length_scale_factors,
        include_axis_scale_variants=use_scale,
        bravais_set="default",
    )
    fom_cfg = fom_config_from_args(
        SimpleNamespace(fom_mode="heuristic", fom_collapse_variants=False, fom_q_abs_tol=1e-6)
    )

    overall = {"raw": [], "pool_elem": [], "pool_map": [], "fom_elem": [], "fom_map": []}
    by_cs: dict[str, dict[str, list[bool]]] = {
        cs: {"raw": [], "pool_elem": [], "fom_elem": []} for cs in CRYSTAL_SYSTEMS
    }

    with torch.no_grad():
        for batch in loader:
            bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(
                bt["pxrd_x"],
                bt["pxrd_y"],
                bt["peak_num"],
                crystal_system_idx=bt["crystal_system_idx"],
                cs_route=config.model.cs_route,
                lattice_phys=bt["lattice"],
                setting_route=config.model.setting_route,
            )
            anchors = _select_anchors(
                mode,
                lattice_norm=out["lattice_norm"],
                lattice_norm_all=out.get("lattice_norm_all"),
                routed_cs_idx=out["routed_cs_idx"],
                cs_logits=out.get("crystal_system_logits"),
                normalizer=normalizer,
            )
            if anchors.shape[1] == 1:
                pools = build_top_k_candidates(anchors[:, 0, :], k=top_k, config=topk_cfg)
            else:
                pools = build_multi_anchor_top_k_candidates(
                    anchors, k=top_k, config=topk_cfg, per_anchor_k=max(8, top_k // anchors.shape[1])
                )

            pred_phys = normalizer.denormalize(out["lattice_norm"])
            for i in range(pred_phys.shape[0]):
                cs = CRYSTAL_SYSTEMS[int(bt["crystal_system_idx"][i].item())]
                t = bt["lattice"][i].cpu().numpy()
                p = pred_phys[i].cpu().numpy()
                raw_ok = bool(lattice_match_elementwise(p, t, ltol=ltol, atol_deg=atol_deg))
                pool = pools[i]
                pool_elem = any(
                    lattice_match_elementwise(_cand_arr(c), t, ltol=ltol, atol_deg=atol_deg)
                    for c in pool
                )
                pool_map = any(
                    lattice_match_pymatgen(_cand_arr(c), t, ltol=ltol, atol_deg=atol_deg)
                    for c in pool
                )
                fom = maybe_rerank_candidates(
                    pool,
                    rerank="fom",
                    pxrd_x=bt["pxrd_x"],
                    pxrd_y=bt["pxrd_y"],
                    peak_num=bt["peak_num"],
                    sample_index=i,
                    fom_config=fom_cfg,
                )
                top = _cand_arr(fom[0])
                fom_elem = bool(lattice_match_elementwise(top, t, ltol=ltol, atol_deg=atol_deg))
                fom_map = bool(lattice_match_pymatgen(top, t, ltol=ltol, atol_deg=atol_deg))

                overall["raw"].append(raw_ok)
                overall["pool_elem"].append(pool_elem)
                overall["pool_map"].append(pool_map)
                overall["fom_elem"].append(fom_elem)
                overall["fom_map"].append(fom_map)
                by_cs[cs]["raw"].append(raw_ok)
                by_cs[cs]["pool_elem"].append(pool_elem)
                by_cs[cs]["fom_elem"].append(fom_elem)

    non = [cs for cs in CRYSTAL_SYSTEMS if cs != "cubic"]
    return {
        "mode": mode,
        "n": len(overall["raw"]),
        "overall": {k: _rate(v) for k, v in overall.items()},
        "non_cubic": {
            "raw": _rate(sum((by_cs[cs]["raw"] for cs in non), [])),
            "pool_elem": _rate(sum((by_cs[cs]["pool_elem"] for cs in non), [])),
            "fom_elem": _rate(sum((by_cs[cs]["fom_elem"] for cs in non), [])),
        },
        "by_crystal_system": {
            cs: {k: _rate(v) for k, v in bucket.items()} | {"n": len(bucket["raw"])}
            for cs, bucket in by_cs.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument(
        "--modes",
        type=str,
        default="single,top2,all7,all7_noscale",
        help="Comma-separated modes to run",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = TrainConfig.from_yaml(args.config).resolve_paths(PROJECT_ROOT)
    model, _, exp = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    normalizer = build_lattice_normalizer(config.data)
    model.set_normalizer(normalizer)

    ds = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    results = {"experiment": exp, "modes": {}}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"[run] mode={mode}", flush=True)
        row = run_mode(
            mode=mode,
            model=model,
            loader=loader,
            normalizer=normalizer,
            device=device,
            config=config,
            top_k=args.top_k,
            ltol=args.ltol,
            atol_deg=args.atol_deg,
        )
        results["modes"][mode] = row
        o = row["overall"]
        n = row["non_cubic"]
        print(
            f"  overall raw={o['raw']*100:.2f}% pool_elem={o['pool_elem']*100:.2f}% "
            f"pool_map={o['pool_map']*100:.2f}% fom_elem={o['fom_elem']*100:.2f}% "
            f"fom_map={o['fom_map']*100:.2f}%"
        )
        print(
            f"  noncub raw={n['raw']*100:.2f}% pool_elem={n['pool_elem']*100:.2f}% "
            f"fom_elem={n['fom_elem']*100:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
