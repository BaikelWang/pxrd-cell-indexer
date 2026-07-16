"""Shared helpers for FOM rerank CLI wiring."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

import torch

from pxrd_cell_indexing.model.fom import (
    DEFAULT_Q_MATCH_ABS_TOL,
    FomRerankConfig,
    rerank_candidates_by_fom,
    slice_observed_intensity,
    slice_observed_two_theta,
)
from pxrd_cell_indexing.model.topk import lattice_params_volume


def add_fom_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fom-mode",
        type=str,
        choices=("heuristic", "strict_dewolff", "intensity_weighted"),
        default="heuristic",
        help="FOM ranking formula when --rerank fom",
    )
    parser.add_argument(
        "--fom-collapse-variants",
        action="store_true",
        help="Collapse scale/axis variants to one representative per base Bravais key",
    )
    parser.add_argument(
        "--fom-q-abs-tol",
        type=float,
        default=DEFAULT_Q_MATCH_ABS_TOL,
        help="Absolute Q tolerance for peak matching (ideal sticks default 1e-6)",
    )
    parser.add_argument(
        "--fom-use-ref-volume",
        action="store_true",
        help="Prefer candidates near NN-predicted volume (fixes half-cell bias under strict elementwise)",
    )
    parser.add_argument(
        "--fom-max-log-volume-ratio",
        type=float,
        default=0.693147,
        help="When --fom-use-ref-volume, hard-drop candidates with |log(V/V_ref)| above this (default log2)",
    )
    parser.add_argument(
        "--fom-volume-log-penalty",
        type=float,
        default=1.0,
        help="Weight of |log(V/V_ref)| in FOM sort key when --fom-use-ref-volume",
    )


def fom_config_from_args(args: argparse.Namespace) -> FomRerankConfig:
    use_ref = bool(getattr(args, "fom_use_ref_volume", False))
    return FomRerankConfig(
        mode=args.fom_mode,  # type: ignore[arg-type]
        collapse_variants=getattr(args, "fom_collapse_variants", False),
        q_match_abs_tol=getattr(args, "fom_q_abs_tol", DEFAULT_Q_MATCH_ABS_TOL),
        # ref_volume is filled per-sample at rerank time when use_ref is set.
        ref_volume=None,
        max_log_volume_ratio=(
            getattr(args, "fom_max_log_volume_ratio", 0.693147) if use_ref else None
        ),
        volume_log_penalty=getattr(args, "fom_volume_log_penalty", 1.0),
    )


def maybe_rerank_candidates(
    candidates: list,
    *,
    rerank: str,
    pxrd_x: torch.Tensor,
    pxrd_y: torch.Tensor,
    peak_num: torch.Tensor,
    sample_index: int,
    fom_config: FomRerankConfig | None = None,
    ref_lattice_params: Sequence[float] | None = None,
) -> list:
    if rerank == "none":
        return candidates
    if rerank == "fom":
        observed = slice_observed_two_theta(pxrd_x, peak_num, sample_index)
        intensity = slice_observed_intensity(pxrd_y, peak_num, sample_index)
        use_intensity = fom_config is not None and fom_config.mode == "intensity_weighted"
        cfg = fom_config
        if (
            cfg is not None
            and cfg.max_log_volume_ratio is not None
            and ref_lattice_params is not None
            and cfg.ref_volume is None
        ):
            cfg = replace(cfg, ref_volume=lattice_params_volume(ref_lattice_params))
        return rerank_candidates_by_fom(
            candidates,
            observed,
            observed_intensity=intensity if use_intensity else None,
            config=cfg,
        )
    raise ValueError(f"Unsupported rerank mode: {rerank!r}")
