#!/usr/bin/env python3
"""R0: per-sample raw lattice error diagnosis (no Top-K / FOM).

Dumps sample-level geometry errors and aggregates by crystal system, peak count,
failure mode, and angle-bias patterns to explain why strict raw accuracy is low.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import infer_crystal_system_idx_from_lattice
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANGLE_NAMES = ("alpha", "beta", "gamma")
LENGTH_NAMES = ("a", "b", "c")


def _percentile(arr: np.ndarray, qs: list[float]) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _safe_mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else float("nan")


def _volume_from_params(params: np.ndarray) -> float:
    matrix = lattice_params_to_matrix(torch.tensor(params, dtype=torch.float64).view(1, 6))
    return float(torch.det(matrix[0]).abs().item())


def _sample_record(
    *,
    pred: np.ndarray,
    truth: np.ndarray,
    crystal_system: str,
    peak_num: int,
    pred_cs_idx: int,
    ltol: float,
    atol_deg: float,
    sample_id: str | None = None,
) -> dict[str, Any]:
    length_abs = np.abs(pred[:3] - truth[:3])
    length_rel = length_abs / np.clip(np.abs(truth[:3]), 1e-6, None)
    angle_abs = np.abs(pred[3:] - truth[3:])
    length_ok = bool(np.all(length_rel <= ltol))
    angle_ok = bool(np.all(angle_abs <= atol_deg))
    elem_ok = length_ok and angle_ok
    # Bias toward orthogonal cell: |pred - 90| vs |truth - 90|
    pred_orth = np.abs(pred[3:] - 90.0)
    truth_orth = np.abs(truth[3:] - 90.0)
    pulled_to_90 = bool(np.mean(pred_orth) + 1e-6 < np.mean(truth_orth))
    v_pred = _volume_from_params(pred)
    v_truth = _volume_from_params(truth)
    log_vol = float(math.log(max(v_pred, 1e-12) / max(v_truth, 1e-12)))
    if elem_ok:
        fail_mode = "ok"
    elif (not length_ok) and angle_ok:
        fail_mode = "length_only"
    elif length_ok and (not angle_ok):
        fail_mode = "angle_only"
    else:
        fail_mode = "both"

    # Discrete 2θ quantization proxy: not available here; peak_num only.
    return {
        "sample_id": sample_id,
        "crystal_system": crystal_system,
        "peak_num": int(peak_num),
        "pred": pred.tolist(),
        "truth": truth.tolist(),
        "pred_cs_idx": int(pred_cs_idx),
        "target_cs_idx": int(CRYSTAL_SYSTEM_TO_IDX.get(crystal_system, -1)),
        "length_abs_mae": float(length_abs.mean()),
        "length_rel_mean": float(length_rel.mean()),
        "length_rel_max": float(length_rel.max()),
        "angle_abs_mae": float(angle_abs.mean()),
        "angle_abs_max": float(angle_abs.max()),
        "length_ok_strict": length_ok,
        "angle_ok_strict": angle_ok,
        "elem_ok_strict": elem_ok,
        "fail_mode": fail_mode,
        "pulled_to_90": pulled_to_90,
        "pred_orth_dev_mean": float(pred_orth.mean()),
        "truth_orth_dev_mean": float(truth_orth.mean()),
        "log_volume_ratio": log_vol,
        "abs_log_volume_ratio": abs(log_vol),
        "per_length_rel": {n: float(v) for n, v in zip(LENGTH_NAMES, length_rel)},
        "per_angle_abs": {n: float(v) for n, v in zip(ANGLE_NAMES, angle_abs)},
        "per_angle_pred": {n: float(v) for n, v in zip(ANGLE_NAMES, pred[3:])},
        "per_angle_truth": {n: float(v) for n, v in zip(ANGLE_NAMES, truth[3:])},
    }


def _aggregate(records: list[dict[str, Any]], *, ltol: float, atol_deg: float) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_samples": 0}

    length_rel = np.array([r["length_rel_mean"] for r in records], dtype=np.float64)
    length_rel_max = np.array([r["length_rel_max"] for r in records], dtype=np.float64)
    angle_mae = np.array([r["angle_abs_mae"] for r in records], dtype=np.float64)
    angle_max = np.array([r["angle_abs_max"] for r in records], dtype=np.float64)
    abs_log_v = np.array([r["abs_log_volume_ratio"] for r in records], dtype=np.float64)
    peak_num = np.array([r["peak_num"] for r in records], dtype=np.float64)

    fail_counts: dict[str, int] = defaultdict(int)
    for r in records:
        fail_counts[r["fail_mode"]] += 1

    # Within-tolerance fractions (how close to gate)
    frac_len_le = {
        f"le_{t}": float(np.mean(length_rel_max <= t))
        for t in (0.05, 0.10, 0.20, 0.30)
    }
    frac_ang_le = {
        f"le_{t}": float(np.mean(angle_max <= t))
        for t in (3.0, 5.0, 10.0, 15.0, 20.0)
    }

    by_cs: dict[str, Any] = {}
    for cs in CRYSTAL_SYSTEMS:
        subset = [r for r in records if r["crystal_system"] == cs]
        if not subset:
            continue
        s_len = np.array([r["length_rel_mean"] for r in subset])
        s_ang = np.array([r["angle_abs_mae"] for r in subset])
        s_ang_max = np.array([r["angle_abs_max"] for r in subset])
        pulled = np.mean([r["pulled_to_90"] for r in subset])
        modes = defaultdict(int)
        for r in subset:
            modes[r["fail_mode"]] += 1
        # For non-cubic: mean |truth-90| and |pred-90|
        truth_dev = np.array([r["truth_orth_dev_mean"] for r in subset])
        pred_dev = np.array([r["pred_orth_dev_mean"] for r in subset])
        by_cs[cs] = {
            "count": len(subset),
            "elem_ok_rate": float(np.mean([r["elem_ok_strict"] for r in subset])),
            "length_ok_rate": float(np.mean([r["length_ok_strict"] for r in subset])),
            "angle_ok_rate": float(np.mean([r["angle_ok_strict"] for r in subset])),
            "length_rel_mean": _safe_mean(s_len),
            "angle_mae": _safe_mean(s_ang),
            "angle_max_median": float(np.median(s_ang_max)),
            "pulled_to_90_rate": float(pulled),
            "truth_orth_dev_mean": _safe_mean(truth_dev),
            "pred_orth_dev_mean": _safe_mean(pred_dev),
            "fail_mode_rates": {k: v / len(subset) for k, v in modes.items()},
            "length_rel_percentiles": _percentile(s_len, [50, 75, 90]),
            "angle_mae_percentiles": _percentile(s_ang, [50, 75, 90]),
        }

    # Peak-count bins
    bins = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)]
    by_peak: dict[str, Any] = {}
    for lo, hi in bins:
        subset = [r for r in records if lo <= r["peak_num"] <= hi]
        if not subset:
            continue
        label = f"{lo}-{hi if hi < 10_000 else 'inf'}"
        by_peak[label] = {
            "count": len(subset),
            "elem_ok_rate": float(np.mean([r["elem_ok_strict"] for r in subset])),
            "length_rel_mean": _safe_mean(np.array([r["length_rel_mean"] for r in subset])),
            "angle_mae": _safe_mean(np.array([r["angle_abs_mae"] for r in subset])),
        }

    # CS confusion (target vs post-hoc pred from Bravais snap)
    confusion = np.zeros((len(CRYSTAL_SYSTEMS), len(CRYSTAL_SYSTEMS) + 1), dtype=np.int64)
    # last column = identity / unknown (-1)
    for r in records:
        t = r["target_cs_idx"]
        p = r["pred_cs_idx"]
        if t < 0:
            continue
        p_col = p if p >= 0 else len(CRYSTAL_SYSTEMS)
        confusion[t, p_col] += 1

    # Correlation peak_num vs errors
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # Angle component breakdown (which angle hurts most)
    ang_comp = {name: _safe_mean(np.array([r["per_angle_abs"][name] for r in records])) for name in ANGLE_NAMES}
    len_comp = {name: _safe_mean(np.array([r["per_length_rel"][name] for r in records])) for name in LENGTH_NAMES}

    # Distance-to-gate: among failures, how far
    fails = [r for r in records if not r["elem_ok_strict"]]
    near_miss = {
        "n_fail": len(fails),
        "frac_length_within_0.10": float(np.mean([r["length_rel_max"] <= 0.10 for r in fails])) if fails else float("nan"),
        "frac_angle_within_5": float(np.mean([r["angle_abs_max"] <= 5.0 for r in fails])) if fails else float("nan"),
        "frac_angle_within_10": float(np.mean([r["angle_abs_max"] <= 10.0 for r in fails])) if fails else float("nan"),
        "median_length_rel_max": float(np.median([r["length_rel_max"] for r in fails])) if fails else float("nan"),
        "median_angle_abs_max": float(np.median([r["angle_abs_max"] for r in fails])) if fails else float("nan"),
    }

    return {
        "n_samples": n,
        "ltol": ltol,
        "atol_deg": atol_deg,
        "overall": {
            "elem_ok_rate": float(np.mean([r["elem_ok_strict"] for r in records])),
            "length_ok_rate": float(np.mean([r["length_ok_strict"] for r in records])),
            "angle_ok_rate": float(np.mean([r["angle_ok_strict"] for r in records])),
            "length_rel_mean": _safe_mean(length_rel),
            "angle_mae": _safe_mean(angle_mae),
            "abs_log_volume_mean": _safe_mean(abs_log_v),
            "pulled_to_90_rate": float(np.mean([r["pulled_to_90"] for r in records])),
            "length_rel_percentiles": _percentile(length_rel, [25, 50, 75, 90, 95]),
            "length_rel_max_percentiles": _percentile(length_rel_max, [25, 50, 75, 90, 95]),
            "angle_mae_percentiles": _percentile(angle_mae, [25, 50, 75, 90, 95]),
            "angle_max_percentiles": _percentile(angle_max, [25, 50, 75, 90, 95]),
            "frac_length_rel_max": frac_len_le,
            "frac_angle_abs_max": frac_ang_le,
            "fail_mode_rates": {k: v / n for k, v in fail_counts.items()},
            "per_length_rel_mean": len_comp,
            "per_angle_abs_mean": ang_comp,
            "corr_peak_num_vs_angle_mae": _corr(peak_num, angle_mae),
            "corr_peak_num_vs_length_rel": _corr(peak_num, length_rel),
            "near_miss_among_failures": near_miss,
        },
        "by_crystal_system": by_cs,
        "by_peak_num_bin": by_peak,
        "cs_confusion_target_rows_pred_cols": {
            "row_labels": list(CRYSTAL_SYSTEMS),
            "col_labels": list(CRYSTAL_SYSTEMS) + ["identity_or_unknown"],
            "matrix": confusion.tolist(),
        },
    }


def _run_valid(
    model,
    normalizer,
    config: TrainConfig,
    device: torch.device,
    *,
    ltol: float,
    atol_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(
                batch_t["pxrd_x"],
                batch_t["pxrd_y"],
                batch_t["peak_num"],
                crystal_system_idx=batch_t["crystal_system_idx"],
                cs_route=getattr(config.model, "cs_route", "oracle"),
                lattice_phys=batch_t["lattice"],
                setting_route=getattr(config.model, "setting_route", "oracle"),
            )
            pred = normalizer.denormalize(outputs["lattice_norm"]).cpu().numpy()
            truth = batch_t["lattice"].cpu().numpy()
            pred_cs = infer_crystal_system_idx_from_lattice(pred)
            for i in range(pred.shape[0]):
                cs_idx = int(batch_t["crystal_system_idx"][i].item())
                cs = CRYSTAL_SYSTEMS[cs_idx]
                records.append(
                    _sample_record(
                        pred=pred[i],
                        truth=truth[i],
                        crystal_system=cs,
                        peak_num=int(batch_t["peak_num"][i].item()),
                        pred_cs_idx=int(pred_cs[i]),
                        ltol=ltol,
                        atol_deg=atol_deg,
                    )
                )
    return records, _aggregate(records, ltol=ltol, atol_deg=atol_deg)


def _run_mp100(
    model,
    normalizer,
    device: torch.device,
    *,
    mp100_dir: Path,
    ltol: float,
    atol_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_mp100_dataset(mp100_dir)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            pxrd_x_np, pxrd_y_np, peak_num_i = peaks_to_model_tensors(
                sample.two_theta, sample.intensity
            )
            pxrd_x = torch.from_numpy(pxrd_x_np).to(device)
            pxrd_y = torch.from_numpy(pxrd_y_np).to(device)
            peak_num = torch.tensor([peak_num_i], dtype=torch.long, device=device)
            outputs = model(pxrd_x, pxrd_y, peak_num)
            pred = normalizer.denormalize(outputs["lattice_norm"]).cpu().numpy()[0]
            truth = np.asarray(sample.truth_lattice, dtype=np.float64)
            pred_cs = int(infer_crystal_system_idx_from_lattice(pred.reshape(1, 6))[0])
            records.append(
                _sample_record(
                    pred=pred,
                    truth=truth,
                    crystal_system=sample.crystal_system,
                    peak_num=int(peak_num_i),
                    pred_cs_idx=pred_cs,
                    ltol=ltol,
                    atol_deg=atol_deg,
                    sample_id=sample.sample_id,
                )
            )
    return records, _aggregate(records, ltol=ltol, atol_deg=atol_deg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose raw lattice regression errors")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="diag")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--mp100-dir", type=Path, default=PROJECT_ROOT / "data" / "MP-100samples-benchmark")
    p.add_argument("--skip-mp100", action="store_true")
    p.add_argument("--save-samples", action="store_true", help="Write full per-sample jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config)
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(ckpt, config, device)
    model.set_normalizer(normalizer)
    model.eval()

    out_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_records, valid_summary = _run_valid(
        model, normalizer, config, device, ltol=args.ltol, atol_deg=args.atol_deg
    )
    payload: dict[str, Any] = {
        "tag": args.tag,
        "experiment": experiment_name,
        "checkpoint": str(ckpt),
        "config": str(config_path),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "valid": valid_summary,
    }
    if args.save_samples:
        sample_path = out_dir / f"{args.tag}_valid_samples.jsonl"
        with sample_path.open("w", encoding="utf-8") as handle:
            for rec in valid_records:
                handle.write(json.dumps(rec) + "\n")
        payload["valid_samples_path"] = str(sample_path)

    if not args.skip_mp100:
        mp_records, mp_summary = _run_mp100(
            model,
            normalizer,
            device,
            mp100_dir=args.mp100_dir if args.mp100_dir.is_absolute() else (PROJECT_ROOT / args.mp100_dir),
            ltol=args.ltol,
            atol_deg=args.atol_deg,
        )
        payload["mp100"] = mp_summary
        if args.save_samples:
            sample_path = out_dir / f"{args.tag}_mp100_samples.jsonl"
            with sample_path.open("w", encoding="utf-8") as handle:
                for rec in mp_records:
                    handle.write(json.dumps(rec) + "\n")
            payload["mp100_samples_path"] = str(sample_path)

    out_path = out_dir / f"{args.tag}_raw_diagnosis.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"wrote": str(out_path), "valid_elem": valid_summary["overall"]["elem_ok_rate"]}, indent=2))


if __name__ == "__main__":
    main()
