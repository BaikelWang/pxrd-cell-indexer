"""R4-P0 (zero-training): post-hoc Bravais-manifold snap on raw Top-1 predictions.

Physical prior test: does projecting the raw regression output onto its own
crystal-system's constrained manifold (a=b for tetragonal, angles=90/120 for
hex, etc. via the existing `model/bravais.py` snap functions) improve strict
elem / angle MAE / pull90, using *no* additional training?

Two routing modes are compared:
  --cs-source oracle    : snap using ground-truth crystal system (upper bound)
  --cs-source predicted : snap using the model's own routed CS (deployable)

For monoclinic we try all three single-free-angle snaps and take the one
closest to the raw prediction (cheapest deployable proxy for "which angle is
the unique one").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.dataset import PXRDDatasetConfig, PeakFilterConfig, build_dataloader
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.model.bravais import (
    _snap_hex_trig_p,
    _snap_monoclinic_p_alpha,
    _snap_monoclinic_p_beta,
    _snap_monoclinic_p_gamma,
    _snap_orthorhombic_p,
    _snap_tetragonal_p,
    _snap_trigonal_r,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SNAP_BY_CS = {
    "tetragonal": _snap_tetragonal_p,
    "orthorhombic": _snap_orthorhombic_p,
    "hexagonal": _snap_hex_trig_p,
    "trigonal": _snap_trigonal_r,
}


def _snap_monoclinic_best(raw: tuple) -> tuple:
    candidates = [
        _snap_monoclinic_p_alpha(raw),
        _snap_monoclinic_p_beta(raw),
        _snap_monoclinic_p_gamma(raw),
    ]

    def dev(snapped: tuple) -> float:
        return sum(abs(snapped[i] - raw[i]) for i in range(3, 6))

    return min(candidates, key=dev)


def snap_lattice(raw_row: np.ndarray, cs_name: str) -> np.ndarray:
    raw = tuple(float(v) for v in raw_row)
    if cs_name == "cubic":
        a, b, c, alpha, beta, gamma = raw
        mean_len = (a + b + c) / 3.0
        mean_ang = (alpha + beta + gamma) / 3.0
        return np.array([mean_len, mean_len, mean_len, mean_ang, mean_ang, mean_ang])
    if cs_name == "monoclinic":
        return np.array(_snap_monoclinic_best(raw))
    if cs_name == "triclinic":
        return raw_row
    fn = _SNAP_BY_CS.get(cs_name)
    if fn is None:
        return raw_row
    return np.array(fn(raw))


def elementwise_ok(pred: np.ndarray, truth: np.ndarray, ltol: float, atol_deg: float) -> bool:
    length_ok = bool(np.all(np.abs(pred[:3] - truth[:3]) / np.maximum(truth[:3], 1e-6) <= ltol))
    angle_ok = bool(np.all(np.abs(pred[3:] - truth[3:]) <= atol_deg))
    return length_ok and angle_ok


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc Bravais snap ablation (no retraining)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    config = TrainConfig.from_yaml(args.config).resolve_paths(PROJECT_ROOT)
    model, _, _ = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    normalizer = build_lattice_normalizer(config.data)
    model.set_normalizer(normalizer)
    model.eval()

    ds = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(ds, batch_size=64, num_workers=2, shuffle=False, pin_memory=True)

    rows = []
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
            pred_phys = normalizer.denormalize(out["lattice_norm"]).cpu().numpy()
            truth = bt["lattice"].cpu().numpy()
            gt_cs = bt["crystal_system_idx"].cpu().numpy()
            pred_cs_idx = out["routed_cs_idx"].cpu().numpy()
            for i in range(pred_phys.shape[0]):
                rows.append(
                    {
                        "pred": pred_phys[i],
                        "truth": truth[i],
                        "gt_cs": CRYSTAL_SYSTEMS[int(gt_cs[i])],
                        "pred_cs": CRYSTAL_SYSTEMS[int(pred_cs_idx[i])],
                    }
                )

    results = {}
    for cs_source in ("oracle", "predicted"):
        n = 0
        elem_raw = elem_snap = 0
        ang_err_raw = []
        ang_err_snap = []
        pull90_snap = 0
        noncub_elem_raw: dict[str, list[bool]] = {}
        noncub_elem_snap: dict[str, list[bool]] = {}
        for row in rows:
            n += 1
            truth = row["truth"]
            raw = row["pred"]
            cs_name = row["gt_cs"] if cs_source == "oracle" else row["pred_cs"]
            snapped = snap_lattice(raw, cs_name)
            ok_raw = elementwise_ok(raw, truth, args.ltol, args.atol_deg)
            ok_snap = elementwise_ok(snapped, truth, args.ltol, args.atol_deg)
            elem_raw += int(ok_raw)
            elem_snap += int(ok_snap)
            ang_err_raw.append(float(np.abs(raw[3:] - truth[3:]).mean()))
            ang_err_snap.append(float(np.abs(snapped[3:] - truth[3:]).mean()))
            pull90_snap += int(np.all(np.abs(snapped[3:] - 90.0) <= 1.0))
            gt_cs = row["gt_cs"]
            if gt_cs != "cubic":
                noncub_elem_raw.setdefault(gt_cs, []).append(ok_raw)
                noncub_elem_snap.setdefault(gt_cs, []).append(ok_snap)

        noncub_rates_raw = [np.mean(v) for v in noncub_elem_raw.values()]
        noncub_rates_snap = [np.mean(v) for v in noncub_elem_snap.values()]
        results[cs_source] = {
            "n": n,
            "elem_raw": elem_raw / n,
            "elem_snap": elem_snap / n,
            "ang_mae_raw": float(np.mean(ang_err_raw)),
            "ang_mae_snap": float(np.mean(ang_err_snap)),
            "pull90_snap": pull90_snap / n,
            "noncub_elem_raw": float(np.mean(noncub_rates_raw)) if noncub_rates_raw else None,
            "noncub_elem_snap": float(np.mean(noncub_rates_snap)) if noncub_rates_snap else None,
        }
        print(
            f"[{cs_source}] elem raw={results[cs_source]['elem_raw']*100:.2f}% "
            f"snap={results[cs_source]['elem_snap']*100:.2f}% | "
            f"ang raw={results[cs_source]['ang_mae_raw']:.2f} "
            f"snap={results[cs_source]['ang_mae_snap']:.2f} | "
            f"pull90(snap)={results[cs_source]['pull90_snap']*100:.1f}% | "
            f"noncub raw={results[cs_source]['noncub_elem_raw']*100:.2f}% "
            f"snap={results[cs_source]['noncub_elem_snap']*100:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
