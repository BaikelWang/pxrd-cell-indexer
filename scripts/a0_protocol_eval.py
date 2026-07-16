#!/usr/bin/env python3
"""A0 protocol evaluation: dual-route raw Top-1 metrics with checkpoint inherit.

Produces canonical valid1400 / MP100 JSON under results/a0/ for Gate checks.
Does not train. Default: R10-slim Niggli, --rerank none, strict 0.05/3°.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.dataset import (
    PXRDDatasetConfig,
    PeakFilterConfig,
    SpectrumAugmentConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import build_a0_metrics_block, infer_crystal_system_idx_from_lattice
from pxrd_cell_indexing.training.checkpoint import (
    apply_checkpoint_protocol_to_config,
    infer_canonical_convention_from_checkpoint,
    load_indexing_model_from_checkpoint,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "scale_100k_r10_slim_film_niggli.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "scale_100k_r10_slim_film_niggli_seed42"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "a0"


def _run_dual_route_valid(
    *,
    model,
    normalizer,
    loader,
    device: torch.device,
    ltol: float,
    atol_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds_pred: list[list[float]] = []
    preds_oracle: list[list[float]] = []
    truths: list[list[float]] = []
    peak_counts: list[int] = []
    target_cs: list[int] = []
    classifier_cs: list[int] = []
    per_sample: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            pxrd_x = batch["pxrd_x"].to(device)
            pxrd_y = batch["pxrd_y"].to(device)
            peak_num = batch["peak_num"].to(device)
            cs_idx = batch["crystal_system_idx"].to(device)
            lattice = batch["lattice"]

            out_pred = model(
                pxrd_x,
                pxrd_y,
                peak_num,
                crystal_system_idx=cs_idx,
                cs_route="predicted",
                lattice_phys=lattice.to(device),
            )
            out_oracle = model(
                pxrd_x,
                pxrd_y,
                peak_num,
                crystal_system_idx=cs_idx,
                cs_route="oracle",
                lattice_phys=lattice.to(device),
            )
            pred = normalizer.denormalize(out_pred["lattice_norm"]).cpu()
            oracle = normalizer.denormalize(out_oracle["lattice_norm"]).cpu()
            logits = out_pred.get("crystal_system_logits")
            clf = (
                logits.argmax(dim=-1).cpu().tolist()
                if logits is not None
                else [-1] * pred.shape[0]
            )
            for i in range(pred.shape[0]):
                p = pred[i].tolist()
                o = oracle[i].tolist()
                t = lattice[i].tolist()
                pk = int(peak_num[i].item())
                tcs = int(cs_idx[i].item())
                ccs = int(clf[i])
                preds_pred.append(p)
                preds_oracle.append(o)
                truths.append(t)
                peak_counts.append(pk)
                target_cs.append(tcs)
                classifier_cs.append(ccs)
                sample_id = batch.get("material_id") or batch.get("id")
                sid = None
                if sample_id is not None:
                    sid = sample_id[i] if hasattr(sample_id, "__getitem__") else None
                    if torch.is_tensor(sid):
                        sid = sid.item() if sid.numel() == 1 else str(sid)
                per_sample.append(
                    {
                        "id": sid,
                        "peak_num": pk,
                        "target_cs_idx": tcs,
                        "classifier_cs_idx": ccs,
                        "pred_predicted_cs_route": p,
                        "pred_oracle_cs_route": o,
                        "truth": t,
                    }
                )

    metrics = build_a0_metrics_block(
        preds_predicted=preds_pred,
        preds_oracle=preds_oracle,
        targets=truths,
        peak_counts=peak_counts,
        target_cs_idx=target_cs,
        classifier_cs_idx=classifier_cs if any(c >= 0 for c in classifier_cs) else None,
        ltol=ltol,
        atol_deg=atol_deg,
    )
    return metrics, per_sample


def _run_mp100_raw(
    *,
    model,
    normalizer,
    samples,
    device: torch.device,
    batch_size: int,
    ltol: float,
    atol_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds_pred: list[list[float]] = []
    preds_oracle: list[list[float]] = []
    truths: list[list[float]] = []
    peak_counts: list[int] = []
    target_cs: list[int] = []
    classifier_cs: list[int] = []
    per_sample: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            pxrd_x_parts = []
            pxrd_y_parts = []
            peak_nums = []
            batch_truths = []
            batch_cs = []
            batch_ids = []
            for sample in batch:
                pxrd_x, pxrd_y, peak_num = peaks_to_model_tensors(
                    sample.two_theta, sample.intensity
                )
                pxrd_x_parts.append(torch.from_numpy(pxrd_x))
                pxrd_y_parts.append(torch.from_numpy(pxrd_y))
                peak_nums.append(int(sample.peak_num))
                batch_truths.append(sample.truth_lattice.tolist())
                batch_cs.append(CRYSTAL_SYSTEM_TO_IDX[sample.crystal_system])
                batch_ids.append(sample.sample_id)

            pxrd_x = torch.cat(pxrd_x_parts, dim=0).to(device)
            pxrd_y = torch.cat(pxrd_y_parts, dim=0).to(device)
            peak_num = torch.tensor(peak_nums, dtype=torch.long, device=device)
            cs_idx = torch.tensor(batch_cs, dtype=torch.long, device=device)
            lattice = torch.tensor(batch_truths, dtype=torch.float32, device=device)

            out_pred = model(
                pxrd_x,
                pxrd_y,
                peak_num,
                crystal_system_idx=cs_idx,
                cs_route="predicted",
                lattice_phys=lattice,
            )
            out_oracle = model(
                pxrd_x,
                pxrd_y,
                peak_num,
                crystal_system_idx=cs_idx,
                cs_route="oracle",
                lattice_phys=lattice,
            )
            pred = normalizer.denormalize(out_pred["lattice_norm"]).cpu()
            oracle = normalizer.denormalize(out_oracle["lattice_norm"]).cpu()
            logits = out_pred.get("crystal_system_logits")
            clf = (
                logits.argmax(dim=-1).cpu().tolist()
                if logits is not None
                else [-1] * pred.shape[0]
            )
            for i in range(pred.shape[0]):
                p = pred[i].tolist()
                o = oracle[i].tolist()
                t = batch_truths[i]
                pk = int(peak_nums[i])
                preds_pred.append(p)
                preds_oracle.append(o)
                truths.append(t)
                peak_counts.append(pk)
                target_cs.append(batch_cs[i])
                classifier_cs.append(int(clf[i]))
                per_sample.append(
                    {
                        "id": batch_ids[i],
                        "peak_num": pk,
                        "target_cs_idx": batch_cs[i],
                        "classifier_cs_idx": int(clf[i]),
                        "pred_predicted_cs_route": p,
                        "pred_oracle_cs_route": o,
                        "truth": t,
                    }
                )

    metrics = build_a0_metrics_block(
        preds_predicted=preds_pred,
        preds_oracle=preds_oracle,
        targets=truths,
        peak_counts=peak_counts,
        target_cs_idx=target_cs,
        classifier_cs_idx=classifier_cs if any(c >= 0 for c in classifier_cs) else None,
        ltol=ltol,
        atol_deg=atol_deg,
    )
    return metrics, per_sample


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config)
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    ckpt_path = (
        args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    )
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = apply_checkpoint_protocol_to_config(config, raw_ckpt)
    convention = args.convention or infer_canonical_convention_from_checkpoint(raw_ckpt)
    model, checkpoint, experiment_name = load_indexing_model_from_checkpoint(
        ckpt_path, config, device
    )
    normalizer = build_lattice_normalizer(config.data)
    model.set_normalizer(normalizer)

    out_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "protocol": "A0",
        "experiment_name": experiment_name,
        "checkpoint": str(ckpt_path),
        "representation": config.data.representation,
        "canonical_convention": convention,
        "lattice_stats": config.data.lattice_stats,
        "strict_ltol": args.ltol,
        "strict_atol_deg": args.atol_deg,
        "rerank": "none",
        "candidate": "raw_top1",
    }

    # --- valid1400 ---
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(
            intensity_min=config.model.intensity_min,
            max_peaks=config.model.max_peaks,
        ),
        xrd_augment=False,
        augment=SpectrumAugmentConfig(shift_range=0.0),
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    valid_metrics, valid_per_sample = _run_dual_route_valid(
        model=model,
        normalizer=normalizer,
        loader=loader,
        device=device,
        ltol=args.ltol,
        atol_deg=args.atol_deg,
    )
    # Consistency check: run twice on first N is expensive; instead recompute from dumps.
    valid_metrics_2, _ = _run_dual_route_valid(
        model=model,
        normalizer=normalizer,
        loader=loader,
        device=device,
        ltol=args.ltol,
        atol_deg=args.atol_deg,
    )
    gap = abs(
        valid_metrics["strict_raw_top1_elementwise_rate"]
        - valid_metrics_2["strict_raw_top1_elementwise_rate"]
    )
    valid_payload = {
        **report,
        "split": "valid1400",
        "valid_jsonl": config.data.valid_jsonl,
        "metrics": valid_metrics,
        "reproduce_gap_pp": float(gap * 100.0),
        "reproduce_ok": gap <= 0.005,
        "n_samples": len(valid_per_sample),
        "per_sample": valid_per_sample,
    }
    valid_path = out_dir / "r10_slim_valid1400_a0_canonical.json"
    valid_path.write_text(json.dumps(valid_payload, indent=2), encoding="utf-8")
    report["valid1400"] = {
        "path": str(valid_path),
        "metrics": valid_metrics,
        "reproduce_gap_pp": float(gap * 100.0),
        "reproduce_ok": gap <= 0.005,
    }

    # --- MP100 ---
    if not args.skip_mp100:
        mp100_dir = (
            args.mp100_dir if args.mp100_dir.is_absolute() else (PROJECT_ROOT / args.mp100_dir)
        )
        samples = load_mp100_dataset(mp100_dir, convention=convention)
        mp_metrics, mp_per_sample = _run_mp100_raw(
            model=model,
            normalizer=normalizer,
            samples=samples,
            device=device,
            batch_size=args.batch_size,
            ltol=args.ltol,
            atol_deg=args.atol_deg,
        )
        mp_payload = {
            **report,
            "split": "mp100",
            "mp100_dir": str(mp100_dir),
            "metrics": mp_metrics,
            "n_samples": len(mp_per_sample),
            "per_sample": mp_per_sample,
        }
        mp_path = out_dir / "r10_slim_mp100_a0_canonical.json"
        mp_path.write_text(json.dumps(mp_payload, indent=2), encoding="utf-8")
        report["mp100"] = {"path": str(mp_path), "metrics": mp_metrics}

    summary_path = out_dir / "a0_summary.json"
    # Drop per_sample from nested report before writing summary.
    slim = {
        k: v
        for k, v in report.items()
        if k not in ("valid1400", "mp100")
    }
    slim["valid1400"] = {
        "path": report["valid1400"]["path"],
        "metrics": report["valid1400"]["metrics"],
        "reproduce_gap_pp": report["valid1400"]["reproduce_gap_pp"],
        "reproduce_ok": report["valid1400"]["reproduce_ok"],
    }
    if "mp100" in report:
        slim["mp100"] = {
            "path": report["mp100"]["path"],
            "metrics": report["mp100"]["metrics"],
        }
    summary_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    report["summary_path"] = str(summary_path)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A0 protocol eval (valid1400 + MP100 raw)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--mp100-dir", type=Path, default=PROJECT_ROOT / "data" / "MP-100samples-benchmark")
    p.add_argument("--convention", type=str, default=None, help="Override; default inherit from ckpt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--skip-mp100", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    m = report["valid1400"]["metrics"]
    print("=== A0 valid1400 ===")
    print(f"strict elem: {m['strict_raw_top1_elementwise_rate']*100:.2f}%")
    print(f"classifier CS: {m.get('classifier_cs_accuracy')}")
    print(f"lattice-inferred CS: {m.get('lattice_inferred_cs_accuracy')}")
    print(f"oracle CS elem: {m.get('oracle_cs_strict_elementwise_rate')}")
    print(f"route gap pp: {m.get('oracle_predicted_route_gap_pp')}")
    print(f"cs_correct subset elem: {m.get('cs_correct_subset_lattice_elementwise')}")
    print(f"reproduce_ok: {report['valid1400']['reproduce_ok']}")
    if "mp100" in report:
        mm = report["mp100"]["metrics"]
        print("=== A0 MP100 ===")
        print(f"strict elem: {mm['strict_raw_top1_elementwise_rate']*100:.2f}%")
        print(f"classifier CS: {mm.get('classifier_cs_accuracy')}")
    print(f"summary: {report['summary_path']}")


if __name__ == "__main__":
    main()
