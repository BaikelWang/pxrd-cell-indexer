#!/usr/bin/env python3
"""Run Phase 0-3 regression accuracy experiments sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "phase03_experiments"
DOC_PATH = (
    PROJECT_ROOT / "docs" / "实验记录" / "20260708-回归精度提升Phase0-3.md"
)
BASE_CONFIG = PROJECT_ROOT / "configs" / "scale_100k_no_cs_matrix6.yaml"
BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "scale_100k_no_cs_matrix6_seed42"
    / "checkpoints"
    / "best.pt"
)
TEST_CONFIG = PROJECT_ROOT / "configs" / "scale_100k_no_cs_matrix6_testset.yaml"

LOSS_CONFIGS = [
    ("length_angle", PROJECT_ROOT / "configs" / "scale_100k_loss_length_angle.yaml"),
    ("cs_mask", PROJECT_ROOT / "configs" / "scale_100k_loss_cs_mask.yaml"),
    ("cs_reweight", PROJECT_ROOT / "configs" / "scale_100k_loss_cs_reweight.yaml"),
    ("combined", PROJECT_ROOT / "configs" / "scale_100k_loss_combined.yaml"),
]


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=False)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _append_doc(section: str) -> None:
    marker = "## 9. 实验执行记录"
    text = DOC_PATH.read_text(encoding="utf-8")
    if marker not in text:
        raise RuntimeError(f"Missing section marker in {DOC_PATH}")
    insertion = f"\n{section}\n"
    if insertion.strip() in text:
        return
    DOC_PATH.write_text(text + insertion, encoding="utf-8")


def _eval_dual(
    *,
    label: str,
    config: Path,
    checkpoint: Path,
    output_path: Path,
    dataset: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_valid.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-path",
        str(output_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {label} ({dataset})")
    metrics = _load_json(output_path)["metrics"]
    _append_doc(
        f"### {label} ({dataset}) — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"- checkpoint: `{checkpoint}`\n"
        f"- raw_top1: **{metrics.get('raw_top1_lattice_match_rate', 0):.1%}**\n"
        f"- rerank_top1: **{metrics.get('top1_lattice_match_rate', 0):.1%}**\n"
        f"- fom_top1: **{metrics.get('fom_top1_lattice_match_rate', 0):.1%}**\n"
        f"- topk: **{metrics.get('topk_lattice_match_rate', 0):.1%}**"
    )
    return metrics


def _train_and_eval_loss(name: str, config_path: Path) -> dict[str, Any]:
    train_proc = _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--config", str(config_path)]
    )
    if train_proc.returncode != 0:
        raise RuntimeError(f"training failed for loss mode {name}")

    exp_name = _load_yaml_experiment_name(config_path)
    ckpt = PROJECT_ROOT / "results" / "experiments" / exp_name / "checkpoints" / "best.pt"
    return _eval_dual(
        label=f"loss_{name}",
        config=BASE_CONFIG,
        checkpoint=ckpt,
        output_path=RESULTS_DIR / f"valid1400_loss_{name}.json",
        dataset="valid1400",
    )


def _load_yaml_experiment_name(path: Path) -> str:
    import yaml

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return str(raw["experiment_name"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"steps": []}

    # Step 1: profiling
    profile_out = RESULTS_DIR / "profile_dataloader.json"
    profile_proc = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "profile_dataloader.py"),
            "--config",
            str(BASE_CONFIG),
            "--max-steps",
            str(args.profile_steps),
            "--output-path",
            str(profile_out),
        ]
    )
    if profile_proc.returncode != 0:
        raise RuntimeError("profile_dataloader failed")
    profile = _load_json(profile_out)
    summary["profile"] = profile
    _append_doc(
        f"### Step1 profiling — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"- data_wait: **{profile['data_wait_ms_per_step']:.1f} ms/step**\n"
        f"- compute: **{profile['compute_ms_per_step']:.1f} ms/step**\n"
        f"- bottleneck: **{profile['bottleneck']}**"
    )

    # Step 2: baseline dual eval (existing checkpoint)
    if BASE_CHECKPOINT.exists():
        baseline_metrics = _eval_dual(
            label="baseline_ckpt",
            config=BASE_CONFIG,
            checkpoint=BASE_CHECKPOINT,
            output_path=RESULTS_DIR / "valid1400_baseline_ckpt.json",
            dataset="valid1400",
        )
        summary["baseline_valid"] = baseline_metrics
    else:
        summary["baseline_valid"] = {"skipped": "checkpoint missing"}

    # Step 3: loss ablations
    loss_results: dict[str, Any] = {}
    for name, cfg in LOSS_CONFIGS:
        if args.skip_training:
            continue
        metrics = _train_and_eval_loss(name, cfg)
        loss_results[name] = metrics
    summary["loss_ablations"] = loss_results

    if loss_results:
        best_name = max(
            loss_results,
            key=lambda key: loss_results[key].get("raw_top1_lattice_match_rate", 0.0),
        )
        summary["best_loss_mode"] = best_name
        _append_doc(
            f"### Step3 loss ablation winner — raw_top1 best: **{best_name}** "
            f"({loss_results[best_name]['raw_top1_lattice_match_rate']:.1%})"
        )

        # Step 4: reduced hyperparam sweep on best loss only
        if not args.skip_sweep:
            sweep_out = RESULTS_DIR / f"sweep_{best_name}.json"
            sweep_proc = _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "sweep_train_hyperparams.py"),
                    "--base-config",
                    str(next(cfg for n, cfg in LOSS_CONFIGS if n == best_name)),
                    "--loss-modes",
                    best_name,
                    "--head-lrs",
                    "0.0005",
                    "0.001",
                    "0.002",
                    "--batch-sizes",
                    "64",
                    "128",
                    "--output-path",
                    str(sweep_out),
                ]
            )
            if sweep_proc.returncode == 0 and sweep_out.exists():
                summary["sweep"] = _load_json(sweep_out)
                best_sweep = summary["sweep"].get("best", {})
                _append_doc(
                    f"### Step4 sweep best — head_lr={best_sweep.get('head_lr')}, "
                    f"batch={best_sweep.get('batch_size')}, "
                    f"valid_raw={best_sweep.get('valid_raw_top1_last_epoch', 0):.1%}"
                )

        # Step 5: test1400 ceiling eval for best loss checkpoint
        if not args.skip_training:
            best_exp = _load_yaml_experiment_name(
                next(cfg for n, cfg in LOSS_CONFIGS if n == best_name)
            )
            best_ckpt = (
                PROJECT_ROOT / "results" / "experiments" / best_exp / "checkpoints" / "best.pt"
            )
            if best_ckpt.exists():
                test_metrics = _eval_dual(
                    label=f"best_{best_name}",
                    config=TEST_CONFIG,
                    checkpoint=best_ckpt,
                    output_path=RESULTS_DIR / f"test1400_best_{best_name}.json",
                    dataset="test1400",
                )
                summary["test1400"] = test_metrics

    summary_path = RESULTS_DIR / "phase03_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0-3 experiment pipeline")
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
