#!/usr/bin/env python3
"""Validate primitive-cell Bravais constraint hypotheses on real data.

Read-only analysis for decision A (geometry snap Top-K). Scans:
  - train100k_seed42.jsonl (LMDB-backed structures via global_idx)
  - MP100 CIF benchmark (cross-check)

Outputs:
  - results/bravais_constraint_validation.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "train100k_seed42.jsonl"
DEFAULT_TRAIN_LMDB = Path(
    "/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"
)
DEFAULT_MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "bravais_constraint_validation.json"
DEFAULT_SYMPREC = 0.01

LENGTH_RTOLS = (0.01, 0.02, 0.05)
ANGLE_ATOLS = (1.0, 2.0, 5.0)
MIN_GROUP_N = 30
CLEAR_THRESHOLD = 0.95

# Candidate hypotheses for primitive-cell snap constraints.
HYPOTHESES: dict[str, str] = {
    "a_eq_b": "a ≈ b (length ratio)",
    "b_eq_c": "b ≈ c (length ratio)",
    "a_eq_c": "a ≈ c (length ratio)",
    "a_eq_b_eq_c": "a ≈ b ≈ c (all lengths)",
    "alpha_eq_90": "α ≈ 90°",
    "beta_eq_90": "β ≈ 90°",
    "gamma_eq_90": "γ ≈ 90°",
    "alpha_eq_60": "α ≈ 60°",
    "beta_eq_60": "β ≈ 60°",
    "gamma_eq_60": "γ ≈ 60°",
    "alpha_eq_109_47": "α ≈ 109.47° (cubic-I primitive)",
    "beta_eq_109_47": "β ≈ 109.47°",
    "gamma_eq_109_47": "γ ≈ 109.47°",
    "gamma_eq_120": "γ ≈ 120° (hex/trig primitive)",
    "alpha_eq_beta": "α ≈ β",
    "beta_eq_gamma": "β ≈ γ",
    "alpha_eq_beta_eq_gamma": "α ≈ β ≈ γ",
    "all_angles_90": "α ≈ β ≈ γ ≈ 90°",
    "all_angles_60": "α ≈ β ≈ γ ≈ 60°",
}

_WORKER_KEYS: list[bytes] | None = None
_WORKER_ENV: lmdb.Environment | None = None


@dataclass
class AnalyzedSample:
    source: str
    sample_id: str
    crystal_system: str
    bravais_letter: str
    space_group_symbol: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    stored_crystal_system: str | None = None
    cs_match_stored: bool | None = None
    lattice_max_abs_diff_stored: float | None = None
    error: str | None = None


@dataclass
class GroupStats:
    crystal_system: str
    bravais_letter: str
    n: int
    low_confidence: bool
    length_ratio_median: dict[str, float] = field(default_factory=dict)
    length_ratio_iqr: dict[str, float] = field(default_factory=dict)
    angle_median: dict[str, float] = field(default_factory=dict)
    angle_iqr: dict[str, float] = field(default_factory=dict)
    hypothesis_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    clear_constraints: list[str] = field(default_factory=list)
    verdict: str = "unknown"


def bravais_letter_from_space_group(symbol: str) -> str:
    """Map Hermann-Mauguin symbol first letter to Bravais centering type."""
    if not symbol:
        return "?"
    letter = symbol[0].upper()
    if letter in {"A", "B"}:
        return "C"  # base-centered variants grouped with C
    if letter in {"P", "I", "F", "C", "R"}:
        return letter
    return "?"


def _init_worker(db_path: str, keys: list[bytes]) -> None:
    global _WORKER_ENV, _WORKER_KEYS
    _WORKER_ENV = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    _WORKER_KEYS = keys


def _load_entry(global_idx: int) -> dict[str, Any]:
    assert _WORKER_ENV is not None and _WORKER_KEYS is not None
    key = _WORKER_KEYS[global_idx]
    raw = _WORKER_ENV.begin().get(key)
    if raw is None:
        raise KeyError(f"missing key at index {global_idx}")
    return pickle.loads(gzip.decompress(raw))


def _lengths_close(
    values: tuple[float, float, float],
    rtol: float,
) -> bool:
    a, b, c = values
    denom = max(abs(a), abs(b), abs(c), 1e-6)
    return (
        abs(a - b) / denom <= rtol
        and abs(b - c) / denom <= rtol
        and abs(a - c) / denom <= rtol
    )


def _pair_close(x: float, y: float, rtol: float) -> bool:
    denom = max(abs(x), abs(y), 1e-6)
    return abs(x - y) / denom <= rtol


def _angle_pair_close(x: float, y: float, atol_deg: float) -> bool:
    return abs(x - y) <= atol_deg


def _angle_close(value: float, target: float, atol_deg: float) -> bool:
    return abs(value - target) <= atol_deg


def _angles_all_close(
    angles: tuple[float, float, float],
    target: float,
    atol_deg: float,
) -> bool:
    return all(_angle_close(v, target, atol_deg) for v in angles)


def evaluate_hypothesis(
    params: tuple[float, float, float, float, float, float],
    name: str,
    *,
    rtol: float,
    atol_deg: float,
) -> bool:
    a, b, c, alpha, beta, gamma = params
    if name == "a_eq_b":
        return _pair_close(a, b, rtol)
    if name == "b_eq_c":
        return _pair_close(b, c, rtol)
    if name == "a_eq_c":
        return _pair_close(a, c, rtol)
    if name == "a_eq_b_eq_c":
        return _lengths_close((a, b, c), rtol)
    if name == "alpha_eq_90":
        return _angle_close(alpha, 90.0, atol_deg)
    if name == "beta_eq_90":
        return _angle_close(beta, 90.0, atol_deg)
    if name == "gamma_eq_90":
        return _angle_close(gamma, 90.0, atol_deg)
    if name == "alpha_eq_60":
        return _angle_close(alpha, 60.0, atol_deg)
    if name == "beta_eq_60":
        return _angle_close(beta, 60.0, atol_deg)
    if name == "gamma_eq_60":
        return _angle_close(gamma, 60.0, atol_deg)
    if name == "alpha_eq_109_47":
        return _angle_close(alpha, 109.47122063449069, atol_deg)
    if name == "beta_eq_109_47":
        return _angle_close(beta, 109.47122063449069, atol_deg)
    if name == "gamma_eq_109_47":
        return _angle_close(gamma, 109.47122063449069, atol_deg)
    if name == "gamma_eq_120":
        return _angle_close(gamma, 120.0, atol_deg)
    if name == "alpha_eq_beta":
        return _angle_pair_close(alpha, beta, atol_deg)
    if name == "beta_eq_gamma":
        return _angle_pair_close(beta, gamma, atol_deg)
    if name == "alpha_eq_beta_eq_gamma":
        return (
            _angle_pair_close(alpha, beta, atol_deg)
            and _angle_pair_close(beta, gamma, atol_deg)
            and _angle_pair_close(alpha, gamma, atol_deg)
        )
    if name == "all_angles_90":
        return _angles_all_close((alpha, beta, gamma), 90.0, atol_deg)
    if name == "all_angles_60":
        return _angles_all_close((alpha, beta, gamma), 60.0, atol_deg)
    raise KeyError(name)


def analyze_structure(
    structure: Structure,
    *,
    source: str,
    sample_id: str,
    symprec: float,
    stored_crystal_system: str | None = None,
    stored_lattice: tuple[float, ...] | None = None,
) -> AnalyzedSample:
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        crystal_system = analyzer.get_crystal_system()
        space_group_symbol = analyzer.get_space_group_symbol()
        bravais_letter = bravais_letter_from_space_group(space_group_symbol)
        primitive = analyzer.find_primitive()
        lat = primitive.lattice
        params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)

        cs_match = None
        lattice_diff = None
        if stored_crystal_system is not None:
            cs_match = stored_crystal_system == crystal_system
        if stored_lattice is not None and len(stored_lattice) == 6:
            recomputed = np.array(params, dtype=np.float64)
            stored = np.array(stored_lattice, dtype=np.float64)
            lattice_diff = float(np.max(np.abs(recomputed - stored)))

        return AnalyzedSample(
            source=source,
            sample_id=sample_id,
            crystal_system=crystal_system,
            bravais_letter=bravais_letter,
            space_group_symbol=space_group_symbol,
            a=params[0],
            b=params[1],
            c=params[2],
            alpha=params[3],
            beta=params[4],
            gamma=params[5],
            stored_crystal_system=stored_crystal_system,
            cs_match_stored=cs_match,
            lattice_max_abs_diff_stored=lattice_diff,
        )
    except Exception as exc:  # noqa: BLE001
        return AnalyzedSample(
            source=source,
            sample_id=sample_id,
            crystal_system="error",
            bravais_letter="?",
            space_group_symbol="",
            a=math.nan,
            b=math.nan,
            c=math.nan,
            alpha=math.nan,
            beta=math.nan,
            gamma=math.nan,
            stored_crystal_system=stored_crystal_system,
            error=str(exc),
        )


def _analyze_train_record(payload: dict[str, Any], symprec: float) -> AnalyzedSample:
    global_idx = int(payload["global_idx"])
    data = _load_entry(global_idx)
    lattice = Lattice(data["p_lattice_matrix"])
    structure = Structure(
        lattice,
        data["p_atom_type"],
        data["p_atom_pos"],
        coords_are_cartesian=False,
    )
    stored_lattice = (
        payload.get("lattice_a"),
        payload.get("lattice_b"),
        payload.get("lattice_c"),
        payload.get("lattice_alpha"),
        payload.get("lattice_beta"),
        payload.get("lattice_gamma"),
    )
    return analyze_structure(
        structure,
        source="train100k",
        sample_id=str(payload.get("lmdb_key", global_idx)),
        symprec=symprec,
        stored_crystal_system=payload.get("crystal_system"),
        stored_lattice=stored_lattice,
    )


def load_train_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def analyze_train100k(
    jsonl_path: Path,
    lmdb_path: Path,
    *,
    symprec: float,
    workers: int,
    limit: int | None,
) -> list[AnalyzedSample]:
    records = load_train_jsonl(jsonl_path)
    if limit is not None:
        records = records[:limit]

    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    env.close()

    worker_count = min(workers, cpu_count(), max(1, len(records)))
    analyze_fn = partial(_analyze_train_record, symprec=symprec)

    with Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(str(lmdb_path), keys),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(analyze_fn, records, chunksize=64),
                total=len(records),
                desc="train100k",
            )
        )
    return results


def analyze_mp100(
    cif_dir: Path,
    *,
    symprec: float,
) -> list[AnalyzedSample]:
    samples: list[AnalyzedSample] = []
    cif_paths = sorted(cif_dir.glob("*.cif"))
    for cif_path in tqdm(cif_paths, desc="mp100"):
        structure = Structure.from_file(cif_path)
        samples.append(
            analyze_structure(
                structure,
                source="mp100",
                sample_id=cif_path.stem,
                symprec=symprec,
            )
        )
    return samples


def _percentile_iqr(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return float(q50), float(q75 - q25)


def aggregate_groups(samples: list[AnalyzedSample]) -> list[GroupStats]:
    valid = [s for s in samples if s.error is None and s.crystal_system != "error"]
    grouped: dict[tuple[str, str], list[AnalyzedSample]] = defaultdict(list)
    for sample in valid:
        grouped[(sample.crystal_system, sample.bravais_letter)].append(sample)

    stats_list: list[GroupStats] = []
    for (crystal_system, bravais_letter), group in sorted(grouped.items()):
        n = len(group)
        low_confidence = n < MIN_GROUP_N

        a = np.array([s.a for s in group], dtype=np.float64)
        b = np.array([s.b for s in group], dtype=np.float64)
        c = np.array([s.c for s in group], dtype=np.float64)
        alpha = np.array([s.alpha for s in group], dtype=np.float64)
        beta = np.array([s.beta for s in group], dtype=np.float64)
        gamma = np.array([s.gamma for s in group], dtype=np.float64)

        ab_ratio = a / np.maximum(b, 1e-6)
        bc_ratio = b / np.maximum(c, 1e-6)
        ac_ratio = a / np.maximum(c, 1e-6)

        length_ratio_median: dict[str, float] = {}
        length_ratio_iqr: dict[str, float] = {}
        for name, arr in [
            ("a_over_b", ab_ratio),
            ("b_over_c", bc_ratio),
            ("a_over_c", ac_ratio),
        ]:
            med, iqr = _percentile_iqr(arr)
            length_ratio_median[name] = med
            length_ratio_iqr[name] = iqr

        angle_median: dict[str, float] = {}
        angle_iqr: dict[str, float] = {}
        for name, arr in [
            ("alpha", alpha),
            ("beta", beta),
            ("gamma", gamma),
        ]:
            med, iqr = _percentile_iqr(arr)
            angle_median[name] = med
            angle_iqr[name] = iqr

        hypothesis_rates: dict[str, dict[str, float]] = {}
        clear_constraints: list[str] = []
        for hyp_name in HYPOTHESES:
            rates: dict[str, float] = {}
            for rtol in LENGTH_RTOLS:
                for atol in ANGLE_ATOLS:
                    key = f"rtol={rtol:.2f},atol={atol:.1f}"
                    hits = 0
                    for sample in group:
                        params = (
                            sample.a,
                            sample.b,
                            sample.c,
                            sample.alpha,
                            sample.beta,
                            sample.gamma,
                        )
                        if evaluate_hypothesis(
                            params,
                            hyp_name,
                            rtol=rtol,
                            atol_deg=atol,
                        ):
                            hits += 1
                    rate = hits / n
                    rates[key] = rate
            hypothesis_rates[hyp_name] = rates

            # Clear if >95% at rtol=2% and atol=2° (primary decision tolerance).
            primary_key = "rtol=0.02,atol=2.0"
            if rates.get(primary_key, 0.0) >= CLEAR_THRESHOLD:
                clear_constraints.append(hyp_name)

        if low_confidence:
            verdict = "low_confidence"
        elif clear_constraints:
            verdict = "clear"
        else:
            verdict = "chaotic"

        stats_list.append(
            GroupStats(
                crystal_system=crystal_system,
                bravais_letter=bravais_letter,
                n=n,
                low_confidence=low_confidence,
                length_ratio_median=length_ratio_median,
                length_ratio_iqr=length_ratio_iqr,
                angle_median=angle_median,
                angle_iqr=angle_iqr,
                hypothesis_rates=hypothesis_rates,
                clear_constraints=clear_constraints,
                verdict=verdict,
            )
        )
    return stats_list


def summarize_cross_checks(samples: list[AnalyzedSample]) -> dict[str, Any]:
    with_stored = [s for s in samples if s.stored_crystal_system is not None and s.error is None]
    cs_matches = [s for s in with_stored if s.cs_match_stored is True]
    lattice_diffs = [
        s.lattice_max_abs_diff_stored
        for s in with_stored
        if s.lattice_max_abs_diff_stored is not None
    ]
    return {
        "n_with_stored_cs": len(with_stored),
        "cs_match_rate": len(cs_matches) / len(with_stored) if with_stored else None,
        "lattice_max_abs_diff_stored": {
            "median": float(np.median(lattice_diffs)) if lattice_diffs else None,
            "p95": float(np.percentile(lattice_diffs, 95)) if lattice_diffs else None,
            "max": float(np.max(lattice_diffs)) if lattice_diffs else None,
        },
    }


def build_report_payload(
    train_samples: list[AnalyzedSample],
    mp100_samples: list[AnalyzedSample],
    *,
    symprec: float,
) -> dict[str, Any]:
    train_groups = aggregate_groups(train_samples)
    mp100_groups = aggregate_groups(mp100_samples)

    def group_key(g: GroupStats) -> str:
        return f"{g.crystal_system}:{g.bravais_letter}"

    train_by_key = {group_key(g): g for g in train_groups}
    mp100_by_key = {group_key(g): g for g in mp100_groups}

    comparison: list[dict[str, Any]] = []
    all_keys = sorted(set(train_by_key) | set(mp100_by_key))
    for key in all_keys:
        train_g = train_by_key.get(key)
        mp100_g = mp100_by_key.get(key)
        comparison.append(
            {
                "group": key,
                "train_n": train_g.n if train_g else 0,
                "train_verdict": train_g.verdict if train_g else None,
                "train_clear": train_g.clear_constraints if train_g else [],
                "mp100_n": mp100_g.n if mp100_g else 0,
                "mp100_verdict": mp100_g.verdict if mp100_g else None,
                "mp100_clear": mp100_g.clear_constraints if mp100_g else [],
            }
        )

    return {
        "meta": {
            "symprec": symprec,
            "length_rtols": list(LENGTH_RTOLS),
            "angle_atols": list(ANGLE_ATOLS),
            "min_group_n": MIN_GROUP_N,
            "clear_threshold": CLEAR_THRESHOLD,
            "primary_tolerance": "rtol=0.02,atol=2.0",
            "hypothesis_labels": HYPOTHESES,
            "bravais_note": "A/B base-centered space groups are grouped under C.",
        },
        "train100k": {
            "n_total": len(train_samples),
            "n_ok": sum(1 for s in train_samples if s.error is None),
            "n_error": sum(1 for s in train_samples if s.error is not None),
            "cross_check": summarize_cross_checks(train_samples),
            "groups": [asdict(g) for g in train_groups],
        },
        "mp100": {
            "n_total": len(mp100_samples),
            "n_ok": sum(1 for s in mp100_samples if s.error is None),
            "n_error": sum(1 for s in mp100_samples if s.error is not None),
            "groups": [asdict(g) for g in mp100_groups],
        },
        "train_vs_mp100_comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--train-lmdb", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--mp100-dir", type=Path, default=DEFAULT_MP100_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symprec", type=float, default=DEFAULT_SYMPREC)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on train100k records (for smoke runs).",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-mp100", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_samples: list[AnalyzedSample] = []
    mp100_samples: list[AnalyzedSample] = []

    if not args.skip_train:
        if not args.train_jsonl.is_file():
            print(f"error: missing train jsonl: {args.train_jsonl}", file=sys.stderr)
            return 1
        if not args.train_lmdb.is_file():
            print(f"error: missing train lmdb: {args.train_lmdb}", file=sys.stderr)
            return 1
        train_samples = analyze_train100k(
            args.train_jsonl,
            args.train_lmdb,
            symprec=args.symprec,
            workers=args.workers,
            limit=args.limit,
        )

    if not args.skip_mp100:
        if not args.mp100_dir.is_dir():
            print(f"error: missing mp100 dir: {args.mp100_dir}", file=sys.stderr)
            return 1
        mp100_samples = analyze_mp100(args.mp100_dir, symprec=args.symprec)

    payload = build_report_payload(
        train_samples,
        mp100_samples,
        symprec=args.symprec,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"wrote {args.output}")
    print(
        "train groups:",
        len(payload["train100k"]["groups"]),
        "| mp100 groups:",
        len(payload["mp100"]["groups"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
