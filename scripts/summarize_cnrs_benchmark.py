#!/usr/bin/env python3
"""Merge CNRS multi-engine results into a single L4-strict leaderboard.

Sources
-------
* Ours + native McMaille: e2e ``l4_prim_compare.json`` (``indexer_union`` or ``indexer_k100``)
* DICVOL / TREOR / ITO:   results/cnrs_benchmark/<engine>/summary.json
* OpenAlphaDiffract:      results/cnrs_benchmark/openalphadiffract/summary.json

Writes ``results/cnrs_benchmark/l4_prim_leaderboard.json``.
Unavailable engines stay in the table with status=unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--e2e",
        default="results/flow_seedgen/cnrs_e2e_multithresh_v4/l4_prim_compare.json",
    )
    ap.add_argument("--ours-name", default="Ours (v4 wide union K=100×2)")
    ap.add_argument("--bench-dir", default="results/cnrs_benchmark")
    ap.add_argument(
        "--out",
        default="results/cnrs_benchmark/l4_prim_leaderboard.json",
    )
    return ap.parse_args()


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def eng_from_block(name: str, block: dict | None, status: str = "ok", reason: str = "") -> dict:
    if block is None or status == "unavailable":
        return {
            "name": name,
            "status": "unavailable",
            "reason": reason or (block or {}).get("reason", "unavailable"),
            "n": (block or {}).get("n"),
            "prim": None,
            "by_system": {},
            "top1_only": False,
        }
    prim = block.get("prim")
    return {
        "name": name,
        "status": status,
        "n": block.get("n"),
        "prim": prim,
        "by_system": block.get("by_system", {}),
        "top1_only": name == "OpenAlphaDiffract",
        "run_dir": block.get("run_dir") or block.get("model_dir"),
        "binary": block.get("binary"),
        "binary_how": block.get("binary_how"),
    }


def pick_ours_block(e2e: dict) -> tuple[dict | None, str]:
    for key in ("indexer_union", "indexer_k100", "indexer"):
        if key in e2e:
            return e2e[key], key
    return None, ""


def classic_label(key: str, summary: dict) -> str:
    """Prefer the actual binary version when we shipped 91/13 instead of 06/12."""
    binary = str(summary.get("binary") or summary.get("binary_how") or "").lower()
    if key == "dicvol06":
        return "DICVOL91" if "91" in binary else "DICVOL06"
    if key == "ito":
        return "ITO13" if "13" in binary else "ITO"
    if key == "treor90":
        return "TREOR90"
    return key


def main() -> None:
    args = parse_args()
    e2e_path = _abs(args.e2e)
    bench = _abs(args.bench_dir)
    out_path = _abs(args.out)

    e2e = json.loads(e2e_path.read_text()) if e2e_path.exists() else {}
    protocol = {}
    proto_path = bench / "protocol.json"
    if proto_path.exists():
        protocol = json.loads(proto_path.read_text())

    engines = []

    # 1 Ours
    ours_block, ours_key = pick_ours_block(e2e)
    if ours_block is not None:
        engines.append(eng_from_block(args.ours_name, ours_block, status="ok"))
    else:
        engines.append(
            eng_from_block(args.ours_name, None, "unavailable", "missing e2e json")
        )

    # 2 Native Mc
    if "native_mcmaille" in e2e:
        engines.append(eng_from_block("Native McMaille", e2e["native_mcmaille"], "ok"))
    else:
        engines.append(
            eng_from_block("Native McMaille", None, "unavailable", "missing e2e json")
        )

    # 3–5 classic
    for key in ("dicvol06", "treor90", "ito"):
        sp = bench / key / "summary.json"
        if not sp.exists():
            engines.append(
                eng_from_block(key.upper(), None, "unavailable", f"missing {sp}")
            )
            continue
        s = json.loads(sp.read_text())
        st = s.get("status", "ok")
        label = classic_label(key, s)
        engines.append(
            eng_from_block(label, s, status=st, reason=s.get("reason", ""))
        )

    # 6 OpenAlpha
    oad = bench / "openalphadiffract" / "summary.json"
    if oad.exists():
        s = json.loads(oad.read_text())
        engines.append(
            eng_from_block("OpenAlphaDiffract", s, status=s.get("status", "ok"))
        )
    else:
        engines.append(
            eng_from_block(
                "OpenAlphaDiffract", None, "unavailable", f"missing {oad}"
            )
        )

    # Rank available by Top-1 strict
    ranked = []
    for e in engines:
        if e["status"] != "ok" or not e.get("prim"):
            ranked.append({**e, "rank_top1_strict": None})
            continue
        ranked.append(e)
    avail = [e for e in ranked if e["status"] == "ok" and e.get("prim")]
    avail_sorted = sorted(
        avail, key=lambda e: e["prim"]["top1_strict"], reverse=True
    )
    rank_map = {e["name"]: i + 1 for i, e in enumerate(avail_sorted)}
    for e in ranked:
        e["rank_top1_strict"] = rank_map.get(e["name"])

    ours = next((e for e in ranked if e["name"].startswith("Ours")), None)
    best = avail_sorted[0] if avail_sorted else None
    leaderboard = {
        "protocol": protocol or e2e.get("protocol"),
        "source_e2e": str(e2e_path),
        "n": (ours or {}).get("n")
        or (e2e.get("protocol") or {}).get("n_samples"),
        "metric": "primitive L4-strict (find_mapping 0.05/3° ∧ |det−1|<0.25)",
        "engines": ranked,
        "verdict": {
            "ours_is_top1": bool(
                ours
                and best
                and ours["name"] == best["name"]
                and ours.get("status") == "ok"
            ),
            "ours_top1_strict": (ours.get("prim") or {}).get("top1_strict")
            if ours
            else None,
            "best_name": (best or {}).get("name"),
            "best_top1_strict": (best.get("prim") or {}).get("top1_strict")
            if best
            else None,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(leaderboard, indent=2))

    print("==== CNRS L4-strict leaderboard ====", flush=True)
    for e in ranked:
        if e["status"] != "ok" or not e.get("prim"):
            print(f"  {e['name']:28s}  UNAVAILABLE  ({e.get('reason','')})", flush=True)
            continue
        p = e["prim"]
        extra = ""
        if not e.get("top1_only"):
            extra = f"  Top-20={p['top20_strict']:.1%}  lib={p['lib_strict']:.1%}"
        print(
            f"  #{e['rank_top1_strict']} {e['name']:28s}  "
            f"Top-1={p['top1_strict']:.1%}{extra}",
            flush=True,
        )
    v = leaderboard["verdict"]
    print(
        f"verdict: ours_is_top1={v['ours_is_top1']} "
        f"(ours={v['ours_top1_strict']}, best={v['best_name']} @ {v['best_top1_strict']})",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
