#!/usr/bin/env python3
"""Merge MP100 multi-engine results into a primitive L4-strict leaderboard.

Sources
-------
* Ours:            eval_reseed_arms JSON (prim topk/lib) OR e2e l4_prim_compare
* JADE9:           results/l4_prim_vs_conv.json (jade_per_sample) + system labels
* Native McMaille: results/l4_prim_vs_conv.json (per_sample.mcm_prim)
* DICVOL/TREOR/ITO: results/mp100_benchmark/<engine>/summary.json

Writes ``results/mp100_benchmark/l4_prim_leaderboard.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIF_DIR = ROOT / "data" / "MP-100samples-benchmark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ours",
        default="results/mp100_benchmark/ours_v4_reseed.json",
        help="eval_reseed_arms output (single arm or multi)",
    )
    ap.add_argument("--ours-arm", default="", help="arm key inside --ours; empty = sole/first")
    ap.add_argument("--ours-name", default="Ours (v4 wide K=100)")
    ap.add_argument(
        "--l4-cache",
        default="results/l4_prim_vs_conv.json",
        help="JADE9 + native Mc prim scores",
    )
    ap.add_argument("--bench-dir", default="results/mp100_benchmark")
    ap.add_argument("--out", default="results/mp100_benchmark/l4_prim_leaderboard.json")
    return ap.parse_args()


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def classic_label(key: str, summary: dict) -> str:
    binary = str(summary.get("binary") or summary.get("binary_how") or "").lower()
    if key == "dicvol06":
        return "DICVOL91" if "91" in binary else "DICVOL06"
    if key == "ito":
        return "ITO13" if "13" in binary else "ITO"
    if key == "treor90":
        return "TREOR90"
    return key


def by_system_from_bools(rows: list[dict], key: str) -> dict:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[r["system"]].append(bool(r[key]))
    out = {}
    for sys, vals in sorted(buckets.items()):
        n = len(vals)
        out[sys] = {
            "n": n,
            "top1_strict": sum(vals) / n if n else 0.0,
            "top20_strict": sum(vals) / n if n else 0.0,
            "lib_strict": sum(vals) / n if n else 0.0,
        }
    return out


def load_system_map(l4: dict) -> dict[str, str]:
    return {r["sample_id"]: r["system"] for r in l4["per_sample"]}


def jade_block(l4: dict, systems: dict[str, str]) -> dict:
    rows = []
    for r in l4["jade_per_sample"]:
        sid = r["sample_id"]
        hit = bool((r.get("jade_prim") or {}).get("top1_strict"))
        rows.append({"sample_id": sid, "system": systems.get(sid, "?"), "hit": hit})
    n = len(rows)
    top1 = sum(r["hit"] for r in rows) / n if n else 0.0
    by_sys = by_system_from_bools(
        [{"system": r["system"], "top1_strict": r["hit"]} for r in rows],
        "top1_strict",
    )
    # Top-1 only engine — mirror rates into top20/lib for table uniformity
    for sys in by_sys:
        by_sys[sys]["top20_strict"] = by_sys[sys]["top1_strict"]
        by_sys[sys]["lib_strict"] = by_sys[sys]["top1_strict"]
    return {
        "name": "JADE9",
        "status": "ok",
        "n": n,
        "prim": {
            "top1_loose": sum(
                1 for r in l4["jade_per_sample"] if (r.get("jade_prim") or {}).get("top1_loose")
            )
            / n,
            "top1_strict": top1,
            "top20_loose": top1,
            "top20_strict": top1,
            "lib_loose": top1,
            "lib_strict": top1,
        },
        "by_system": by_sys,
        "top1_only": True,
        "source": "results/l4_prim_vs_conv.json ← jade-index .hkl",
    }


def mcm_block(l4: dict) -> dict:
    rows = l4["per_sample"]
    n = len(rows)

    def rate(field: str) -> float:
        return sum(1 for r in rows if (r.get("mcm_prim") or {}).get(field)) / n

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["system"]].append(r.get("mcm_prim") or {})
    by_sys = {}
    for sys, rs in sorted(buckets.items()):
        m = len(rs)
        by_sys[sys] = {
            "n": m,
            "top1_strict": sum(1 for x in rs if x.get("top1_strict")) / m,
            "top20_strict": sum(1 for x in rs if x.get("top20_strict")) / m,
            "lib_strict": sum(1 for x in rs if x.get("top20_strict")) / m,
        }
    return {
        "name": "Native McMaille",
        "status": "ok",
        "n": n,
        "prim": {
            "top1_loose": rate("top1_loose"),
            "top1_strict": rate("top1_strict"),
            "top20_loose": rate("top20_loose"),
            "top20_strict": rate("top20_strict"),
            "lib_loose": rate("top20_loose"),
            "lib_strict": rate("top20_strict"),
        },
        "by_system": by_sys,
        "top1_only": False,
        "source": "results/l4_prim_vs_conv.json ← mp100_compare/original",
    }


def ours_from_reseed(path: Path, arm: str, name: str) -> dict:
    data = json.loads(path.read_text())
    if arm and arm in data:
        block = data[arm]
    elif len(data) == 1:
        block = next(iter(data.values()))
    elif "prim" in data:
        block = data
    else:
        # prefer key containing Ours / v4 / first
        keys = list(data.keys())
        pick = next((k for k in keys if "v4" in k.lower() or "ours" in k.lower()), keys[0])
        block = data[pick]
    prim = block["prim"]
    topk = prim["topk_strict"]
    topk_lo = prim["topk_loose"]

    def _k(d: dict, k: int) -> float:
        return float(d.get(k, d.get(str(k), 0.0)))

    # by_system if present in per_sample
    by_sys: dict = {}
    if "per_sample" in block:
        buckets: dict[str, list] = defaultdict(list)
        systems = {}
        l4_path = ROOT / "results/l4_prim_vs_conv.json"
        if l4_path.exists():
            systems = {
                r["sample_id"]: r["system"]
                for r in json.loads(l4_path.read_text())["per_sample"]
            }
        for r in block["per_sample"]:
            sys = systems.get(r["sample_id"], "?")
            buckets[sys].append(r["prim"])
        for sys, rs in sorted(buckets.items()):
            m = len(rs)
            by_sys[sys] = {
                "n": m,
                "top1_strict": sum(1 for x in rs if x["topk_strict"].get(1) or x["topk_strict"].get("1")) / m,
                "top20_strict": sum(1 for x in rs if x["topk_strict"].get(20) or x["topk_strict"].get("20")) / m,
                "lib_strict": sum(1 for x in rs if x["lib_strict"]) / m,
            }
    return {
        "name": name,
        "status": "ok",
        "n": block["n_samples"],
        "prim": {
            "top1_loose": _k(topk_lo, 1),
            "top1_strict": _k(topk, 1),
            "top20_loose": _k(topk_lo, 20),
            "top20_strict": _k(topk, 20),
            "lib_loose": prim["lib_loose"],
            "lib_strict": prim["lib_strict"],
        },
        "by_system": by_sys,
        "top1_only": False,
        "run_dir": block.get("run_dir"),
        "source": str(path),
    }


def eng_classic(name: str, summary: dict) -> dict:
    st = summary.get("status", "ok")
    if st != "ok" or not summary.get("prim"):
        return {
            "name": name,
            "status": "unavailable",
            "reason": summary.get("reason", "unavailable"),
            "n": summary.get("n"),
            "prim": None,
            "by_system": {},
            "top1_only": False,
        }
    return {
        "name": name,
        "status": "ok",
        "n": summary.get("n"),
        "prim": summary["prim"],
        "by_system": summary.get("by_system", {}),
        "top1_only": False,
        "binary": summary.get("binary"),
        "binary_how": summary.get("binary_how"),
    }


def main() -> None:
    args = parse_args()
    bench = _abs(args.bench_dir)
    out_path = _abs(args.out)
    l4 = json.loads(_abs(args.l4_cache).read_text())
    systems = load_system_map(l4)

    engines = []
    ours_path = _abs(args.ours)
    if ours_path.exists():
        engines.append(ours_from_reseed(ours_path, args.ours_arm, args.ours_name))
    else:
        engines.append(
            {
                "name": args.ours_name,
                "status": "unavailable",
                "reason": f"missing {ours_path}",
                "n": None,
                "prim": None,
                "by_system": {},
                "top1_only": False,
            }
        )

    engines.append(jade_block(l4, systems))
    engines.append(mcm_block(l4))

    for key in ("dicvol06", "treor90", "ito"):
        sp = bench / key / "summary.json"
        if not sp.exists():
            engines.append(
                eng_classic(key.upper(), {"status": "unavailable", "reason": f"missing {sp}"})
            )
            continue
        s = json.loads(sp.read_text())
        engines.append(eng_classic(classic_label(key, s), s))

    avail = [e for e in engines if e["status"] == "ok" and e.get("prim")]
    avail_sorted = sorted(avail, key=lambda e: e["prim"]["top1_strict"], reverse=True)
    rank_map = {e["name"]: i + 1 for i, e in enumerate(avail_sorted)}
    for e in engines:
        e["rank_top1_strict"] = rank_map.get(e["name"])

    protocol = {}
    proto_path = bench / "protocol.json"
    if proto_path.exists():
        protocol = json.loads(proto_path.read_text())

    ours = next((e for e in engines if e["name"].startswith("Ours")), None)
    best = avail_sorted[0] if avail_sorted else None
    leaderboard = {
        "protocol": protocol,
        "n": 100,
        "metric": "primitive L4-strict (find_mapping 0.05/3° ∧ |det−1|<0.25)",
        "engines": engines,
        "verdict": {
            "ours_is_top1": bool(
                ours and best and ours["name"] == best["name"] and ours.get("status") == "ok"
            ),
            "ours_top1_strict": (ours.get("prim") or {}).get("top1_strict") if ours else None,
            "best_name": (best or {}).get("name"),
            "best_top1_strict": (best.get("prim") or {}).get("top1_strict") if best else None,
            "jade9_top1_strict": next(
                (e["prim"]["top1_strict"] for e in engines if e["name"] == "JADE9"), None
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(leaderboard, indent=2))

    print("==== MP100 L4-strict leaderboard ====", flush=True)
    for e in engines:
        if e["status"] != "ok" or not e.get("prim"):
            print(f"  {e['name']:28s}  UNAVAILABLE  ({e.get('reason','')})", flush=True)
            continue
        p = e["prim"]
        rk = e.get("rank_top1_strict")
        extra = "" if e.get("top1_only") else f"  Top-20={p['top20_strict']:.1%}  lib={p['lib_strict']:.1%}"
        print(
            f"  #{rk} {e['name']:26s}  Top-1={p['top1_strict']:.1%}{extra}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
