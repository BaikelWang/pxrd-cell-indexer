#!/usr/bin/env bash
# R6-A ladder: P0-700 (baseline + peak w015) → 10k → 100k (if gates pass).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r6a

run_one() {
  local cfg="$1"
  local name
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
  echo "=== TRAIN $name ==="
  python scripts/train.py --config "$cfg" \
    2>&1 | tee "results/beat_engine/raw_diag/r6a/${name}.log"
  echo "=== DONE $name ==="
}

# P0 gate: train elem ≥ 80% and peak not worse than baseline.
run_one configs/overfit700_r6a_baseline_clf.yaml
run_one configs/overfit700_r6a_peak_w015.yaml

python - <<'PY'
import json
from pathlib import Path
root = Path("results/experiments")
rows = []
for name in [
    "overfit700_r6a_baseline_clf_seed42",
    "overfit700_r6a_peak_w015_seed42",
]:
    summary = root / name / "summary.json"
    if not summary.exists():
        print(f"MISSING {summary}")
        continue
    data = json.loads(summary.read_text())
    best_elem, best_ep = -1.0, None
    for row in data.get("history", []):
        m = row.get("valid") or {}
        elem = m.get("strict_raw_top1_elementwise_rate")
        if elem is not None and elem > best_elem:
            best_elem, best_ep = float(elem), row.get("epoch")
    print(f"{name}: best_elem={best_elem*100:.2f}% epoch={best_ep}")
    rows.append((name, best_elem))
if len(rows) == 2:
    base, peak = rows[0][1], rows[1][1]
    ok = peak >= 0.80 and peak + 1e-9 >= base - 0.02
    print(f"P0 GATE: peak={peak*100:.2f}% baseline={base*100:.2f}% -> {'PASS' if ok else 'FAIL'}")
PY
