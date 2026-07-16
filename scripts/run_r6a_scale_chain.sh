#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r6a

echo "=== 10k peak w005 ==="
python scripts/train.py --config configs/scale_10k_r6a_peak_w005.yaml \
  2>&1 | tee results/beat_engine/raw_diag/r6a/scale_10k_r6a_peak_w005.log

python - <<'PY'
import json
from pathlib import Path
p=Path('results/experiments/scale_10k_r6a_peak_w005_seed42/summary.json')
d=json.loads(p.read_text())
best=max(((r.get('valid') or {}).get('strict_raw_top1_elementwise_rate') or -1, r.get('epoch')) for r in d['history'])
print(f"10k best elem={best[0]*100:.2f}% ep={best[1]}")
# Gate: not clearly below ~5.7% (cs_pred 10k reference)
ok = best[0] >= 0.04
print(f"10k GATE (>=4% smoke): {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PY

echo "=== 100k peak w005 ==="
python scripts/train.py --config configs/scale_100k_r6a_peak_w005.yaml \
  2>&1 | tee results/beat_engine/raw_diag/r6a/scale_100k_r6a_peak_w005.log

python - <<'PY'
import json
from pathlib import Path
p=Path('results/experiments/scale_100k_r6a_peak_w005_seed42/summary.json')
d=json.loads(p.read_text())
best_row=None
for r in d['history']:
  v=r.get('valid') or {}
  e=v.get('strict_raw_top1_elementwise_rate')
  if e is None: continue
  if best_row is None or e>best_row[0]:
    best_row=(e, r.get('epoch'), v.get('angle_mae'), v)
print(f"100k best elem={best_row[0]*100:.2f}% ang={best_row[2]:.2f} ep={best_row[1]}")
champ=0.1543
ok=best_row[0]+1e-9 >= champ and True  # ang/noncub checked in writeup
print(f"100k vs champ 15.43%: {'>=' if best_row[0]>=champ else '<'} -> review")
PY
