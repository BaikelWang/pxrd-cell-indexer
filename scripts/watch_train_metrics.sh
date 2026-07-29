#!/usr/bin/env bash
# Live-tail training metrics (updates once per epoch when metrics.csv grows).
#
#   bash scripts/watch_train_metrics.sh
#   RUN=results/flow_seedgen/pxrd_indexer_full6m_v2_gbs2048_bf16 INTERVAL=15 \
#     bash scripts/watch_train_metrics.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-pick the newest pxrd_indexer_full6m_v2* dir unless RUN is set.
if [ -z "${RUN:-}" ]; then
    RUN=$(ls -td results/flow_seedgen/pxrd_indexer_full6m_v2* 2>/dev/null | head -1 || true)
fi
RUN=${RUN:-results/flow_seedgen/pxrd_indexer_full6m_v2}
METRICS=${RUN}/metrics.csv
LOG=logs/$(basename "${RUN}").log
INTERVAL=${INTERVAL:-20}

echo "watching ${METRICS}  (every ${INTERVAL}s, Ctrl-C to stop)"
chmod +x "$0" 2>/dev/null || true

prev_lines=0
while true; do
    clear 2>/dev/null || printf '\033[2J\033[H'
    date -u '+%Y-%m-%d %H:%M:%S UTC'
    echo "run: ${RUN}"
    echo
    if [ -f "${METRICS}" ]; then
        lines=$(wc -l < "${METRICS}")
        echo "=== metrics.csv (${lines} lines) ==="
        python3 - "${METRICS}" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    print("(header only — waiting for ep001)")
    raise SystemExit
print(f"{'ep':>4} {'train':>8} {'valid':>8} {'<0.2%':>7} {'<1%':>7} {'<5%':>7} {'select':>7} {'elapsed':>8}")
scores = [float(r["select_score"]) for r in rows if r.get("select_score") not in ("", None)]
best = max(scores) if scores else -1.0
for r in rows[-15:]:
    sel = r.get("select_score") or ""
    mark = " *" if sel and float(sel) == best and best >= 0 else ""

    def pct(k):
        v = r.get(k) or ""
        return f"{100 * float(v):.0f}%" if v not in ("", "None") else "-"

    def fnum(k):
        v = r.get(k) or ""
        return f"{float(v):8.4f}" if v not in ("", "None") else f"{'-':>8}"

    el = r.get("elapsed_s") or ""
    el_s = f"{float(el) / 60:.1f}m" if el else "-"
    print(
        f"{int(float(r['epoch'])):4d} {fnum('train_loss')} {fnum('valid_loss')} "
        f"{pct('valid_hit_0.2pct'):>7} {pct('valid_hit_1pct'):>7} {pct('valid_hit_5pct'):>7} "
        f"{pct('select_score'):>7}{mark} {el_s:>8}"
    )
print()
last = rows[-1]
if last.get("select_score"):
    print(f"latest select (<1%) = {100 * float(last['select_score']):.1f}% @ ep{int(float(last['epoch']))}")
if best >= 0:
    be = next(r for r in rows if r.get("select_score") and float(r["select_score"]) == best)
    print(f"best select so far  = {100 * best:.1f}% @ ep{int(float(be['epoch']))}  (*)")
PY
        [ "${lines}" -gt "${prev_lines}" ] && echo "(file grew)" && prev_lines=${lines}
    else
        echo "waiting for ${METRICS} ..."
    fi
    echo
    echo "=== log tail ==="
    if [ -f "${LOG}" ]; then
        rg -v 'fused_|not installed|expandable_segments' "${LOG}" | tail -8
    else
        echo "(no log yet at ${LOG})"
    fi
    sleep "${INTERVAL}"
done
