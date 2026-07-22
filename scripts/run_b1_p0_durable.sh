#!/usr/bin/env bash
# Durable B1/B2 driver: pairs of 2 in parallel, auto-resume on SIGKILL.
# Pair1: B1a P0 + B1b P0
# Pair2: B2a + B2b 100k (after both P0 finish; 100k B1 launched separately if gates pass)
set -uo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
LOGDIR=results/beat_engine/a2_a2p5
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/b1_p0_driver.log" 2>&1

gate_ok() {
  python3 - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print("NO_METRICS"); sys.exit(2)
rows=json.loads(p.read_text())
best=max((float(r.get("valid",{}).get("strict_raw_top1_elementwise_rate") or 0), r.get("epoch")) for r in rows)
print(f"best_ep={best[1]} best_strict={best[0]:.6f}")
sys.exit(0 if best[0]>=0.95 else 1)
PY
}

is_train_running() {
  local cfg="$1"
  # Match only the python trainer (not this bash driver, whose cmdline also contains cfg paths).
  pgrep -af "python.*train.py --config ${cfg}" | grep -v pgrep >/dev/null 2>&1
}

run_finished() {
  local run="$1"
  local metrics="results/experiments/${run}/metrics.json"
  if [ -f "results/experiments/${run}/summary.json" ]; then
    return 0
  fi
  if [ -f "$metrics" ]; then
    local n
    n=$(python3 -c "import json; print(len(json.load(open('$metrics'))))")
    if [ "$n" -ge 1200 ]; then
      return 0
    fi
    if gate_ok "$metrics" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

# One durable worker (auto-resume). Runs in foreground for caller that backgrounds it.
run_until_done() {
  local cfg="$1" run="$2" log="$3" max_ep="${4:-1200}"
  local max_tries=30 try=1
  while [ "$try" -le "$max_tries" ]; do
    if run_finished "$run"; then
      echo "[durable] $run finished/gate-ok"
      return 0
    fi
    # Another process already training this config — wait
    if is_train_running "$cfg"; then
      echo "[durable] $run already running; wait ($(date -u))"
      while is_train_running "$cfg"; do sleep 30; done
      try=$((try + 1))
      continue
    fi

    local metrics="results/experiments/${run}/metrics.json"
    local last="results/experiments/${run}/checkpoints/last.pt"
    local n=0
    if [ -f "$metrics" ]; then
      n=$(python3 -c "import json; print(len(json.load(open('$metrics'))))")
    fi
    local resume_args=()
    if [ -f "$last" ] && [ "$n" -gt 0 ]; then
      resume_args=(--resume "$last")
      echo "[durable] try=$try resume $run from ep~$n $(date -u)"
    else
      echo "[durable] try=$try fresh $run $(date -u)"
    fi

    set +e
    python -u scripts/train.py --config "$cfg" "${resume_args[@]}" >>"$log" 2>&1
    local rc=$?
    set -e
    echo "[durable] $run exit_rc=$rc at $(date -u)"

    if run_finished "$run"; then
      return 0
    fi
    sleep 5
    try=$((try + 1))
  done
  echo "[durable] $run FAILED after $max_tries tries"
  return 1
}

run_pair() {
  local label="$1"
  shift
  # remaining args: cfg1 run1 log1 cfg2 run2 log2
  echo "=== pair $label start $(date -u) ==="
  run_until_done "$1" "$2" "$3" &
  local p1=$!
  run_until_done "$4" "$5" "$6" &
  local p2=$!
  wait "$p1" || echo "[pair $label] leg1 failed"
  wait "$p2" || echo "[pair $label] leg2 failed"
  echo "=== pair $label end $(date -u) ==="
}

echo "=== B1/B2 parallel durable driver start $(date -u) ==="

# Pair 1: B1 P0
run_pair b1_p0 \
  configs/overfit700_a2p5_b1a_rel_scalar.yaml \
  overfit700_a2p5_b1a_rel_scalar_seed42 \
  "$LOGDIR/b1a_p0_rel_scalar.log" \
  configs/overfit700_a2p5_b1b_rel_mlp.yaml \
  overfit700_a2p5_b1b_rel_mlp_seed42 \
  "$LOGDIR/b1b_p0_rel_mlp.log"

echo "--- P0 gates ---"
gate_ok results/experiments/overfit700_a2p5_b1a_rel_scalar_seed42/metrics.json \
  | tee /dev/stderr | sed 's/^/B1a /' || echo "B1a gate FAIL"
gate_ok results/experiments/overfit700_a2p5_b1b_rel_mlp_seed42/metrics.json \
  | tee /dev/stderr | sed 's/^/B1b /' || echo "B1b gate FAIL"

# Pair 2: B2 100k recovery (archive stale incomplete active dirs)
if [ -d results/experiments/scale_100k_a2p5_b2b_peaks96_seed42 ] && \
   [ ! -f results/experiments/scale_100k_a2p5_b2b_peaks96_seed42/summary.json ]; then
  n=$(python3 -c "import json; print(len(json.load(open('results/experiments/scale_100k_a2p5_b2b_peaks96_seed42/metrics.json'))))" 2>/dev/null || echo 0)
  if [ "${n:-0}" -le 50 ]; then
    mv results/experiments/scale_100k_a2p5_b2b_peaks96_seed42 \
       "results/experiments/scale_100k_a2p5_b2b_peaks96_seed42_stale_ep${n}" || true
  fi
fi

run_pair b2_100k \
  configs/scale_100k_a2p5_b2a_peaks64.yaml \
  scale_100k_a2p5_b2a_peaks64_seed42 \
  "$LOGDIR/b2a_peaks64.log" \
  configs/scale_100k_a2p5_b2b_peaks96.yaml \
  scale_100k_a2p5_b2b_peaks96_seed42 \
  "$LOGDIR/b2b_peaks96.log"

# Pair 3: B1 100k only if both P0 gates pass
b1a_ok=1; b1b_ok=1
gate_ok results/experiments/overfit700_a2p5_b1a_rel_scalar_seed42/metrics.json || b1a_ok=0
gate_ok results/experiments/overfit700_a2p5_b1b_rel_mlp_seed42/metrics.json || b1b_ok=0
if [ "$b1a_ok" -eq 1 ] && [ "$b1b_ok" -eq 1 ]; then
  echo "=== both P0 gates pass → B1 100k pair $(date -u) ==="
  run_pair b1_100k \
    configs/scale_100k_a2p5_b1a_rel_scalar.yaml \
    scale_100k_a2p5_b1a_rel_scalar_seed42 \
    "$LOGDIR/b1a_100k_rel_scalar.log" \
    configs/scale_100k_a2p5_b1b_rel_mlp.yaml \
    scale_100k_a2p5_b1b_rel_mlp_seed42 \
    "$LOGDIR/b1b_100k_rel_mlp.log"
else
  echo "=== skip B1 100k (P0 gates a=$b1a_ok b=$b1b_ok) $(date -u) ==="
fi

echo "=== durable driver done $(date -u) ==="
echo 'AGENT_LOOP_TICK_b1p0 {"prompt":"B1/B2 parallel durable driver finished. Report P0 gates, B2 finals vs A2-ctrl 40.57%, and B1 100k if launched. Update B1/B2 experiment logs."}'
