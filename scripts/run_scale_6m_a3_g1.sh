#!/usr/bin/env bash
# Chain: wait for full niggli jsonl → gstar6 stats → A3-G1 6M train.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

JSONL=data/processed/train_full_niggli_seed42.jsonl
META=data/processed/train_full_niggli_seed42.meta.json
STATS=data/processed/lattice_gstar6_stats_full_niggli_seed42.json
CONFIG=configs/scale_6m_a3_g1_gstar6.yaml
LOG=logs/scale_6m_a3_g1_gstar6_seed42.train.log
mkdir -p logs

echo "[$(date -Is)] waiting for ${JSONL} (+ meta) ..."
while [[ ! -f "$JSONL" || ! -f "$META" ]]; do
  sleep 30
done
# Ensure exporter finished (partial replaced atomically; meta written last).
# Also wait until no export process is writing.
while pgrep -f 'export_train_full_niggli.py' >/dev/null 2>&1; do
  echo "[$(date -Is)] export still running; n_lines=$(wc -l < "$JSONL" 2>/dev/null || echo 0)"
  sleep 60
done
echo "[$(date -Is)] export done: $(wc -l < "$JSONL") lines"
cat "$META"

if [[ ! -f "$STATS" ]]; then
  echo "[$(date -Is)] computing gstar6 stats (sample 500k) ..."
  python scripts/compute_gstar6_stats.py \
    --input-jsonl "$JSONL" \
    --output-path "$STATS" \
    --max-records 500000 \
    --seed 42
fi

echo "[$(date -Is)] starting train: $CONFIG"
python scripts/train.py --config "$CONFIG" 2>&1 | tee "$LOG"
echo "[$(date -Is)] train finished"
