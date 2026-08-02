#!/usr/bin/env bash
# Launch A3-G1 6M rerun: RealPXRD train augment + larger batch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

CONFIG=configs/scale_6m_a3_g1_gstar6_aug.yaml
LOG=logs/scale_6m_a3_g1_gstar6_aug_bs512_seed42.train.log
mkdir -p logs

JSONL=data/processed/train_full_niggli_seed42.jsonl
STATS=data/processed/lattice_gstar6_stats_full_niggli_seed42.json
if [[ ! -f "$JSONL" || ! -f "$STATS" ]]; then
  echo "missing jsonl/stats; refuse to start" >&2
  exit 1
fi

echo "[$(date -Is)] starting train: $CONFIG"
python scripts/train.py --config "$CONFIG" 2>&1 | tee "$LOG"
echo "[$(date -Is)] train finished"
