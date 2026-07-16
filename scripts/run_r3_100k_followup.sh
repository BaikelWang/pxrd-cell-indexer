#!/usr/bin/env bash
# After the main R3 100k chain finishes: diagnose + setting-classifier runs.
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r3

wait_for_split_joint() {
  echo "[$(date -Is)] waiting for split_joint / chain..."
  while pgrep -f 'configs/scale_100k_r3_split_joint.yaml' >/dev/null 2>&1 \
     || pgrep -f 'scripts/run_r3_100k_chain.sh' >/dev/null 2>&1; do
    # still training if train.py child of chain exists
    if pgrep -f 'python scripts/train.py --config configs/scale_100k_r3_split_joint' >/dev/null 2>&1; then
      sleep 90
      continue
    fi
    # chain script may linger briefly after last train
    if pgrep -f 'scripts/run_r3_100k_chain.sh' >/dev/null 2>&1; then
      sleep 30
      continue
    fi
    break
  done
  echo "[$(date -Is)] chain idle"
}

wait_for_split_joint

echo "[$(date -Is)] summarize + diagnose finished R3 100k runs"
python scripts/summarize_r3_runs.py \
  --runs \
    scale_100k_r3_cs_pred_seed42 \
    scale_100k_r3_cubic_split_seed42 \
    scale_100k_r3_cubic_split_oracle_seed42 \
    scale_100k_r3_joint_phys_seed42 \
    scale_100k_r3_split_joint_seed42 \
  --output results/beat_engine/raw_diag/r3/g2_partial_summary.json \
  2>&1 | tee results/beat_engine/raw_diag/r3/followup_summarize.log

echo "[$(date -Is)] start cubic_split_clf from scratch"
python scripts/train.py --config configs/scale_100k_r3_cubic_split_clf.yaml \
  2>&1 | tee results/beat_engine/raw_diag/r3/scale_100k_r3_cubic_split_clf.log

echo "[$(date -Is)] start cubic_split_clf warm-start from oracle"
python scripts/train.py --config configs/scale_100k_r3_cubic_split_clf_ws.yaml \
  2>&1 | tee results/beat_engine/raw_diag/r3/scale_100k_r3_cubic_split_clf_ws.log

echo "[$(date -Is)] summarize classifier runs"
python scripts/summarize_r3_runs.py \
  --runs \
    scale_100k_r3_cs_pred_seed42 \
    scale_100k_r3_cubic_split_seed42 \
    scale_100k_r3_cubic_split_oracle_seed42 \
    scale_100k_r3_joint_phys_seed42 \
    scale_100k_r3_split_joint_seed42 \
    scale_100k_r3_cubic_split_clf_seed42 \
    scale_100k_r3_cubic_split_clf_ws_seed42 \
  --output results/beat_engine/raw_diag/r3/g2_summary.json \
  2>&1 | tee -a results/beat_engine/raw_diag/r3/followup_summarize.log

echo "[$(date -Is)] r3 followup done"
