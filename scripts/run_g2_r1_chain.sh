#!/usr/bin/env bash
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
mkdir -p results/beat_engine/raw_diag/g2
echo "[$(date -Is)] start hist_imin0"
python scripts/train.py --config configs/scale_100k_r1_hist_imin0.yaml 2>&1 | tee results/beat_engine/raw_diag/g2/hist_imin0.log
echo "[$(date -Is)] start hist_imin5"
python scripts/train.py --config configs/scale_100k_r1_hist_imin5.yaml 2>&1 | tee results/beat_engine/raw_diag/g2/hist_imin5.log
echo "[$(date -Is)] start legacy_imin0"
python scripts/train.py --config configs/scale_100k_r1_legacy_imin0.yaml 2>&1 | tee results/beat_engine/raw_diag/g2/legacy_imin0.log
echo "[$(date -Is)] g2 chain done"
