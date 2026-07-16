#!/usr/bin/env bash
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r2
echo "[$(date -Is)] shared 100k"
python scripts/train.py --config configs/scale_100k_r2_shared.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/100k_shared.log
echo "[$(date -Is)] oracle_cs 100k"
python scripts/train.py --config configs/scale_100k_r2_oracle_cs.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/100k_oracle_cs.log
echo "[$(date -Is)] cs_pred 100k"
python scripts/train.py --config configs/scale_100k_r2_cs_pred.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/100k_cs_pred.log
echo "[$(date -Is)] g2 100k chain done"
