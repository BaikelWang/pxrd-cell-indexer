#!/usr/bin/env bash
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
echo "[$(date -Is)] shared"
python scripts/train.py --config configs/scale_10k_r2_shared.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/10k_shared.log
echo "[$(date -Is)] oracle_cs"
python scripts/train.py --config configs/scale_10k_r2_oracle_cs.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/10k_oracle_cs.log
echo "[$(date -Is)] angle_prior"
python scripts/train.py --config configs/scale_10k_r2_angle_prior.yaml 2>&1 | tee results/beat_engine/raw_diag/r2/10k_angle_prior.log
echo "[$(date -Is)] 10k chain done"
