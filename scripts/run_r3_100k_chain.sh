#!/usr/bin/env bash
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r3
for cfg in \
  configs/scale_100k_r3_cs_pred.yaml \
  configs/scale_100k_r3_cubic_split.yaml \
  configs/scale_100k_r3_cubic_split_oracle.yaml \
  configs/scale_100k_r3_joint_phys.yaml \
  configs/scale_100k_r3_split_joint.yaml
do
  name=$(basename "$cfg" .yaml)
  echo "[$(date -Is)] start $name"
  python scripts/train.py --config "$cfg" 2>&1 | tee "results/beat_engine/raw_diag/r3/${name}.log"
done
echo "[$(date -Is)] r3 100k chain done"
