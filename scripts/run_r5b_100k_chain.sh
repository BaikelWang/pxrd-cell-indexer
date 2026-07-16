#!/usr/bin/env bash
# R5-B: input-side ablations on champion architecture (hist bins / intensity_min).
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r5

for cfg in \
  configs/scale_100k_r5_hist512.yaml \
  configs/scale_100k_r5_imin0.yaml
do
  name=$(basename "$cfg" .yaml)
  echo "[$(date -Is)] start $name"
  python scripts/train.py --config "$cfg" 2>&1 | tee "results/beat_engine/raw_diag/r5/${name}.log"
done

echo "[$(date -Is)] summarize R5-B"
python scripts/summarize_r3_runs.py \
  --runs \
    scale_100k_r3_cubic_split_clf_seed42 \
    scale_100k_r5_hist512_seed42 \
    scale_100k_r5_imin0_seed42 \
  --output results/beat_engine/raw_diag/r5/r5b_summary.json \
  --diag-subdir r5 \
  2>&1 | tee results/beat_engine/raw_diag/r5/r5b_summarize.log

echo "[$(date -Is)] r5b chain done"
