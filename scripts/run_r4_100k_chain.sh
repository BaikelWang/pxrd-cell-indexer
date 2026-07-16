#!/usr/bin/env bash
# R4: manifold-consistency loss ablation + hard-CS finetune round 2 on top of
# the R3 champion (cubic_split_clf, 100k strict elem 15.43%).
set -euo pipefail
cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
export PYTHONPATH=src
mkdir -p results/beat_engine/raw_diag/r4

for cfg in \
  configs/scale_100k_r4_manifold_w01.yaml \
  configs/scale_100k_r4_manifold_w025.yaml \
  configs/scale_100k_r4_hardcs_ft2.yaml
do
  name=$(basename "$cfg" .yaml)
  echo "[$(date -Is)] start $name"
  python scripts/train.py --config "$cfg" 2>&1 | tee "results/beat_engine/raw_diag/r4/${name}.log"
done
echo "[$(date -Is)] r4 100k chain done"

echo "[$(date -Is)] summarize + diagnose R4 runs"
python scripts/summarize_r3_runs.py \
  --runs \
    scale_100k_r3_cubic_split_clf_seed42 \
    scale_100k_r4_manifold_w01_seed42 \
    scale_100k_r4_manifold_w025_seed42 \
    scale_100k_r4_hardcs_ft2_seed42 \
  --output results/beat_engine/raw_diag/r4/g2_summary.json \
  --diag-subdir r4 \
  2>&1 | tee results/beat_engine/raw_diag/r4/followup_summarize.log

echo "[$(date -Is)] r4 followup done"
