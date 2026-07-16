#!/usr/bin/env bash
# Sequential Phase 0-3 training + eval pipeline (steps 3-5).
set -euo pipefail

ROOT="/nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706"
cd "$ROOT"

OUT_DIR="results/phase03_experiments"
LOG="$OUT_DIR/pipeline.log"
mkdir -p "$OUT_DIR"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

eval_loss() {
  local name="$1"
  local cfg="$2"
  local exp_name
  exp_name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
  local ckpt="results/experiments/${exp_name}/checkpoints/best.pt"
  log "Evaluating loss_${name} from ${ckpt}"
  python scripts/eval_valid.py \
    --config configs/scale_100k_no_cs_matrix6.yaml \
    --checkpoint "$ckpt" \
    --output-path "$OUT_DIR/valid1400_loss_${name}.json" \
    2>&1 | tee -a "$LOG"
}

train_loss() {
  local name="$1"
  local cfg="$2"
  local exp_name
  exp_name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment_name'])")
  local ckpt="results/experiments/${exp_name}/checkpoints/best.pt"
  if [[ -f "$ckpt" ]]; then
    log "Skip training ${name}: checkpoint exists (${ckpt})"
    eval_loss "$name" "$cfg"
    return
  fi
  log "=== Training loss: ${name} (${cfg}) ==="
  python scripts/train.py --config "$cfg" 2>&1 | tee -a "$LOG"
  eval_loss "$name" "$cfg"
}

LOSS_PAIRS=(
  "length_angle:configs/scale_100k_loss_length_angle.yaml"
  "cs_mask:configs/scale_100k_loss_cs_mask.yaml"
  "cs_reweight:configs/scale_100k_loss_cs_reweight.yaml"
  "combined:configs/scale_100k_loss_combined.yaml"
)

log "Starting loss ablation trainings (4 x 20 epochs)"
for pair in "${LOSS_PAIRS[@]}"; do
  name="${pair%%:*}"
  cfg="${pair##*:}"
  train_loss "$name" "$cfg"
done

log "Selecting best loss mode by raw_top1"
best_name=""
best_raw=0
for pair in "${LOSS_PAIRS[@]}"; do
  name="${pair%%:*}"
  json="$OUT_DIR/valid1400_loss_${name}.json"
  raw=$(python -c "import json; m=json.load(open('$json'))['metrics']['raw_top1_lattice_match_rate']; print(m)")
  log "  ${name}: raw_top1=${raw}"
  if python -c "import sys; sys.exit(0 if float('$raw') > float('$best_raw') else 1)"; then
    best_raw="$raw"
    best_name="$name"
  fi
done
log "Best loss mode: ${best_name} (raw_top1=${best_raw})"

best_cfg=""
for pair in "${LOSS_PAIRS[@]}"; do
  name="${pair%%:*}"
  if [[ "$name" == "$best_name" ]]; then
    best_cfg="${pair##*:}"
    break
  fi
done

log "=== Hyperparam sweep on ${best_name} ==="
python scripts/sweep_train_hyperparams.py \
  --base-config "$best_cfg" \
  --loss-modes "$best_name" \
  --head-lrs 0.0005 0.001 0.002 \
  --batch-sizes 64 128 \
  --output-path "$OUT_DIR/sweep_${best_name}.json" \
  2>&1 | tee -a "$LOG"

log "=== Test1400 eval for best loss checkpoint ==="
best_exp=$(python -c "import yaml; print(yaml.safe_load(open('$best_cfg'))['experiment_name'])")
python scripts/eval_valid.py \
  --config configs/scale_100k_no_cs_matrix6_testset.yaml \
  --checkpoint "results/experiments/${best_exp}/checkpoints/best.pt" \
  --output-path "$OUT_DIR/test1400_best_${best_name}.json" \
  2>&1 | tee -a "$LOG"

log "Pipeline complete. Summary in ${OUT_DIR}/"
