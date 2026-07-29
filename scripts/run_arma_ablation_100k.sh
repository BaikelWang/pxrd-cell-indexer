#!/usr/bin/env bash
# Arm A 100k ablations, run sequentially on one GPU.
#
# Baselines already measured (same protocol):
#   peaktf pilot100k_equiv_off  best library@100 = 51%
#   armA   pilot100k_equiv_off  best library@100 = 43%  (int pos, both frozen)
#
# Arms:
#   A  unfreeze both (Bert + CSPNet)      -- launched separately, already running
#   B  unfreeze encoder only              -- DROPPED: the Bert encoder is only 36,928
#                                            params and sits downstream of the 1-degree
#                                            2-theta quantization, so it cannot move.
#   C  unfreeze CSPNet only
#   D  drop CSPNet entirely (Bert -> flow)
#   E  continuous Fourier positions (removes the 1-degree input bottleneck)
#   F  D + encoder widened/deepened to d256/L6/ffn1024 (4.98M), int positions.
#      NOTE: resizing makes every pretrained tensor shape-incompatible, so F's
#      encoder trains FROM SCRATCH. F is a scratch control that reuses the
#      architecture -- it is NOT transfer learning. Do not report it as Arm A.
#
# D/E/F form a 2x2 over (encoder capacity) x (position encoding):
#            int pos (1 deg bins)   Fourier pos
#   37k      D                      E
#   ~5M      F                      ~ peaktf (51%)
#
# Order is C -> D -> F -> E so that E's CSPNet setting can be decided once both
# D and F are known (E currently keeps CSPNet, matching the frozen 43% baseline).
#
# This script only runs C, D, F, E. A is left alone; the table reads its best_meta.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/flow_seedgen

COMMON=(
  --backbone armA
  --train-jsonl data/processed/train100k_niggli_seed42.jsonl
  --valid-jsonl data/processed/valid1400_niggli_seed42.jsonl
  --stats data/processed/lattice_gstar6_stats_100k_niggli_seed42.json
  --train-lmdb /nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb
  --valid-lmdb /nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb
  --equiv-target off --epochs 60 --batch-size 256 --lr 1e-3 --weight-decay 0.05
  --warmup-epochs 2.0 --flow-layers 6 --flow-hidden 512 --sample-steps 50
  --eval-k 100 --eval-every 3 --augment on --num-workers 6 --seed 42
)

run_one() {
  local tag="$1" out="$2" log="$3"
  shift 3
  if [[ -f "${out}/best_meta.json" ]]; then
    echo "======== SKIP ${tag} (already have ${out}/best_meta.json) ========"
    return 0
  fi
  echo "======== START ${tag} -> ${out}  [$(date -Is)] ========"
  # Keep going if one arm dies, so the rest of the queue still runs.
  if python3 -u scripts/train_flow_seedgen.py "${COMMON[@]}" "$@" --out-dir "${out}" > "${log}" 2>&1; then
    echo "======== DONE ${tag}  [$(date -Is)] ========"
  else
    echo "======== FAILED ${tag} (exit $?) -- see ${log} ========"
    tail -20 "${log}"
  fi
}

run_one C_unfreeze_cspnet \
  results/flow_seedgen/armA_pilot100k_unfreeze_decoder \
  logs/armA_pilot100k_unfreeze_decoder.log \
  --arma-unfreeze decoder

run_one D_no_cspnet \
  results/flow_seedgen/armA_pilot100k_no_cspnet \
  logs/armA_pilot100k_no_cspnet.log \
  --arma-unfreeze none --arma-no-cspnet

run_one F_scratch_d256L6 \
  results/flow_seedgen/armA_pilot100k_scratch_d256L6 \
  logs/armA_pilot100k_scratch_d256L6.log \
  --arma-unfreeze encoder --arma-no-cspnet --arma-encoder-scale d256L6

run_one E_fourier_pos \
  results/flow_seedgen/armA_pilot100k_fourier_pos \
  logs/armA_pilot100k_fourier_pos.log \
  --arma-unfreeze encoder --arma-pos fourier

bash scripts/summarize_arma_ablation.sh
