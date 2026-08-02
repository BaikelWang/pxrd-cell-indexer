#!/usr/bin/env bash
# PXRD-indexer full-6M v3: fixed selection + the batch schedule we should have used on v2.
#
# History
# -------
# v2 actually ran: 4×GPU, per-GPU bs=128 → global 512, lr=1e-3, amp=off
#   (VRAM was mostly idle; user confirmed on 2026-07-29).
# Recommended then (and still): global 2048 = 4×512, lr=4e-3 (linear scale), amp=bf16.
#   Script defaults were updated, but v2 was never restarted — there is no gbs2048 run.
# v3 (this): that recommended schedule + select=valid_macro@0.2% n=700.
#   MP100 every 3 epochs for curve analysis only (never drives best.pt).
#
# Usage (need 4 visible GPUs, Arm A stopped or elsewhere):
#   bash scripts/launch_pxrd_indexer_full6m_v3.sh
#
# To force the old under-utilized schedule for an apples-to-apples v2 compare:
#   target_global_batch=512 lr=1e-3 amp=off bash scripts/launch_pxrd_indexer_full6m_v3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

n_gpu=${n_gpu:-4}
n_visible=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "${n_visible}" -lt "${n_gpu}" ]; then
    echo "ERROR: need ${n_gpu} GPUs, nvidia-smi sees ${n_visible}." >&2
    echo "       Free GPUs / set CUDA_VISIBLE_DEVICES, then re-run." >&2
    exit 1
fi

export MASTER_PORT=${MASTER_PORT:-16521}
n_gpu="${n_gpu}" \
  target_global_batch="${target_global_batch:-2048}" \
  lr="${lr:-4e-3}" \
  amp="${amp:-bf16}" \
  batch_size="${batch_size:-512}" \
  epochs=34 \
  select_metric=valid_macro \
  select_tol=0.002 \
  valid_eval_n=700 \
  eval_every=1 \
  eval_k=100 \
  sample_steps=50 \
  mp100_every=3 \
  out_dir="${out_dir:-results/flow_seedgen/pxrd_indexer_full6m_v3_macro}" \
  log="${log:-logs/pxrd_indexer_full6m_v3_macro.log}" \
  bash scripts/train_pxrd_indexer_full6m.sh
