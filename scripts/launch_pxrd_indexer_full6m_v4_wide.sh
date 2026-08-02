#!/usr/bin/env bash
# PXRD-indexer full-6M v4 (plan B): wide peaktf — stable schedule after lr=8e-3 diverged.
#
# History
# -------
# v3 (base, gbs2048, lr=4e-3, bf16): died on first eval — lu_factor_cublas bf16.
#   Fixed in geometry.py (_linalg_float32) + eval decode .float().
# v4 first attempt (wide, gbs4096, lr=8e-3): ran but diverged ep1→ep3
#   train 0.86→1.24, valid_macro@0.2% 3%→0%, mp100@100=2%. Linear LR scale
#   is too aggressive for ~6× larger model. Do NOT resume that out_dir.
#
# Current defaults (stable restart)
# --------------------------------
#   peaktf-scale=wide   d512 / L8 / ffn2048 / out1024
#   flow                8 × 1024
#   batch               512 / GPU → global 2048 (accum 1)
#   lr                  2e-3  (~ sqrt-scale vs gbs512/1e-3; safer than 8e-3)
#   amp                 bf16
#   select              valid_macro @ 0.2%, n=700 stratified
#   mp100               every 3 epochs, report only
#   out_dir             results/flow_seedgen/pxrd_indexer_full6m_v4_wide_lr2e3
#
# Volcano / 4-GPU entry (same style as v2):
#   cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
#   n_gpu=4 MASTER_PORT=16522 \
#     target_global_batch=2048 batch_size=512 lr=2e-3 amp=bf16 \
#     peaktf_scale=wide flow_layers=8 flow_hidden=1024 \
#     select_metric=valid_macro select_tol=0.002 valid_eval_n=700 \
#     mp100_every=3 \
#     out_dir=results/flow_seedgen/pxrd_indexer_full6m_v4_wide_lr2e3 \
#     log=logs/pxrd_indexer_full6m_v4_wide_lr2e3.log \
#     bash scripts/train_pxrd_indexer_full6m.sh
#
# Convenience wrapper (equivalent):
#   n_gpu=4 MASTER_PORT=16522 bash scripts/launch_pxrd_indexer_full6m_v4_wide.sh
#
# Even more conservative (align lr with v2, only enlarge model):
#   lr=1e-3 out_dir=.../pxrd_indexer_full6m_v4_wide_lr1e3 \
#     bash scripts/launch_pxrd_indexer_full6m_v4_wide.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MASTER_PORT=${MASTER_PORT:-16522}
n_gpu="${n_gpu:-4}" \
  target_global_batch="${target_global_batch:-2048}" \
  batch_size="${batch_size:-512}" \
  lr="${lr:-2e-3}" \
  amp="${amp:-bf16}" \
  peaktf_scale=wide \
  flow_layers=8 \
  flow_hidden=1024 \
  epochs=34 \
  select_metric=valid_macro \
  select_tol=0.002 \
  valid_eval_n=700 \
  eval_every=1 \
  eval_k=100 \
  sample_steps=50 \
  mp100_every=3 \
  out_dir="${out_dir:-results/flow_seedgen/pxrd_indexer_full6m_v4_wide_lr2e3}" \
  log="${log:-logs/pxrd_indexer_full6m_v4_wide_lr2e3.log}" \
  bash scripts/train_pxrd_indexer_full6m.sh
