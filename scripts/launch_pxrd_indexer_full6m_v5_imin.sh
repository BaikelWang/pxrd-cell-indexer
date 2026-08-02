#!/usr/bin/env bash
# PXRD-indexer full-6M v5: v4-wide schedule at 60 epochs + train-time peak-threshold
# randomization. Single new variable vs v4 so the gain stays attributable.
#
# Why 60 epochs
# -------------
# v4 (34ep) never flattened: valid loss still fell 0.2399 (ep27) -> 0.2167 (ep34)
# and MP100 rose 71% -> 72% -> 74% on its last three evals. But cosine had already
# annealed to ~0 by ep31, so the tail gain was anneal-driven, not headroom. 88M
# params had seen only ~200M samples. Expect a moderate lift, not a step change.
#
# Why peak-threshold randomization
# --------------------------------
# Training saw a fixed I>5 while the CNRS protocol unions I>=5 and I>=1, which is
# worth +10.6pp lib and took orthorhombic raw recall 12.9% -> 45.2%. Sampling the
# threshold per training sample removes that train/inference mismatch.
#
# NOTE: this only became possible after fixing augment_spectrum, which hardcoded
# a second `pre_filter_intensity > 5` mask that silently reverted every sub-5
# threshold back to I>5.
#
# Config vs v4 (only epochs and peak_imin_choices differ)
# ------------------------------------------------------
#   peaktf-scale=wide   d512 / L8 / ffn2048 / out1024   (same)
#   flow                8 x 1024                        (same)
#   batch               512/GPU -> global 2048          (same)
#   lr                  2e-3, warmup 1ep                (same)
#   amp                 bf16                            (same)
#   select              valid_macro @ 0.2%, n=700       (same, valid still I>5)
#   epochs              34 -> 60
#   peak_imin_choices   fixed I>5 -> {5, 2, 1}
#
# Volcano / 4-GPU entry:
#   cd /nanolab/users/wyx/Task/PRXD-Cell-indexing-model-0706
#   n_gpu=4 MASTER_PORT=16523 bash scripts/launch_pxrd_indexer_full6m_v5_imin.sh
#
# Ablation arm (60ep only, no threshold change) to separate the two effects:
#   peak_imin_choices= out_dir=results/flow_seedgen/pxrd_indexer_full6m_v5_60ep_only \
#     log=logs/pxrd_indexer_full6m_v5_60ep_only.log \
#     bash scripts/launch_pxrd_indexer_full6m_v5_imin.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MASTER_PORT=${MASTER_PORT:-16523}
n_gpu="${n_gpu:-4}" \
  target_global_batch="${target_global_batch:-2048}" \
  batch_size="${batch_size:-512}" \
  lr="${lr:-2e-3}" \
  amp="${amp:-bf16}" \
  peaktf_scale=wide \
  flow_layers=8 \
  flow_hidden=1024 \
  epochs="${epochs:-60}" \
  peak_imin_choices="${peak_imin_choices-5,2,1}" \
  select_metric=valid_macro \
  select_tol=0.002 \
  valid_eval_n=700 \
  eval_every=1 \
  eval_k=100 \
  sample_steps=50 \
  mp100_every=3 \
  out_dir="${out_dir:-results/flow_seedgen/pxrd_indexer_full6m_v5_imin}" \
  log="${log:-logs/pxrd_indexer_full6m_v5_imin.log}" \
  bash scripts/train_pxrd_indexer_full6m.sh
