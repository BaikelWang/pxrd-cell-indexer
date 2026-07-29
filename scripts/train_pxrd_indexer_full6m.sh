#!/usr/bin/env bash
# Retrain PXRD-indexer on the full ~5.96M set, from scratch.
#
# Defaults (4-GPU, VRAM-rich): global batch 2048, lr 4e-3 (linear scale from
# the old single-GPU 512/1e-3), bf16 autocast.
#
# Examples:
#   n_gpu=4 MASTER_PORT=16520 bash scripts/train_pxrd_indexer_full6m.sh
#   # match the old single-GPU schedule instead:
#   n_gpu=4 target_global_batch=512 lr=1e-3 amp=off \
#     out_dir=results/flow_seedgen/pxrd_indexer_full6m_v2 \
#     bash scripts/train_pxrd_indexer_full6m.sh
#   n_gpu=1 epochs=2 limit_train_batches=50 out_dir=/tmp/dbg \
#     bash scripts/train_pxrd_indexer_full6m.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${n_gpu:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
        n_gpu=$(python3 -c 'import os; print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))')
    else
        n_gpu=$(nvidia-smi -L | wc -l)
    fi
fi
[ "${n_gpu}" -lt 1 ] && { echo "ERROR: no GPU visible" >&2; exit 1; }

# 2048 = 4 × 512: same per-GPU microbatch that fitted on a single 4090.
target_global_batch=${target_global_batch:-2048}
if [ $((target_global_batch % n_gpu)) -ne 0 ]; then
    echo "ERROR: target_global_batch=${target_global_batch} must divide by n_gpu=${n_gpu}" >&2
    exit 1
fi
per_replica=$((target_global_batch / n_gpu))
batch_size=${batch_size:-${per_replica}}
if [ $((per_replica % batch_size)) -ne 0 ]; then
    echo "ERROR: batch_size=${batch_size} must divide per-replica batch ${per_replica}" >&2
    exit 1
fi
grad_accum=$((per_replica / batch_size))

epochs=${epochs:-34}
# Linear LR scale vs the old global-batch-512 / lr=1e-3 baseline.
lr=${lr:-$(python3 -c "print(1e-3 * ${target_global_batch} / 512)")}
warmup_epochs=${warmup_epochs:-1.0}
amp=${amp:-bf16}
eval_k=${eval_k:-100}
eval_every=${eval_every:-1}
# 25 vs 400 integration steps moved the <1% hit rate by 1pp (probe_sample_steps),
# so 50 is enough for selection; the final report can re-sample at 200.
sample_steps=${sample_steps:-50}
select_metric=${select_metric:-valid_1pct}
select_tol=${select_tol:-0.01}
valid_eval_n=${valid_eval_n:-300}
eval_workers=${eval_workers:-48}
mp100_every=${mp100_every:-0}
num_workers=${num_workers:-6}
seed=${seed:-42}
limit_train_batches=${limit_train_batches:-0}
out_dir=${out_dir:-results/flow_seedgen/pxrd_indexer_full6m_v2_gbs${target_global_batch}_${amp}}

data_dir=${data_dir:-/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets}
train_jsonl=${train_jsonl:-data/processed/train_full_niggli_seed42.jsonl}
valid_jsonl=${valid_jsonl:-data/processed/valid1400_niggli_seed42.jsonl}
stats=${stats:-data/processed/lattice_gstar6_stats_full_niggli_seed42.json}

for f in "${train_jsonl}" "${valid_jsonl}" "${stats}" \
         "${data_dir}/pxrd_241113_train.lmdb" "${data_dir}/pxrd_241113_valid.lmdb"; do
    [ -e "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

mkdir -p "${out_dir}" logs
log=${log:-logs/$(basename "${out_dir}").log}

echo "==== PXRD-indexer full-6M retrain (from scratch) ===="
echo "GPUs              = ${n_gpu}"
echo "global batch      = ${target_global_batch} (${batch_size} x ${n_gpu} x accum ${grad_accum})"
echo "epochs / lr / amp = ${epochs} / ${lr} / ${amp} (warmup ${warmup_epochs})"
echo "selection         = ${select_metric} @ tol ${select_tol}, n=${valid_eval_n}, K=${eval_k}"
echo "mp100             = every ${mp100_every} epoch(s) (0 = final only), report only"
echo "out_dir / log     = ${out_dir} / ${log}"
echo "====================================================="

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

launcher=(torchrun --nproc_per_node="${n_gpu}" --master_port="${MASTER_PORT:-16520}")
[ "${n_gpu}" -eq 1 ] && launcher=(python3 -u)

"${launcher[@]}" scripts/train_flow_seedgen.py \
    --backbone peaktf \
    --equiv-target off \
    --train-jsonl "${train_jsonl}" \
    --valid-jsonl "${valid_jsonl}" \
    --stats "${stats}" \
    --train-lmdb "${data_dir}/pxrd_241113_train.lmdb" \
    --valid-lmdb "${data_dir}/pxrd_241113_valid.lmdb" \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --grad-accum "${grad_accum}" \
    --lr "${lr}" \
    --weight-decay 0.05 \
    --warmup-epochs "${warmup_epochs}" \
    --amp "${amp}" \
    --flow-layers 6 \
    --flow-hidden 512 \
    --sample-steps "${sample_steps}" \
    --eval-k "${eval_k}" \
    --eval-every "${eval_every}" \
    --select-metric "${select_metric}" \
    --select-tol "${select_tol}" \
    --valid-eval-n "${valid_eval_n}" \
    --eval-workers "${eval_workers}" \
    --mp100-every "${mp100_every}" \
    --limit-train-batches "${limit_train_batches}" \
    --augment on \
    --num-workers "${num_workers}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    2>&1 | tee -a "${log}"
