#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." || exit 1
# shellcheck disable=SC1091
source src/PLMs_cmds/.env

GPUS="${1:-2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_WORKERS=${#GPU_ARR[@]}

FASTA="data/uniref50/uniref50.fasta"
CHUNK_DIR="data/uniref50/tmvec_chunks"
OUT_DIR="data/db/db_ur50_tmvec/chunks"
SEQS_PER_CHUNK="${TMVEC_SEQS_PER_CHUNK:-50000}"
PY="${TMVEC_PYTHON_PATH:?}"
WORKER="src/ColabFold/ur50/data_build/tmvec/encode_ur50_tmvec_worker.py"
LOGDIR="data/uniref50"

echo "==================== sharded TM-Vec start $(date) (GPUs $GPUS) ===================="
[ -s "$FASTA" ] || { echo "[ERROR] $FASTA missing"; exit 1; }
mkdir -p "$CHUNK_DIR" "$OUT_DIR"

# ---- 1. split fasta into chunks of SEQS_PER_CHUNK records (once, idempotent) ----
if [ ! -f "$CHUNK_DIR/.split_done" ]; then
  echo "[split] $FASTA -> $CHUNK_DIR ($SEQS_PER_CHUNK seqs/chunk) ..."
  rm -f "$CHUNK_DIR"/chunk_*.fasta
  awk -v n="$SEQS_PER_CHUNK" -v dir="$CHUNK_DIR" '
    /^>/ { idx=int(c/n);
           if (out=="" || idx!=cur) { if (out) close(out); cur=idx;
             out=sprintf("%s/chunk_%04d.fasta", dir, idx) }
           c++ }
    { print > out }
  ' "$FASTA"
  touch "$CHUNK_DIR/.split_done"
  echo "[split] done: $(ls "$CHUNK_DIR"/chunk_*.fasta 2>/dev/null | wc -l) chunks"
fi
N_CHUNKS=$(ls "$CHUNK_DIR"/chunk_*.fasta 2>/dev/null | wc -l)
echo "[info] $N_CHUNKS chunks, $NUM_WORKERS workers on GPUs $GPUS"

# ---- 2. launch one worker per GPU (each takes chunk_idx % NUM_WORKERS == wid) ----
pids=()
for wid in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$wid]}"
  log="$LOGDIR/encode_ur50_tmvec_gpu${gpu}.log"
  echo "[launch] worker $wid on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$WORKER" \
    --chunk-dir "$CHUNK_DIR" --out-dir "$OUT_DIR" \
    --worker-id "$wid" --num-workers "$NUM_WORKERS" >> "$log" 2>&1 &
  pids+=($!)
done

echo "[info] launched ${#pids[@]} workers: ${pids[*]}"
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done

done_n=$(ls "$OUT_DIR"/chunk_*.npy 2>/dev/null | wc -l)
echo "[ALL DONE] $done_n/$N_CHUNKS chunk npys present  rc=$rc  $(date)"
echo "==================== sharded TM-Vec end $(date) ===================="
exit "$rc"
