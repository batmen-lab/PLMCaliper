#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." || exit 1

source src/PLMs_cmds/.env

GPU="${UR50_PLM_GPU:-7}"
FASTA="data/uniref50/uniref50.fasta"
CHUNK_DIR="data/uniref50/plm_chunks"        
OUT_DIR="data/db/db_ur50_plm/chunks"    
SEQS_PER_CHUNK="${PLM_SEQS_PER_CHUNK:-500000}"
ESM="${PLMSEARCH_SRC_DIR}/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt"
PY="${PLMSEARCH_PYTHON_PATH:?}"
GEN="${PLMSEARCH_SRC_DIR}/plmsearch/embedding_generate_esm1b.py"
LOG=data/uniref50/encode_ur50_plm_sharded.log

exec >> "$LOG" 2>&1
echo "==================== sharded PLM start $(date) (GPU $GPU) ===================="

[ -s "$FASTA" ] || { echo "[ERROR] $FASTA missing"; exit 1; }
[ -s "$ESM" ]   || { echo "[ERROR] ESM ckpt $ESM missing"; exit 1; }
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

# ---- 2. encode each chunk -> own pkl, skip done (resumable) ----
mapfile -t CHUNKS < <(ls "$CHUNK_DIR"/chunk_*.fasta | sort)
echo "[info] ${#CHUNKS[@]} chunks total, GPU $GPU"
done_n=0
for ch in "${CHUNKS[@]}"; do
  name=$(basename "$ch" .fasta)
  out="$OUT_DIR/${name}.pkl"
  if [ -s "$out" ]; then done_n=$((done_n+1)); continue; fi
  echo "  [$(date +%H:%M)] encoding ${name} -> ${out}"
  tmp="${out}.tmp"
  if CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$GEN" -emp "$ESM" -f "$ch" -e "$tmp"; then
    mv "$tmp" "$out"
    done_n=$((done_n+1))
    echo "  [$(date +%H:%M)] done ${name} (${done_n}/${#CHUNKS[@]})"
  else
    echo "  [ERROR] ${name} failed (rc=$?); removing tmp, re-run to resume"
    rm -f "$tmp"; exit 1
  fi
done

echo "[ALL DONE] ${done_n}/${#CHUNKS[@]} chunks -> $OUT_DIR  $(date)"
echo "==================== sharded PLM end $(date) ===================="
