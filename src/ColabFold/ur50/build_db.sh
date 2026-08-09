#!/usr/bin/env bash

# Usage:  bash src/ColabFold/ur50/build_db.sh <method> [GPUS]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
# shellcheck disable=SC1091
source src/PLMs_cmds/.env
METHOD="${1:?usage: build_db.sh <dhr_postprocess|tmvec|plm> [GPUS]}"
GPUS="${2:-0,1,2,3,4,5}"
DB=src/ColabFold/ur50/data_build

case "$METHOD" in
  dhr_postprocess)
    AGG="${DHR_SRC_DIR}/data/db_ur50/agg/index-ebd.index"
    [ -e "$AGG" ] && { echo "[skip] DHR index exists: $AGG"; exit 0; }
    echo "[build] DHR UR50 index (sharded single-GPU encode + aggregate)"
    DHR_GPU_OVERRIDE="$GPUS" bash "$DB/dhr/build_index_dhr.sh"
    ;;
  tmvec)
    F16=data/db/db_ur50_tmvec/ur50_embedding.f16
    [ -s "$F16" ] && { echo "[skip] TM-Vec DB exists: $F16"; exit 0; }
    echo "[build] TM-Vec UR50 DB (sharded encode -> merge to memmap)"
    bash "$DB/tmvec/encode_ur50_tmvec_sharded.sh" "$GPUS"
    "${TMVEC_PYTHON_PATH}" "$DB/tmvec/merge_tmvec_chunks.py" \
      --chunk-dir data/db/db_ur50_tmvec/chunks --split-dir data/uniref50/tmvec_chunks \
      --out-prefix data/db/db_ur50_tmvec/ur50_embedding
    ;;
  plm)
    F16=data/db/db_ur50_plm/ur50_embedding.f16
    [ -s "$F16" ] && { echo "[skip] PLM DB exists: $F16"; exit 0; }
    echo "[build] PLM UR50 DB (sharded encode -> staged merge -> memmap)"
    bash "$DB/plm/encode_ur50_plm_sharded_gpu7.sh"
    "${PLMSEARCH_PYTHON_PATH}" "$DB/plm/merge_plm_chunks_staged.py"
    "${PLMSEARCH_PYTHON_PATH}" "$DB/shared/pkl_to_memmap.py" \
      data/db/db_ur50_plm/ur50_embedding.pkl
    ;;
  *) echo "[ERROR] unknown method '$METHOD'"; exit 1 ;;
esac
echo "[DONE] $METHOD DB ready"
