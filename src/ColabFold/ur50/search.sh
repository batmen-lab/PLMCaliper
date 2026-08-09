#!/usr/bin/env bash

# Usage:  bash src/ColabFold/ur50/search.sh <method> [GPU]
#         method = dhr_postprocess | tmvec | plm      GPU defaults to 0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
# shellcheck disable=SC1091
source src/PLMs_cmds/.env
METHOD="${1:?usage: search.sh <dhr_postprocess|tmvec|plm> [GPU]}"
GPU="${2:-0}"
DB=src/ColabFold/ur50/data_build
SD=data/search_data; mkdir -p "$SD"
REAL="$SD/result_${METHOD}_astral4f_ur50.txt"
DECOY="$SD/result_${METHOD}_astral4f_extended_shuf_ur50.txt"

if [ -s "$REAL" ] && [ -s "$DECOY" ]; then
  echo "[skip] both tables present in $SD for $METHOD"; exit 0
fi

case "$METHOD" in
  tmvec)
    TDB=data/db/db_ur50_tmvec
    [ -s "$TDB/ur50_embedding.f16" ] || { echo "[ERROR] TM-Vec DB missing -> run build_db.sh tmvec"; exit 1; }
    tv() {  # $1=query tsv basename  $2=emb name  $3=out
      local tsv="data/$1.tsv" emb="$TDB/query_$2"
      [ -s "$3" ] && { echo "[skip] $3"; return; }
      [ -s "$emb.npy" ] || CUDA_VISIBLE_DEVICES="$GPU" "${TMVEC_PYTHON_PATH}" \
        "$DB/tmvec/encode_query_tmvec.py" --query-tsv "$tsv" --out-prefix "$emb"
      "${TMVEC_PYTHON_PATH}" "$DB/tmvec/tmvec_search_ur50_chunked.py" \
        --query-emb "$emb.npy" --query-ids "$emb.ids.txt" \
        --target-emb "$TDB/ur50_embedding.f16" --out "$3" -k 1000000 --device "$GPU"
    }
    tv astral4f_query          astral4f               "$REAL"
    tv astral4f_extended_shuf  astral4f_extended_shuf "$DECOY"
    ;;
  plm)
    PDB=data/db/db_ur50_plm; MODEL=libs/PLMSearch-main/plmsearch_data/model/plmsearch.sav
    RAW=data/uniref50/plm_search_astral4f; mkdir -p "$RAW"
    [ -s "$PDB/ur50_embedding.f16" ] || { echo "[ERROR] PLM DB missing -> run build_db.sh plm"; exit 1; }
    ps() {  # $1=query pkl  $2=raw tag  $3=out
      [ -s "$3" ] && { echo "[skip] $3"; return; }
      "${PLMSEARCH_PYTHON_PATH}" "$DB/plm/plmsearch_ur50_chunked.py" \
        -iqe "$1" -ite "$PDB/ur50_embedding.f16" -smp "$MODEL" \
        -osr "$RAW/$2.txt" -k 1000000 --device "$GPU"
      awk -F'\t' 'BEGIN{OFS="\t"; print "qid","tid","score","homo_type","rank"}
        $1!=last{last=$1; r=0} {r++; print $1,$2,$3,-1,r}' "$RAW/$2.txt" > "$3"
    }
    ps "$PDB/query_astral4f_emb.pkl"      astral4f      "$REAL"
    ps "$PDB/query_astral4f_shuf_emb.pkl" astral4f_shuf "$DECOY"
    ;;
  dhr_postprocess)
    AGG="${DHR_SRC_DIR}/data/db_ur50/agg/index-ebd.index"
    [ -e "$AGG" ] || { echo "[ERROR] DHR index missing -> run build_db.sh dhr_postprocess"; exit 1; }
    # legacy script writes into data/; then move the two tables into search_data
    CUDA_VISIBLE_DEVICES="$GPU" bash "$DB/dhr/search_dhr_astral4f_ur50.sh"
    for suf in astral4f_ur50 astral4f_extended_shuf_ur50; do
      src="data/result_dhr_postprocess_${suf}.txt"
      [ -s "$src" ] && mv "$src" "$SD/$(basename "$src")"
    done
    ;;
  *) echo "[ERROR] unknown method '$METHOD'"; exit 1 ;;
esac
echo "[DONE] $METHOD search -> $REAL , $DECOY"
