#!/usr/bin/env bash
# bash run_search.sh -m <plm|tmvec|dhr|blastp> -q <query> -t <target>
# Requires: bash src/setup.sh && source src/PLM_searching_cmds/.env    Query/target are .tsv for dhr, .fa otherwise.
set -euo pipefail
trap 'echo "[run_search] FAILED (line $LINENO): $BASH_COMMAND" >&2' ERR

METHOD=""; QUERY_PATH=""; TARGET_PATH=""
while getopts "m:q:t:" opt; do
    case $opt in
        m) METHOD="$OPTARG" ;;
        q) QUERY_PATH="$OPTARG" ;;
        t) TARGET_PATH="$OPTARG" ;;
        *) echo "Usage: bash $0 -m <plm|tmvec|dhr|blastp> -q <query> -t <target>" >&2; exit 1 ;;
    esac
done

[[ -n "$METHOD" && -n "$QUERY_PATH" && -n "$TARGET_PATH" ]] || {
    echo "Usage: bash $0 -m <plm|tmvec|dhr|blastp> -q <query> -t <target>" >&2; exit 1; }
[[ -n "${BASE_DIR:-}" ]] || {
    echo "[run_search] BASE_DIR unset -- run 'bash src/setup.sh && source src/PLM_searching_cmds/.env' first" >&2; exit 1; }

# OUT_TAG feeds the parser's output filename -- keep these strings exact.
case "$METHOD" in
    plm)    ENV_PREFIX=PLMSEARCH; QUERY_EXT=".fa";  OUT_TAG="result_plm" ;;
    tmvec)  ENV_PREFIX=TMVEC;     QUERY_EXT=".fa";  OUT_TAG="result_tmvec" ;;
    dhr)    ENV_PREFIX=DHR;       QUERY_EXT=".tsv"; OUT_TAG="result_dhr" ;;
    blastp) ENV_PREFIX=BLASTP;    QUERY_EXT=".fa";  OUT_TAG="result_blastp" ;;
    *) echo "[run_search] unknown method: $METHOD (expected plm|tmvec|dhr|blastp)" >&2; exit 1 ;;
esac

[[ -s "$TARGET_PATH" ]] || {
    echo "[run_search] target not found or empty: $TARGET_PATH" >&2; exit 1; }

# Hits to keep per query: the whole target DB, so a database of any size works
# without hand-tuning DB_KHITS/PARSER_KHITS. DHR reads a `id<TAB>seq` .tsv; the
# others read FASTA.
case "$TARGET_PATH" in
    *.tsv) TARGET_N=$(grep -c . "$TARGET_PATH") ;;
    *)     TARGET_N=$(grep -c '^>' "$TARGET_PATH") ;;
esac
[[ "$TARGET_N" -gt 0 ]] || {
    echo "[run_search] no sequences found in $TARGET_PATH" >&2; exit 1; }
DEFAULT_KHITS=$TARGET_N

# SCOP targets carry SCOP labels in the FASTA header; anything else (the BAGEL
# bacteriocin sets) get their homology labels from the BAGEL class encoded in the id.
case "$(basename "$TARGET_PATH" "$QUERY_EXT")" in
    astral95) TARGET_MODE=scop; BASEDB="SCOPe95" ;;
    astral*)  TARGET_MODE=scop; BASEDB="SCOPe40" ;;
    *)        TARGET_MODE=bagel ;;
esac

DB_KHITS="${DB_KHITS:-$DEFAULT_KHITS}"
PARSER_KHITS="${PARSER_KHITS:-$DEFAULT_KHITS}"
[[ "$DB_KHITS" -ge "$PARSER_KHITS" ]] || {
    echo "[run_search] DB_KHITS ($DB_KHITS) must be >= PARSER_KHITS ($PARSER_KHITS)" >&2; exit 1; }

_env() { local n="${ENV_PREFIX}_$1"; echo "${!n:-}"; }
SRC_DIR="$(_env SRC_DIR)"
GPU_DEVICES="$(_env GPU_DEVICES)"
NUM_PARALLEL="$(_env NUM_PARALLEL)"
export PATH="$(dirname "$(_env PYTHON_PATH)"):$PATH"   # the tools shell out to bare `python`

if [[ "$METHOD" == "plm" ]]; then
    PLM_EMBED_SCRIPT="$SRC_DIR/plmsearch/embedding_generate.py"
    [[ -f "$PLM_EMBED_SCRIPT" ]] || PLM_EMBED_SCRIPT="$SRC_DIR/plmsearch/embedding_generate_esm1b.py"
    [[ -f "$PLM_EMBED_SCRIPT" ]] || {
        echo "[run_search] no PLMSearch embedding script found under $SRC_DIR/plmsearch" >&2
        exit 1
    }
fi

query_basename="$(basename "$QUERY_PATH" "$QUERY_EXT")"
target_basename="$(basename "$TARGET_PATH" "$QUERY_EXT")"
if [[ "$METHOD" == "blastp" && "$TARGET_MODE" == "bagel" ]]; then
    OUT_TAG="${OUT_TAG}_${query_basename}_${target_basename}_blosum62_Q11R1"
elif [[ "$METHOD" == "blastp" ]]; then
    OUT_TAG="${OUT_TAG}_${query_basename}_blosum62_Q11R1"
else
    OUT_TAG="${OUT_TAG}_${query_basename}"
fi
OUTPUT_DIR="$TEMP_DIR/$OUT_TAG"
TEMP_DATA_SPLIT_DIR="$TEMP_DIR/${query_basename}_batch"
mkdir -p "$OUTPUT_DIR"

echo "[run_search] $METHOD | query=$query_basename target=$target_basename (${TARGET_N} seqs) | k=$DB_KHITS"

build_db() {
    case "$METHOD" in
    plm)
        DB_PATH="$BASE_DIR/data/db/db_${target_basename}_plm/${target_basename}_embedding.pkl"
        [[ -f "$DB_PATH" ]] && return
        mkdir -p "$(dirname "$DB_PATH")"
        CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$PLM_EMBED_SCRIPT" \
            -emp "$SRC_DIR/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt" \
            -f "$TARGET_PATH" -e "$DB_PATH"
        ;;
    tmvec)
        DB_PATH="$BASE_DIR/data/db/db_${target_basename}_tmvec"
        [[ -d "$DB_PATH" ]] && return
        mkdir -p "$(dirname "$DB_PATH")"
        CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/scripts/tmvec-build-database" \
            --input-fasta "$TARGET_PATH" \
            --tm-vec-model "$SRC_DIR/model/tm_vec_cath_model.ckpt" \
            --tm-vec-config-path "$SRC_DIR/model/tm_vec_cath_model_params.json" \
            --protrans-model "Rostlab/prot_t5_xl_half_uniref50-enc" \
            --device "gpu" --output "$DB_PATH"
        ;;
    dhr)
        DB_PATH="$DATA_DIR/db/db_${target_basename}_dhr"
        [[ -d "$DB_PATH/agg" ]] && return
        CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/do_embedding.py" \
            trainer.ur90_path="$(realpath "$TARGET_PATH")" \
            model.ckpt_path="$SRC_DIR/dhr2_ckpt" hydra.run.dir="$DB_PATH"
        CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/do_agg.py" \
            -s "$TARGET_PATH" -e "$DB_PATH/ebd" -o "$DB_PATH/agg"
        ;;
    blastp)
        DB_DIR="$BASE_DIR/data/db/db_${target_basename}_blastp"
        DB_PATH="$DB_DIR/${target_basename}_db"
        [[ -d "$DB_DIR" ]] && return
        mkdir -p "$DB_DIR"
        # the BAGEL sets were built without -parse_seqids; keep both as they were
        if [[ "$TARGET_MODE" == "scop" ]]; then
            makeblastdb -in "$TARGET_PATH" -parse_seqids -dbtype prot -out "$DB_PATH"
        else
            makeblastdb -in "$TARGET_PATH" -dbtype prot -out "$DB_PATH"
        fi
        ;;
    esac
}

search_plm() {
    # Query embeddings have always been cached under db_astral_plm/ regardless of the
    # target; kept as-is so the existing cache stays valid.
    local query_embed="$BASE_DIR/data/db/db_astral_plm/db_${query_basename}_embedding.pkl"
    mkdir -p "$(dirname "$query_embed")"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$PLM_EMBED_SCRIPT" \
        -emp "$SRC_DIR/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt" \
        -f "$QUERY_PATH" -e "$query_embed"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/plmsearch/main_similarity.py" \
        -iqe "$query_embed" -ite "$DB_PATH" \
        -smp "$SRC_DIR/plmsearch_data/model/plmsearch.sav" \
        -osr "$OUTPUT_DIR/${query_basename}.txt" -k "$DB_KHITS"
}

search_tmvec() {
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/scripts/tmvec-search" \
        --query "$QUERY_PATH" --output "$OUTPUT_DIR/${query_basename}.txt" \
        --tm-vec-model "$SRC_DIR/model/tm_vec_cath_model.ckpt" \
        --tm-vec-config "$SRC_DIR/model/tm_vec_cath_model_params.json" \
        --database "$DB_PATH/db.npy" --metadata "$DB_PATH/meta.npy" \
        --database-fasta "$TARGET_PATH" \
        --protrans-model "Rostlab/prot_t5_xl_half_uniref50-enc" \
        --device "gpu" --output-format "tabular" --k-nearest-neighbors "$DB_KHITS"
}

search_dhr() {
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python "$SRC_DIR/do_retrieval.py" \
        -i "$QUERY_PATH" -d "$DB_PATH/agg" \
        -o "$OUTPUT_DIR/${query_basename}.txt" -n "$DB_KHITS"
}

search_blastp() {
    # BAGEL queries are small enough to run in one shot, and report every target
    if [[ "$TARGET_MODE" == "bagel" ]]; then
        blastp -num_alignments 0 -num_descriptions "$(grep -c '^>' "$TARGET_PATH")" \
               -gapopen 11 -gapextend 1 -evalue 100000000000 \
               -matrix BLOSUM62 -comp_based_stats -2 -max_hsps 1 \
               -db "$DB_PATH" -query "$QUERY_PATH" \
               > "$OUTPUT_DIR/${query_basename}.out"
        return
    fi

    # ASTRAL: blastp has no batching of its own, so shard and run NUM_PARALLEL at a time
    [[ -d "$TEMP_DATA_SPLIT_DIR" ]] || \
        python "$BASE_DIR/src/utils/parallel_utils.py" \
            --fasta_path "$QUERY_PATH" --output_dir "$TEMP_DATA_SPLIT_DIR"

    local n=0
    for batch in "$TEMP_DATA_SPLIT_DIR"/*.fa; do
        blastp -num_alignments 0 -num_descriptions "$DB_KHITS" \
               -gapopen 11 -gapextend 1 -evalue 100000000000 \
               -matrix BLOSUM62 -comp_based_stats -2 \
               -db "$DB_PATH" -query "$batch" -max_hsps 1 \
               > "$OUTPUT_DIR/$(basename "${batch%.fa}").out" &
        if (( ++n >= NUM_PARALLEL )); then wait; n=0; fi
    done
    wait
}

build_db
"search_${METHOD}"

export PATH="$(dirname "$EVALPLMS_PYTHON_PATH"):$PATH"

if [[ "$TARGET_MODE" == "scop" ]]; then
    python "$BASE_DIR/src/utils/process_searching_score.py" parse \
        --data_type "$METHOD" \
        --input_path "$OUTPUT_DIR" \
        --save_path "$DATA_DIR" \
        --max_hits "$PARSER_KHITS" \
        --basedb "$BASEDB"
    PARSED="$DATA_DIR/parsed_result/${OUT_TAG}_hit${PARSER_KHITS}.txt"
else
    # decoy suffix to strip before the id's BAGEL class is read off it
    case "$query_basename" in
        *_extended_shuf|*_shuf) TYPE_SUFFIX="shuf" ;;
        *_extended_mkv*)        TYPE_SUFFIX="mkv" ;;
        *_mkv1)                 TYPE_SUFFIX="mkv1" ;;
        *_mkv2)                 TYPE_SUFFIX="mkv2" ;;
        *_rev)                  TYPE_SUFFIX="rev" ;;
        *_dplm)                 TYPE_SUFFIX="dplm" ;;
        *)                      TYPE_SUFFIX="" ;;
    esac
    [[ "$METHOD" == "blastp" ]] && RAW_EXT="out" || RAW_EXT="txt"
    PARSED="$DATA_DIR/result_${METHOD}_${query_basename}_${target_basename}.txt"
    python "$BASE_DIR/src/utils/process_searching_score.py" parse-bagel \
        --method "$METHOD" \
        --score-file "$OUTPUT_DIR/${query_basename}.${RAW_EXT}" \
        --out-path "$PARSED" \
        --type-suffix "$TYPE_SUFFIX" \
        --max-hits "$PARSER_KHITS"
fi

echo "[run_search] $METHOD done -> $PARSED"
