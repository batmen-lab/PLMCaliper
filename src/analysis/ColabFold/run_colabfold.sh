#!/usr/bin/env bash
# Run the FDR-guided MSA / structure-prediction experiment end to end.
# Once: tsv. Per method: db -> search -> calibration -> msa -> predict -> evaluate.
# Then plot once, over every method at the same time.
#
#   bash run_colabfold.sh                          # all three methods, all steps
#   bash run_colabfold.sh -m "dhr plm"             # one subset
#   bash run_colabfold.sh -m dhr -s evaluate,plot  # redo the tail only
#   bash run_colabfold.sh -n                       # print the commands, run nothing
#
#   -m LIST   search methods from plm, tmvec, dhr_postprocess (default: all three)
#   -d NAME   dataset: ur50 or astral (default ur50)
#   -j N      jackhmmer iterations, also names the iter<N>/ run dir (default 1)
#   -g LIST   GPUs for the db step (default 0,1,2,3)
#   -G N      single GPU for the search step (default 0)
#   -f DIR    figures (default <repo>/results/ColabFold)
#   -s LIST   steps from tsv,db,search,calibration,msa,predict,evaluate,plot
#             (default: all eight)
#   -n        dry run: print the commands, run nothing
#   -h        this help
#
# `dhr` is shorthand for `dhr_postprocess`. `tsv` and `plot` run once; everything
# between them runs once per method. `plot` takes the whole method list at once,
# because plot.py compares the methods in a single figure.
#
# `tsv` rewrites data/uniref50/uniref50.fasta as the id<TAB>sequence table the
# pipeline shards and streams (build_data.py refuses to start without it). It is
# skipped when that table already exists -- delete it to rebuild.
#
# Keep dhr ahead of plm/tmvec: the decoy query table is written by the dhr search
# stage, and the other two only read it.
#
# The whole run tree (MSAs, predictions, metrics) is derived data and lives under
# data/ColabFold; only the figures go to results/ColabFold. The UniRef50 database
# and the retrieval score tables stay in data/ itself.
#
# build_data.py re-invokes itself inside each retrieval environment, so
# src/PLM_searching_cmds/.env has to be loaded first; this sources it for you if
# BASE_DIR is not already set. The predict step additionally needs localcolabfold.
# Run this after `conda activate plmcaliper`; set PYTHON=... to override.
set -uo pipefail

CF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CF_DIR/../../.." && pwd)"
PY="${PYTHON:-python}"

METHODS_ALL=(dhr_postprocess plm tmvec)
STEPS_ALL="tsv,db,search,calibration,msa,predict,evaluate,plot"
UR50_FASTA_REL="data/uniref50/uniref50.fasta"   # hardcoded in build_data.py too

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'; }

# ---- arguments -------------------------------------------------------------
DATASET=ur50; JACK=1; GPUS="0,1,2,3"; GPU=0; STEPS="$STEPS_ALL"; DRY=0; METHODS=(); FIG_DIR=""

collect() { local -n dest="$1"; local item; for item in ${2//,/ }; do dest+=("$item"); done; }

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY=1; shift; continue ;;
        -h|--help)    usage; exit 0 ;;
        -*) ;;
        *) echo "[colabfold] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    [ $# -ge 2 ] || { echo "[colabfold] $1 needs a value" >&2; exit 1; }
    case "$1" in
        -m|--methods)     collect METHODS "$2" ;;
        -d|--dataset)     DATASET="$2" ;;
        -j|--jack-iters)  JACK="$2" ;;
        -g|--gpus)        GPUS="$2" ;;
        -G|--gpu)         GPU="$2" ;;
        -f|--fig-dir)     FIG_DIR="$2" ;;
        -s|--steps)       STEPS="$2" ;;
        *) echo "[colabfold] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift 2
done

[ ${#METHODS[@]} -eq 0 ] && METHODS=("${METHODS_ALL[@]}")
FIG_DIR="${FIG_DIR:-$REPO_ROOT/results/ColabFold}"

case "$DATASET" in ur50|astral) ;;
    *) echo "[colabfold] unknown dataset '$DATASET'; expected ur50 or astral" >&2; exit 1 ;;
esac

# ---- normalise and validate ------------------------------------------------
in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

NORM=(); SEEN=()
for m in "${METHODS[@]}"; do
    [ "$m" = dhr ] && { echo "[colabfold] 'dhr' -> 'dhr_postprocess'"; m=dhr_postprocess; }
    in_list "$m" "${METHODS_ALL[@]}" || {
        echo "[colabfold] unknown method '$m'; expected one of: ${METHODS_ALL[*]}" >&2; exit 1; }
    in_list "$m" ${SEEN[@]+"${SEEN[@]}"} || { NORM+=("$m"); SEEN+=("$m"); }
done
METHODS=("${NORM[@]}")

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    in_list "$s" ${STEPS_ALL//,/ } || {
        echo "[colabfold] unknown step '$s'; expected from ${STEPS_ALL}" >&2; exit 1; }
done

# ---- retrieval interpreters -------------------------------------------------
# build_data.py fans out to $TMVEC_PYTHON_PATH / $PLMSEARCH_PYTHON_PATH / $DHR_PYTHON_PATH
ENV_FILE="$REPO_ROOT/src/PLM_searching_cmds/.env"
if [ -z "${BASE_DIR:-}" ]; then
    if [ -f "$ENV_FILE" ]; then
        echo "[colabfold] sourcing $ENV_FILE"
        # shellcheck disable=SC1090
        . "$ENV_FILE"
    elif want db || want search; then
        echo "[colabfold] $ENV_FILE not found -- run 'bash src/setup.sh' first" >&2
        [ "$DRY" = 1 ] || exit 1
    fi
fi

# ---- run -------------------------------------------------------------------
echo "[colabfold] methods    : ${METHODS[*]}"
echo "[colabfold] dataset    : $DATASET   jack-iters: $JACK"
echo "[colabfold] gpus       : db=$GPUS  search=$GPU"
echo "[colabfold] figures    : $FIG_DIR"
echo "[colabfold] steps      : $STEPS$([ "$DRY" = 1 ] && echo '  (dry run)')"

step() {
    local label="$1"; shift
    echo
    echo "[colabfold] === $label ==="
    printf '[colabfold] $ %s\n' "$*"
    [ "$DRY" = 1 ] && return 0
    "$@"
}

run_method() {
    local m="$1"
    if want db; then
        step "db $m" "$PY" "$CF_DIR/build_data.py" db \
            --dataset "$DATASET" --method "$m" --gpus "$GPUS" || return $?
    fi
    if want search; then
        step "search $m" "$PY" "$CF_DIR/build_data.py" search \
            --dataset "$DATASET" --method "$m" --gpu "$GPU" || return $?
    fi
    if want calibration; then
        step "calibration $m" "$PY" "$CF_DIR/calibration.py" \
            --dataset "$DATASET" --method "$m" || return $?
    fi
    if want msa; then
        step "msa $m" "$PY" "$CF_DIR/build_msa.py" \
            --dataset "$DATASET" --method "$m" --jack-iters "$JACK" || return $?
    fi
    if want predict; then
        step "predict $m" "$PY" "$CF_DIR/predict.py" \
            --dataset "$DATASET" --method "$m" --jack-iters "$JACK" || return $?
    fi
    if want evaluate; then
        step "evaluate $m" "$PY" "$CF_DIR/evaluate.py" \
            --dataset "$DATASET" --method "$m" --jack-iters "$JACK" --sync-plot-data || return $?
    fi
    return 0
}

if want tsv; then
    ur50_fa="$REPO_ROOT/$UR50_FASTA_REL"
    ur50_tsv="${ur50_fa%.fasta}.tsv"
    echo
    echo "[colabfold] === tsv ==="
    if [ "$DATASET" != ur50 ]; then
        echo "[colabfold] dataset $DATASET does not use the UniRef50 table, skipping"
    elif [ -s "$ur50_tsv" ]; then
        echo "[colabfold] $ur50_tsv already built, skipping (delete it to rebuild)"
    else
        printf '[colabfold] $ %s\n' "$PY $REPO_ROOT/src/utils/fa2tsv.py --fasta $ur50_fa"
        if [ "$DRY" != 1 ]; then
            "$PY" "$REPO_ROOT/src/utils/fa2tsv.py" --fasta "$ur50_fa" || {
                echo "[colabfold] fa2tsv failed" >&2; exit 1; }
        fi
    fi
fi

for m in "${METHODS[@]}"; do
    run_method "$m" || { echo "[colabfold] failed on method $m" >&2; exit 1; }
done

# plot.py takes every method at once -- one figure comparing them
if want plot; then
    step "plot" "$PY" "$CF_DIR/plot.py" \
        --dataset "$DATASET" --methods "${METHODS[@]}" --jack-iters "$JACK" \
        --plot-dir "$FIG_DIR" || {
        echo "[colabfold] plot failed" >&2; exit 1; }
fi

echo
[ "$DRY" = 1 ] && echo "[colabfold] dry run only, nothing was executed" || echo "[colabfold] done"
