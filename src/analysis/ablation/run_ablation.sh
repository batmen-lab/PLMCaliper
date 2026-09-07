#!/usr/bin/env bash
# Run the decoy/calibration ablation end to end: calibrate -> fdr -> plot.
#
#   bash run_ablation.sh all_ablation                 every search method x every decoy design
#   bash run_ablation.sh -m "plm dhr" -d "shuf mkv"   one explicit sub-grid
#   bash run_ablation.sh -m plm -d shuf -d rev        -m/-d repeat, or take a quoted list
#
#   -m LIST   search methods: plm, tmvec, dhr_postprocess, blastp_postprocessed
#   -d LIST   decoy designs:  shuf, rev, dplm, mkv2 (also extended_shuf, extended_mkv2)
#   -j N      worker processes for the FDR step (default 4)
#   -t TAU    pinball tau; goes to calibrate and into the figure names (default 0.25)
#   -o DIR    derived data: calibrated tables + plotting CSVs
#             (default <repo>/data/ablation)
#   -f DIR    figures (default <repo>/results/ablation)
#   -s LIST   steps to run, from calibrate,fdr,plot (default: all three)
#   -n        dry run: print the commands, run nothing
#   -h        this help
#
# `dhr` and `blastp` are shorthand for the postprocessed tables the ablation
# actually reads, and `mkv` for `mkv2`. Give -m or -d alone and the other axis
# runs in full. Set PYTHON=... to pick an interpreter; the default `python` is
# the plmcaliper env from the README.
#
# Everything this experiment derives lands under data/ablation, and only the
# figures under results/ablation. The retrieval score tables it reads stay in
# data/ itself, because the other experiments read the same files.
set -uo pipefail

ABL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ABL_DIR/../../.." && pwd)"
PY="${PYTHON:-python}"

SEARCH_ALL=(plm tmvec dhr_postprocess blastp_postprocessed)
DECOY_ALL=(shuf rev dplm mkv2)
SEARCH_KNOWN=("${SEARCH_ALL[@]}")
DECOY_KNOWN=("${DECOY_ALL[@]}" extended_shuf extended_mkv2)

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'; }

# ---- arguments -------------------------------------------------------------
ALL=0; N_JOBS=4; TAU=0.25; OUT_DIR=""; FIG_DIR=""; STEPS="calibrate,fdr,plot"; DRY=0
SEARCH=(); DECOY=()

REST=()
for a in "$@"; do
    case "$a" in
        all_ablation|all) ALL=1 ;;
        *) REST+=("$a") ;;
    esac
done
set -- ${REST[@]+"${REST[@]}"}

# one -m/-d may carry a whitespace- or comma-separated list, and may repeat
collect() {  # $1 = array name, $2 = raw OPTARG
    local -n dest="$1"
    local item
    for item in ${2//,/ }; do dest+=("$item"); done
}

while getopts "m:d:j:t:o:f:s:nh" opt; do
    case "$opt" in
        m) collect SEARCH "$OPTARG" ;;
        d) collect DECOY "$OPTARG" ;;
        j) N_JOBS="$OPTARG" ;;
        t) TAU="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        f) FIG_DIR="$OPTARG" ;;
        s) STEPS="$OPTARG" ;;
        n) DRY=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ $# -gt 0 ]; then
    echo "[ablation] unexpected argument: $1" >&2; usage >&2; exit 1
fi

if [ "$ALL" = 0 ] && [ ${#SEARCH[@]} -eq 0 ] && [ ${#DECOY[@]} -eq 0 ]; then
    echo "[ablation] nothing selected. Pass 'all_ablation' for the full grid," >&2
    echo "           or -m/-d for a sub-grid." >&2
    usage >&2; exit 1
fi

[ ${#SEARCH[@]} -eq 0 ] && SEARCH=("${SEARCH_ALL[@]}")
[ ${#DECOY[@]} -eq 0 ] && DECOY=("${DECOY_ALL[@]}")
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/ablation}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/results/ablation}"

# ---- normalise and validate ------------------------------------------------
in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

norm() {  # $1 = array name, $2 = kind, $3.. = allowed values
    local -n arr="$1"; local kind="$2"; shift 2
    local allowed=("$@") out=() seen=() v c
    for v in "${arr[@]}"; do
        case "$kind:$v" in
            search:dhr)    c=dhr_postprocess ;;
            search:blastp) c=blastp_postprocessed ;;
            decoy:mkv)     c=mkv2 ;;
            *)             c="$v" ;;
        esac
        [ "$c" != "$v" ] && echo "[ablation] '$v' -> '$c'"
        if ! in_list "$c" "${allowed[@]}"; then
            echo "[ablation] unknown $kind method '$v'; expected one of: ${allowed[*]}" >&2
            exit 1
        fi
        in_list "$c" ${seen[@]+"${seen[@]}"} || { out+=("$c"); seen+=("$c"); }
    done
    arr=("${out[@]}")
}
norm SEARCH search "${SEARCH_KNOWN[@]}"
norm DECOY decoy "${DECOY_KNOWN[@]}"

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    case "$s" in calibrate|fdr|plot) ;;
        *) echo "[ablation] unknown step '$s'; expected calibrate, fdr and/or plot" >&2; exit 1 ;;
    esac
done

# ---- run -------------------------------------------------------------------
echo "[ablation] search methods : ${SEARCH[*]}"
echo "[ablation] decoy designs  : ${DECOY[*]}"
echo "[ablation] combinations   : $(( ${#SEARCH[@]} * ${#DECOY[@]} ))  (tau=$TAU)"
echo "[ablation] derived data   : $OUT_DIR"
echo "[ablation] figures        : $FIG_DIR"
echo "[ablation] steps          : $STEPS${DRY:+ }$([ "$DRY" = 1 ] && echo '(dry run)')"

step() {  # $1 = label, rest = command
    local label="$1"; shift
    echo
    echo "[ablation] === $label ==="
    printf '[ablation] $ %s\n' "$*"
    [ "$DRY" = 1 ] && return 0
    "$@"
}

if want calibrate; then
    step "calibrate" "$PY" "$ABL_DIR/calibrate.py" \
        --search-methods "${SEARCH[@]}" --decoy-methods "${DECOY[@]}" \
        --tau "$TAU" --out-dir "$OUT_DIR" || {
        echo "[ablation] calibrate failed" >&2; exit 1; }
fi

if want fdr; then
    step "fdr" "$PY" "$ABL_DIR/fdr.py" \
        --search-methods "${SEARCH[@]}" --decoy-methods "${DECOY[@]}" \
        --n-jobs "$N_JOBS" --out-dir "$OUT_DIR" || {
        echo "[ablation] fdr failed" >&2; exit 1; }
fi

plot_status=0
if want plot; then
    step "plot" "$PY" "$ABL_DIR/plot.py" \
        --search-methods "${SEARCH[@]}" --decoy-methods "${DECOY[@]}" \
        --tau "$TAU" --data-dir "$OUT_DIR/plot_data" --plot-dir "$FIG_DIR" || plot_status=$?
fi

echo
if [ "$DRY" = 1 ]; then
    echo "[ablation] dry run only, nothing was executed"
elif [ "$plot_status" -ne 0 ]; then
    echo "[ablation] plot.py reported failures (see [ERROR] lines above);" >&2
    echo "           combinations without score tables are the usual cause." >&2
    exit "$plot_status"
else
    echo "[ablation] done"
    want fdr  && echo "[ablation] plotting CSVs -> $OUT_DIR/plot_data"
    want plot && echo "[ablation] figures       -> $FIG_DIR"
fi
