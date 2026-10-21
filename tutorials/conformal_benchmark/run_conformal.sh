#!/usr/bin/env bash
# Run the conformal-prediction benchmark: compute, then plot.
set -uo pipefail

CB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CB_DIR/../../.." && pwd)"
PY="${PYTHON:-python}"

METHODS_ALL=(plm tmvec dhr_postprocess blastp_postprocessed)
LEVELS_ALL=(random fold superfamily family)

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'; }

# ---- arguments -------------------------------------------------------------
LEVEL=random; N_CALIB=1000; N_TRIALS=5; STEPS="compute,plot"; DRY=0; METHODS=()
OUT_DIR=""; FIG_DIR=""

collect() { local -n dest="$1"; local item; for item in ${2//,/ }; do dest+=("$item"); done; }

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY=1; shift; continue ;;
        -h|--help)    usage; exit 0 ;;
        -*) ;;
        *) echo "[conformal] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    [ $# -ge 2 ] || { echo "[conformal] $1 needs a value" >&2; exit 1; }
    case "$1" in
        -m|--methods)                          collect METHODS "$2" ;;
        -split_level|--split-level|--split_level|-l) LEVEL="$2" ;;
        -c|--n-calib)                          N_CALIB="$2" ;;
        -t|--n-trials)                         N_TRIALS="$2" ;;
        -o|--out-dir)                          OUT_DIR="$2" ;;
        -f|--fig-dir)                          FIG_DIR="$2" ;;
        -s|--steps)                            STEPS="$2" ;;
        *) echo "[conformal] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift 2
done

[ ${#METHODS[@]} -eq 0 ] && METHODS=(blastp_postprocessed)
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/conformal}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/results/conformal}"

# ---- normalise and validate ------------------------------------------------
in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }

NORM=(); SEEN=()
for m in "${METHODS[@]}"; do
    case "$m" in
        dhr)    echo "[conformal] 'dhr' -> 'dhr_postprocess'";        m=dhr_postprocess ;;
        blastp) echo "[conformal] 'blastp' -> 'blastp_postprocessed'"; m=blastp_postprocessed ;;
    esac
    in_list "$m" "${METHODS_ALL[@]}" || {
        echo "[conformal] unknown method '$m'; expected one of: ${METHODS_ALL[*]}" >&2; exit 1; }
    in_list "$m" ${SEEN[@]+"${SEEN[@]}"} || { NORM+=("$m"); SEEN+=("$m"); }
done
METHODS=("${NORM[@]}")

in_list "$LEVEL" "${LEVELS_ALL[@]}" || {
    echo "[conformal] unknown split level '$LEVEL'; expected one of: ${LEVELS_ALL[*]}" >&2; exit 1; }

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    in_list "$s" compute plot || {
        echo "[conformal] unknown step '$s'; expected compute and/or plot" >&2; exit 1; }
done

# ---- run -------------------------------------------------------------------
echo "[conformal] methods     : ${METHODS[*]}"
echo "[conformal] split level : $LEVEL"
echo "[conformal] n-calib=$N_CALIB  n-trials=$N_TRIALS"
echo "[conformal] derived data: $OUT_DIR"
echo "[conformal] figures     : $FIG_DIR"
echo "[conformal] steps       : $STEPS$([ "$DRY" = 1 ] && echo '  (dry run)')"

step() {
    local label="$1"; shift
    echo
    echo "[conformal] === $label ==="
    printf '[conformal] $ %s\n' "$*"
    [ "$DRY" = 1 ] && return 0
    "$@"
}

for m in "${METHODS[@]}"; do
    if want compute; then
        step "compute $m" "$PY" "$CB_DIR/compute.py" all \
            --search-method "$m" --split-level "$LEVEL" \
            --n-calib "$N_CALIB" --n-trials "$N_TRIALS" \
            --out-dir "$OUT_DIR" --cache-dir "$OUT_DIR/cache" || {
            echo "[conformal] compute failed for $m" >&2; exit 1; }
    fi
    if want plot; then
        step "plot $m" "$PY" "$CB_DIR/plot.py" all \
            --search-method "$m" --split-level "$LEVEL" \
            --n-calib "$N_CALIB" --n-trials "$N_TRIALS" \
            --out-dir "$OUT_DIR" --plot-dir "$FIG_DIR" || {
            echo "[conformal] plot failed for $m" >&2; exit 1; }
    fi
done

echo
[ "$DRY" = 1 ] && echo "[conformal] dry run only, nothing was executed" || echo "[conformal] done"
