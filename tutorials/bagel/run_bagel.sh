#!/usr/bin/env bash
# Run the BAGEL4 bacteriocin experiment end to end
set -uo pipefail

BAGEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BAGEL_DIR/../../.." && pwd)"
PY="${PYTHON:-python}"

METHODS_ALL=(plm tmvec dhr blastp)

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'; }

# ---- arguments -------------------------------------------------------------
TARGET_FDR=0.20; DECOY=extended_mkv2; SUFFIX=""; OUT_DIR=""; FIG_DIR=""
STEPS="calibrate,discover,prep,plot,addon"; DRY=0; FORCE=0; METHODS=()

collect() { local -n dest="$1"; local item; for item in ${2//,/ }; do dest+=("$item"); done; }

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY=1; shift; continue ;;
        --force)      FORCE=1; shift; continue ;;
        -h|--help)    usage; exit 0 ;;
        -*) ;;
        *) echo "[bagel] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    [ $# -ge 2 ] || { echo "[bagel] $1 needs a value" >&2; exit 1; }
    case "$1" in
        -target_fdr|--target-fdr|--target_fdr|-t) TARGET_FDR="$2" ;;
        -m|--methods)      collect METHODS "$2" ;;
        -d|--decoy-method) DECOY="$2" ;;
        -x|--decoy-suffix) SUFFIX="$2" ;;
        -o|--out-dir)      OUT_DIR="$2" ;;
        -f|--fig-dir)      FIG_DIR="$2" ;;
        -s|--steps)        STEPS="$2" ;;
        *) echo "[bagel] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift 2
done

[ ${#METHODS[@]} -eq 0 ] && METHODS=("${METHODS_ALL[@]}")

# id suffix the decoy generator stamped on every decoy query
if [ -z "$SUFFIX" ]; then
    case "$DECOY" in
        extended_mkv*) SUFFIX="_mkv" ;;
        mkv*)          SUFFIX="_mkv${DECOY#mkv}" ;;
        *shuf)         SUFFIX="_shuf" ;;
        *rev)          SUFFIX="_rev" ;;
        *dplm)         SUFFIX="_dplm" ;;
        *) echo "[bagel] cannot derive a decoy suffix for '$DECOY'; pass -x" >&2; exit 1 ;;
    esac
fi

# tsne.py / addon_plot.py take the short name; bagel_main.py takes the disk name
disk_name() {
    case "$1" in
        dhr)    echo dhr_postprocess ;;
        blastp) echo blastp_postprocessed ;;
        *)      echo "$1" ;;
    esac
}
in_list() { local n="$1"; shift; local x; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }
for m in "${METHODS[@]}"; do
    in_list "$m" "${METHODS_ALL[@]}" || {
        echo "[bagel] unknown method '$m'; expected one of: ${METHODS_ALL[*]}" >&2; exit 1; }
done

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    case "$s" in calibrate|discover|prep|plot|addon) ;;
        *) echo "[bagel] unknown step '$s'; expected calibrate, discover, prep, plot and/or addon" >&2; exit 1 ;;
    esac
done

OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/bagel}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/results/bagel}"
TSNE_DIR="$OUT_DIR/tsne_per_query"      # the three figure steps share this one

# ---- run -------------------------------------------------------------------
echo "[bagel] methods    : ${METHODS[*]}"
echo "[bagel] decoy      : $DECOY (id suffix $SUFFIX)"
echo "[bagel] target FDR : $TARGET_FDR"
echo "[bagel] derived data: $OUT_DIR"
echo "[bagel] figures     : $FIG_DIR"
echo "[bagel] steps      : $STEPS$([ "$DRY" = 1 ] && echo '  (dry run)')"

step() {
    local label="$1"; shift
    echo
    echo "[bagel] === $label ==="
    printf '[bagel] $ %s\n' "$*"
    [ "$DRY" = 1 ] && return 0
    "$@"
}

if want calibrate; then
    FORCE_FLAG=(); [ "$FORCE" = 1 ] && FORCE_FLAG=(--force)
    for m in "${METHODS[@]}"; do
        step "calibrate $m" "$PY" "$BAGEL_DIR/calibrate.py" \
            --search-method "$(disk_name "$m")" \
            --decoy-method "$DECOY" \
            --decoy-suffix "$SUFFIX" \
            --target-fdr "$TARGET_FDR" \
            --out-data-dir "$OUT_DIR" --work-dir "$OUT_DIR/calibration/$(disk_name "$m")" \
            ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} || {
            echo "[bagel] calibrate failed for $m" >&2; exit 1; }
    done
fi

if want discover; then
    for m in "${METHODS[@]}"; do
        step "discover $m" "$PY" "$BAGEL_DIR/bagel_main.py" discover \
            --search-method "$(disk_name "$m")" \
            --decoy-method "$DECOY" \
            --decoy-suffix "$SUFFIX" \
            --target-fdr "$TARGET_FDR" \
            --data-dir "$OUT_DIR" --out-dir "$FIG_DIR" || {
            echo "[bagel] discover failed for $m" >&2; exit 1; }
    done
fi

if want prep; then
    step "tsne prep" "$PY" "$BAGEL_DIR/tsne.py" prep \
        --methods "${METHODS[@]}" --decoy-method "$DECOY" --data-dir "$TSNE_DIR" \
        --coords-dir "$OUT_DIR" --scan-dir "$OUT_DIR" || {
        echo "[bagel] tsne prep failed" >&2; exit 1; }
fi

if want plot; then
    step "tsne plot" "$PY" "$BAGEL_DIR/tsne.py" plot \
        --methods "${METHODS[@]}" --decoy-method "$DECOY" \
        --data-dir "$TSNE_DIR" --plot-dir "$FIG_DIR/tsne_per_query" || {
        echo "[bagel] tsne plot failed" >&2; exit 1; }
fi

if want addon; then
    step "addon figures" "$PY" "$BAGEL_DIR/addon_plot.py" \
        --methods "${METHODS[@]}" --data-dir "$TSNE_DIR" \
        --score-dir "$OUT_DIR" --out-dir "$FIG_DIR/addon" || {
        echo "[bagel] addon_plot failed" >&2; exit 1; }
fi

echo
if [ "$DRY" = 1 ]; then
    echo "[bagel] dry run only, nothing was executed"
else
    echo "[bagel] done"
    want plot  && echo "[bagel] t-SNE figures -> $FIG_DIR/tsne_per_query"
    want addon && echo "[bagel] addon figures -> $FIG_DIR/addon"
fi
