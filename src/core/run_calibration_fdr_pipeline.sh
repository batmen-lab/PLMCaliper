set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PERFORMANCE_DIR="${PERFORMANCE_DIR:-${PROJECT_ROOT}/results_final/performance}"
CURVE_ARCHIVE_DIR="${CURVE_ARCHIVE_DIR:-${PERFORMANCE_DIR}/data}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate calib_gam

if [ -f "${PROJECT_ROOT}/src/PLMs_cmds/.env" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/src/PLMs_cmds/.env"
fi

PYTHON="${CALIB_FDR_PYTHON:-$(command -v python)}"

# Large datasets auto-stream one query at a time during FDR (see --force-load-all).
# FDR is CPU-bound (NumPy); use N_JOBS for CPU parallelization, not GPUs.
QUERY_NAME="${QUERY_NAME:-astral}"
TARGET_NAME="${TARGET_NAME:-astral}"
N_JOBS="${N_JOBS:-16}"

# Add more entries to run all combinations in a loop
# (overridable via SEARCH_METHODS / DECOY_METHODS env vars, space-separated)
read -r -a SEARCH_METHODS <<< "${SEARCH_METHODS:-plm tmvec dhr_postprocess}"
read -r -a DECOY_METHODS <<< "${DECOY_METHODS:-extended_shuf shuf rev dplm PGmsk25pct}"
# -----------------------------------------------------

total=$((${#SEARCH_METHODS[@]} * ${#DECOY_METHODS[@]}))
idx=0
failed=0

echo "conda env: calib_gam"
echo "Using Python: ${PYTHON}"
echo "query_name=${QUERY_NAME}"
echo "target_name=${TARGET_NAME}"
echo "search_methods=${SEARCH_METHODS[*]}"
echo "decoy_methods=${DECOY_METHODS[*]}"
echo "n_jobs=${N_JOBS}"
echo "performance_dir=${PERFORMANCE_DIR}"
echo "curve_archive_dir=${CURVE_ARCHIVE_DIR}"
echo "total runs: ${total}"
echo "=============================="

mkdir -p "${PERFORMANCE_DIR}" "${CURVE_ARCHIVE_DIR}"

for search in "${SEARCH_METHODS[@]}"; do
    for decoy in "${DECOY_METHODS[@]}"; do
        idx=$((idx + 1))
        echo ""
        echo ">>> Run ${idx} / ${total}: ${search} / ${decoy}"
        echo "=============================="

        if "${PYTHON}" "${SCRIPT_DIR}/run_calibration_fdr_pipeline.py" \
            --query-name "${QUERY_NAME}" \
            --target-name "${TARGET_NAME}" \
            --search-methods "${search}" \
            --decoy-methods "${decoy}" \
            --n-jobs "${N_JOBS}" \
            --curve-archive-dir "${CURVE_ARCHIVE_DIR}" \
            --plot-dir "${PERFORMANCE_DIR}" \
            "$@"; then
            echo "[OK] ${search} / ${decoy}"
        else
            echo "[ERROR] ${search} / ${decoy} failed (exit $?)"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "=============================="
echo "Finished ${total} runs: $((total - failed)) ok, ${failed} failed"
if [ "${failed}" -gt 0 ]; then
    exit 1
fi
