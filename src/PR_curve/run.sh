set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f "${SCRIPT_DIR}/../PLMs_cmds/.env" ]; then
    source "${SCRIPT_DIR}/../PLMs_cmds/.env"
fi

PYTHON="${CALIB_FDR_PYTHON:-${EVALPLMS_PYTHON_PATH:-python3}}"
export PYTHONUNBUFFERED=1

# Large datasets (e.g. astral) auto-stream one query at a time to avoid OOM.
# Use --force-load-all only if you need the old in-memory path.
SEQ_NAME="astral"
SEARCH_METHODS=(plm)
DECOY_METHODS=(extended_shuf)

# --- Query subset for pooling (one qid per line; # for comments) ---
# Set to a .txt path to pool only those queries (grep-extract on large files).
# Set to "" to pool all queries in the dataset.
QUERY_LIST="${SCRIPT_DIR}/query_list.txt"
# QUERY_LIST=""
# -----------------------------------------------------

echo "Using Python: ${PYTHON}"
echo "seq_name=${SEQ_NAME}"
echo "search_methods=${SEARCH_METHODS[*]}"
echo "decoy_methods=${DECOY_METHODS[*]}"

EXTRA_ARGS=()
if [[ -n "${QUERY_LIST}" ]]; then
    if [[ ! -f "${QUERY_LIST}" ]]; then
        echo "ERROR: QUERY_LIST is set but file not found: ${QUERY_LIST}" >&2
        echo "  Create the file (one qid per line) or set QUERY_LIST=\"\" to pool all queries." >&2
        exit 1
    fi
    echo "query_list=${QUERY_LIST} (pooled metrics = mean over this subset only)"
    EXTRA_ARGS+=(--query-list "${QUERY_LIST}")
else
    echo "query_list=(none) — pooling over all queries"
fi

exec "${PYTHON}" "${SCRIPT_DIR}/compute_efdr_recall.py" \
    --seq-name "${SEQ_NAME}" \
    --search-methods "${SEARCH_METHODS[@]}" \
    --decoy-methods "${DECOY_METHODS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
