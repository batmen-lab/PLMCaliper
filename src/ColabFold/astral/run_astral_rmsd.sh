#!/usr/bin/env bash
# Usage:
#   bash src/ColabFold/astral/run_astral_rmsd.sh                       # dhr_postprocess, 4-family list
#   METHOD=plm QUERY_LIST=data/querylist_a.4.1.1.txt bash src/ColabFold/astral/run_astral_rmsd.sh
#   MAXQ=10 bash src/ColabFold/astral/run_astral_rmsd.sh               # first 10 queries (smoke)
#   SYNC_PLOT_DATA=1 bash src/ColabFold/astral/run_astral_rmsd.sh       # also update plot_data table
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source src/PLMs_cmds/.env
PY="${EVALPLMS_PYTHON_PATH}"

METHOD="${METHOD:-dhr_postprocess}"
QUERY_LIST="${QUERY_LIST:-data/querylist_4families.txt}"
JACK_ITERS="${JACK_ITERS:-1}"
OUTDIR="results/ColabFold/astral_db/${METHOD}/iter${JACK_ITERS}"
OUT="${OUTDIR}/metrics.tsv"
mkdir -p "${OUTDIR}"

MAXQ_ARG=()
[ -n "${MAXQ:-}" ] && MAXQ_ARG=(--max-queries "${MAXQ}")
SYNC_ARG=()
[ "${SYNC_PLOT_DATA:-0}" = "1" ] && SYNC_ARG=(--sync-plot-data)

echo "[compute] vanilla full-astral vs FDR-subset -> metrics [GPU ${CUDA_VISIBLE_DEVICES:-0}]"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PY}" src/ColabFold/astral/pipeline_astral.py \
  --method "${METHOD}" --query-list "${QUERY_LIST}" --jack-iters "${JACK_ITERS}" \
  --out "${OUT}" "${MAXQ_ARG[@]}" "${SYNC_ARG[@]}" \
  || echo "[WARN] pipeline nonzero (partial saved; re-run to resume)"

echo "[DONE] -> ${OUT}"
