#!/usr/bin/env bash

# Usage:  bash src/ColabFold/ur50/reproduce_ur50.sh <method> [GPU]
#         method = dhr_postprocess | tmvec | plm     GPU defaults to 0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${ROOT}"
# shellcheck disable=SC1091
source src/PLMs_cmds/.env
PY="${EVALPLMS_PYTHON_PATH}"
APY="${CALIB_FDR_PYTHON}"

METHOD="${1:?usage: reproduce_ur50.sh <dhr_postprocess|tmvec|plm> [GPU]}"
GPU="${2:-0}"
case "${METHOD}" in dhr_postprocess|tmvec|plm) ;; *)
  echo "[ERROR] unknown method '${METHOD}' (expected dhr_postprocess|tmvec|plm)"; exit 1 ;; esac

NOISY="data/result_${METHOD}_astral4f_target_noisy_ur50.txt"
CALIB="data/result_${METHOD}_astral4f_extended_shuf_calibrated_gam_AdaptiveBell_ur50.txt"
OUTDIR="results/ur50_exp/${METHOD}"
OUT="${OUTDIR}/plddt_time.tsv"
TTEST="${OUTDIR}/paired_ttest.tsv"
PLOT="${OUTDIR}/plddt_time_vs_efdr_boxplot.pdf"
mkdir -p "${OUTDIR}"

# ---- input contract: tables live in data/search_data/; symlink them into data/
#      so calibration + pipeline_ur50 (which read from data/) find them ----
miss=0
for base in "result_${METHOD}_astral4f_ur50.txt" \
            "result_${METHOD}_astral4f_extended_shuf_ur50.txt"; do
  if [ -s "${ROOT}/data/search_data/${base}" ]; then
    ln -sf "search_data/${base}" "${ROOT}/data/${base}"
  else
    echo "[MISSING INPUT] data/search_data/${base}"; miss=1
  fi
done
if [ "${miss}" -ne 0 ]; then
  echo "  provide the search tables, or regenerate them:"
  echo "    bash src/ColabFold/ur50/build_db.sh ${METHOD} ; bash src/ColabFold/ur50/search.sh ${METHOD} ${GPU}"
  exit 1
fi

echo "========== 1. calibration / FDR control (${METHOD}) =========="
if [ -s "${NOISY}" ] && [ -s "${CALIB}" ]; then
  echo "[skip] calibrated files already present"
else
  bash src/ColabFold/ur50/run_calibration.sh "${METHOD}"
fi

echo "========== 2. eFDR-thresholded MSA build + fold + eval [GPU ${GPU}] =========="
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" src/ColabFold/ur50/pipeline_ur50.py \
  --method "${METHOD}" --out "${OUT}" \
  || echo "[WARN] pipeline exited nonzero (partial saved; re-run to resume)"

echo "========== 3. paired t-test + boxplot =========="
if [ -s "${OUT}" ]; then
  "${APY}" src/ColabFold/astral/stats.py --summary "${OUT}" --out "${TTEST}" \
    || echo "[WARN] stats failed"
  "${APY}" src/plot/ColabFold/plot_astral_plddt_time.py --summary "${OUT}" \
    --out "${PLOT}" --ttest-out "${TTEST}" \
    --title "UR50 (${METHOD}): structure quality & time vs eFDR (vs vanilla)" \
    || echo "[WARN] plot failed"
fi
echo "[DONE] ${METHOD} -> ${OUT}"
