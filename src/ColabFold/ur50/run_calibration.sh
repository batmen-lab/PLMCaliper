#!/usr/bin/env bash

# Usage:  bash src/ColabFold/ur50/run_calibration.sh [dhr_postprocess]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source src/PLMs_cmds/.env

METHOD="${1:-dhr_postprocess}"

# All methods (dhr_postprocess / tmvec / plm) use the same GAM + AdaptiveBell
# calibration (src/core), calibrated identically. Feasible at top-1M
# (~32M rows/table; ~30-40GB RAM). Do NOT run on full-UR50 output (would OOM).
"${CALIB_FDR_PYTHON}" src/core/run_calibration_fdr_pipeline.py \
  --search-methods "${METHOD}" \
  --decoy-methods extended_shuf \
  --query-name astral4f \
  --target-name ur50 \
  --weight-method AdaptiveBell \
  --skip-fdr \
  --keep-intermediates \
  --force-target-noisy \
  --data-dir data

echo "[DONE] calibration -> data/result_${METHOD}_astral4f_target_noisy_ur50.txt"
echo "                      data/result_${METHOD}_astral4f_extended_shuf_calibrated_gam_AdaptiveBell_ur50.txt"
