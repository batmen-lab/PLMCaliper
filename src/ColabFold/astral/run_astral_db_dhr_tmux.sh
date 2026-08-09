#!/usr/bin/env bash
# Launch in tmux:
#   tmux new-session -d -s af2_dhr "bash src/ColabFold/astral/run_astral_db_dhr_tmux.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
source src/PLMs_cmds/.env
"$CALIB_FDR_PYTHON" src/ColabFold/common/parallel_fold.py --exp astral_db --gpus 5,6,7 \
  --method dhr_postprocess --jack-iters 1 \
  >> results/ColabFold/astral_db/dhr_postprocess/iter1/dispatch.log 2>&1
