#!/usr/bin/env bash
# Launch in tmux:
#   tmux new-session -d -s af2_tmvec "bash src/ColabFold/astral/run_astral_db_tmvec_tmux.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
source src/PLMs_cmds/.env
"$CALIB_FDR_PYTHON" src/ColabFold/common/parallel_fold.py --exp astral_db --gpus 0,1 \
  --method tmvec --jack-iters 1 \
  >> results/ColabFold/astral_db/tmvec/iter1/dispatch.log 2>&1
