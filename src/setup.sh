#!/usr/bin/env bash
# Write src/PLM_searching_cmds/.env for this machine.
#
#   bash src/setup.sh                 detect everything and write .env
#   bash src/setup.sh --create-envs   conda-create the four envs first, then detect
#   bash src/setup.sh --print         show what would be written, change nothing
#   bash src/setup.sh --gpus 0,1,2,3  restrict to these GPUs (default: every one detected)
#
# Repo paths are derived from this script's location, so they are never wrong.
# Interpreters, GPUs and ColabFold are detected; anything missing is reported with
# the command that provides it, and .env is still written with the rest filled in.
set -uo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SRC_ROOT/.." && pwd)"
ENV_FILE="$BASE_DIR/src/PLM_searching_cmds/.env"
ENV_YML_DIR="$BASE_DIR/env"
CREATE=0; PRINT_ONLY=0; GPU_OVERRIDE=""
while [ $# -gt 0 ]; do
  a="$1"
  case "$a" in
    --create-envs) CREATE=1 ;;
    --print) PRINT_ONLY=1 ;;
    --gpus) shift; GPU_OVERRIDE="${1:?--gpus needs a list, e.g. 0,1,2,3}" ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
  shift
done

missing=()
note() { printf '  %-22s %s\n' "$1" "$2"; }
need() { missing+=("$1"); }

# ---- conda -----------------------------------------------------------------
CONDA_BIN="$(command -v conda 2>/dev/null)"
[ -z "$CONDA_BIN" ] && for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
                                "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
  [ -x "$c" ] && { CONDA_BIN="$c"; break; }
done
if [ -z "$CONDA_BIN" ]; then
  echo "[setup] conda not found. Install miniforge, then re-run:" >&2
  echo "        https://github.com/conda-forge/miniforge" >&2
  exit 1
fi
CONDA_ROOT="$("$CONDA_BIN" info --base 2>/dev/null)"
echo "[setup] conda: $CONDA_BIN  (envs under $CONDA_ROOT/envs)"

# ---- the four environments -------------------------------------------------
# Each retriever needs its own: their dependencies genuinely conflict.
ENVS=(calib_gam plmfooler dhr_env tmvec)

if [ "$CREATE" = 1 ]; then
  for e in "${ENVS[@]}"; do
    if [ -x "$CONDA_ROOT/envs/$e/bin/python" ]; then
      echo "[setup] env '$e' already exists, skipping"
    elif [ -f "$ENV_YML_DIR/$e.yml" ]; then
      echo "[setup] creating env '$e' from env/$e.yml ..."
      "$CONDA_BIN" env create -f "$ENV_YML_DIR/$e.yml" -n "$e" || \
        echo "[setup] WARNING: env '$e' failed to build" >&2
    else
      echo "[setup] WARNING: env/$e.yml missing, cannot create '$e'" >&2
    fi
  done
fi

py_of() {  # $1 = env name -> interpreter path, or empty
  local p="$CONDA_ROOT/envs/$1/bin/python"
  [ -x "$p" ] && echo "$p"
}
echo "[setup] interpreters"
declare -A PY
for e in "${ENVS[@]}"; do
  PY[$e]="$(py_of "$e")"
  if [ -n "${PY[$e]}" ]; then note "$e" "${PY[$e]}"
  else note "$e" "MISSING"; need "conda env create -f env/$e.yml -n $e"; fi
done

# ---- GPUs ------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
else
  N_GPU=0
fi
if [ -n "$GPU_OVERRIDE" ]; then
  GPU_LIST="$GPU_OVERRIDE"
  echo "[setup] GPUs: $N_GPU visible, restricted to $GPU_LIST by --gpus"
elif [ "$N_GPU" -gt 0 ]; then
  GPU_LIST=$(seq -s, 0 $((N_GPU - 1)))
  echo "[setup] GPUs: $N_GPU visible -> $GPU_LIST"
else
  GPU_LIST=""
  echo "[setup] GPUs: none detected (CPU-only; the PLM searches need a GPU)"
fi
# keeping a GPU free is a deliberate choice, so say so rather than overwrite silently
if [ -f "$ENV_FILE" ]; then
  PREV=$(grep -oE 'DHR_GPU_DEVICES="[^"]*"' "$ENV_FILE" 2>/dev/null | head -1 | cut -d'"' -f2)
  if [ -n "$PREV" ] && [ "$PREV" != "$GPU_LIST" ]; then
    echo "[setup] NOTE: existing .env used GPUs $PREV; writing $GPU_LIST."
    echo "              Pass --gpus $PREV to keep the old set."
  fi
fi
# one worker per GPU; BLASTp is CPU-only so it scales on cores instead
N_PAR=$(echo "$GPU_LIST" | awk -F, '{print (NF && $1!="") ? NF : 1}')
N_CPU=$(nproc 2>/dev/null || echo 4)
BLASTP_PAR=$(( N_CPU > 10 ? 10 : N_CPU ))

# ---- ColabFold (installed outside the repo) --------------------------------
CF_BATCH="$(command -v colabfold_batch 2>/dev/null)"
[ -z "$CF_BATCH" ] && for c in "$HOME/localcolabfold/colabfold-conda/bin/colabfold_batch" \
                               "$CONDA_ROOT/envs/colabfold/bin/colabfold_batch"; do
  [ -x "$c" ] && { CF_BATCH="$c"; break; }
done
if [ -n "$CF_BATCH" ]; then
  CF_BIN="$(dirname "$CF_BATCH")"
  echo "[setup] colabfold: $CF_BATCH"
else
  CF_BIN=""; CF_BATCH=""
  echo "[setup] colabfold: MISSING (only needed for the structure steps)"
  need "install localcolabfold: https://github.com/YoshitakaMo/localcolabfold"
fi

# ---- vendored method sources ----------------------------------------------
echo "[setup] vendored sources under libs/"
for d in Dense-Homolog-Retrieval-main PLMSearch-main tm-vec-master; do
  if [ -d "$BASE_DIR/libs/$d" ]; then note "$d" "ok"
  else note "$d" "MISSING"; need "run bash src/PLM_searching_cmds/setup_retrieval_libs.sh"; fi
done
QJACK="$BASE_DIR/libs/Dense-Homolog-Retrieval-main/bin/qjackhmmer"
[ -x "$QJACK" ] || need "qjackhmmer not executable at libs/.../bin/qjackhmmer"

# ---- write .env ------------------------------------------------------------
render() {
cat <<EOF
# Generated by setup.sh on $(date -Iseconds) -- re-run setup.sh to refresh.
# Repo paths are derived; interpreters, GPUs and ColabFold were detected.

export BASE_DIR="$BASE_DIR"
export DATA_DIR="\$BASE_DIR/data"
export TEMP_DIR="\$DATA_DIR/temp"

## calibration / FDR / evaluation
export CALIB_FDR_PYTHON="${PY[calib_gam]}"

## BLASTp (CPU only)
export BLASTP_PYTHON_PATH="${PY[plmfooler]}"
export BLASTP_NUM_PARALLEL=$BLASTP_PAR

## PLMsearch
export PLMSEARCH_PYTHON_PATH="${PY[plmfooler]}"
export PLMSEARCH_SRC_DIR="\$BASE_DIR/libs/PLMSearch-main"
export PLMSEARCH_GPU_DEVICES="$GPU_LIST"
export PLMSEARCH_NUM_PARALLEL=$N_PAR

## DHR (Dense Homolog Retrieval)
export DHR_PYTHON_PATH="${PY[dhr_env]}"
export DHR_SRC_DIR="\$BASE_DIR/libs/Dense-Homolog-Retrieval-main"
export DHR_GPU_DEVICES="$GPU_LIST"
export DHR_NUM_PARALLEL=$N_PAR

## TM-Vec / DeepBLAST
export TMVEC_PYTHON_PATH="${PY[tmvec]}"
export TMVEC_SRC_DIR="\$BASE_DIR/libs/tm-vec-master"
export TMVEC_GPU_DEVICES="$GPU_LIST"
export TMVEC_NUM_PARALLEL=$N_PAR

## pipeline driver
export EVALPLMS_PYTHON_PATH="${PY[plmfooler]}"

## external binaries
export QJACKHMMER_BIN="\$DHR_SRC_DIR/bin/qjackhmmer"
export COLABFOLD_BIN="$CF_BIN"
export COLABFOLD_BATCH="$CF_BATCH"
EOF
}

if [ "$PRINT_ONLY" = 1 ]; then
  echo; render; exit 0
fi
mkdir -p "$(dirname "$ENV_FILE")"
[ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$ENV_FILE.bak" && echo "[setup] backed up previous .env -> .env.bak"
render > "$ENV_FILE"
echo "[setup] wrote $ENV_FILE"

# ---- report ----------------------------------------------------------------
echo
if [ ${#missing[@]} -eq 0 ]; then
  echo "[setup] everything found. Next:"
  echo "        source src/PLM_searching_cmds/.env"
else
  echo "[setup] .env written, but ${#missing[@]} thing(s) still needed:"
  for m in "${missing[@]}"; do echo "        - $m"; done
  echo "        Re-run 'bash src/setup.sh' once they are in place."
fi
