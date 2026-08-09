set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${ROOT}"

source src/PLMs_cmds/.env
DHR_PY="${DHR_PYTHON_PATH:?}"

GPUS="${DHR_GPU_OVERRIDE:-${DHR_GPU_DEVICES:-0,1,2,3,4,5}}"
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NG=${#GPU_ARR[@]}
NSH="${DHR_N_SHARDS:-60}"

TSV="${DHR_ENCODE_TSV:-data/uniref50/uniref50.tsv}"
SHARD_DIR="${DHR_SHARD_DIR:-data/uniref50/shards}"
DHR_DB="${DHR_DB_OUT:-${DHR_SRC_DIR}/data/db_ur50}"
CKPT="${DHR_SRC_DIR}/dhr2_ckpt"

[ -s "${TSV}" ] || { echo "[ERROR] ${TSV} missing"; exit 1; }

# ---- 2. split into NSH contiguous shards (once) ----
mkdir -p "${SHARD_DIR}"
have=$(ls "${SHARD_DIR}"/shard_*.tsv 2>/dev/null | wc -l)
if [ "${have}" -ne "${NSH}" ]; then
  echo "[2] splitting ${TSV} into ${NSH} shards (had ${have})"
  rm -f "${SHARD_DIR}"/shard_*.tsv
  split -n "l/${NSH}" -d -a 2 --additional-suffix=.tsv "${TSV}" "${SHARD_DIR}/shard_"
fi
mapfile -t SHARDS < <(ls "${SHARD_DIR}"/shard_*.tsv | sort)
echo "[info] ${#SHARDS[@]} shards, ${NG} GPUs, encoding in waves"

pt_of() { echo "${DHR_DB}/$(basename "$1" .tsv)/ebd/0/predictions.pt"; }

# ---- 3. encode pending shards in waves of NG ----
encode_one() {  # $1 = shard tsv, $2 = physical gpu
  local shard="$1" g="$2"
  local name; name=$(basename "${shard}" .tsv)
  local outdir="${DHR_DB}/${name}"
  rm -rf "${outdir}/ebd" "${outdir}/.hydra"; mkdir -p "${outdir}"
  ( cd "${DHR_SRC_DIR}" && "${DHR_PY}" ./do_embedding.py \
      trainer.ur50_path="${ROOT}/${shard}" model.ckpt_path="${CKPT}" \
      "trainer.gpus=[0]" "trainer.devices='${g}'" hydra.run.dir="${outdir}" \
      > "${outdir}/encode.log" 2>&1 )
}

done_n=0; todo=()
for s in "${SHARDS[@]}"; do
  if [ -s "$(pt_of "$s")" ]; then done_n=$((done_n+1)); else todo+=("$s"); fi
done
echo "[3] ${done_n}/${#SHARDS[@]} already done; ${#todo[@]} to encode"

idx=0
while [ "${idx}" -lt "${#todo[@]}" ]; do
  pids=(); slot=0
  while [ "${slot}" -lt "${NG}" ] && [ "${idx}" -lt "${#todo[@]}" ]; do
    shard="${todo[$idx]}"; g="${GPU_ARR[$slot]}"
    echo "  [$(date +%H:%M)] GPU ${g} <- $(basename "${shard}")"
    encode_one "${shard}" "${g}" & pids+=("$!")
    idx=$((idx+1)); slot=$((slot+1))
  done
  for p in "${pids[@]}"; do wait "${p}" || echo "  [WARN] a shard exited nonzero"; done
done

# ---- 4. verify all done, then aggregate ----
miss=0
for s in "${SHARDS[@]}"; do [ -s "$(pt_of "$s")" ] || { echo "[INCOMPLETE] $(basename "$s")"; miss=$((miss+1)); }; done
if [ "${miss}" -ne 0 ]; then echo "[STOP] ${miss} shards incomplete; re-run to resume"; exit 1; fi

echo "[4] all ${#SHARDS[@]} shards done -> aggregating"
"${DHR_PY}" "${ROOT}/src/ColabFold/ur50/data_build/shared/agg_shards.py" \
  --shard-tsv-glob "${ROOT}/${SHARD_DIR}/shard_*.tsv" --ebd-base "${DHR_DB}" --out "${DHR_DB}/agg"
echo "[DONE] DHR UR50 index -> ${DHR_DB}/agg/index-ebd.index"
