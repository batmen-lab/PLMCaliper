#!/usr/bin/env bash

#   tmux new-session -d -s ur50_build "bash src/ColabFold/ur50/run_ur50_build.sh"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
source src/PLMs_cmds/.env
PY="$EVALPLMS_PYTHON_PATH"
METHOD="${1:-dhr_postprocess}"   # retriever token: dhr_postprocess (default) | tmvec | plm
N="${UR50_BUILD_WORKERS:-4}"
BASE=results/ColabFold/ur50_db/$METHOD/iter1
PAR=$BASE/parallel
CAND=$PAR/cand_seqs.tsv
RUNS=$BASE/runs
TMP=$BASE/tmp
mkdir -p "$PAR" "$RUNS" "$TMP"

QLIST=data/querylist_4families.txt
mapfile -t QIDS < <(grep -vE '^\s*#|^\s*$' "$QLIST")
echo "[info] $((${#QIDS[@]})) queries, $N workers, MSA-only build"

# split into N shards (round-robin)
for i in $(seq 0 $((N-1))); do : > "$PAR/qshard_$i.txt"; done
idx=0
for q in "${QIDS[@]}"; do echo "$q" >> "$PAR/qshard_$((idx % N)).txt"; idx=$((idx+1)); done

pids=()
for i in $(seq 0 $((N-1))); do
  [ -s "$PAR/qshard_$i.txt" ] || continue
  echo "  [worker $i] $(wc -l < "$PAR/qshard_$i.txt") queries -> msa_out_$i.tsv"
  "$PY" src/ColabFold/ur50/pipeline_ur50.py --method "$METHOD" \
    --query-list "$PAR/qshard_$i.txt" --skip-af2 --cand-seqs "$CAND" \
    --out "$PAR/msa_out_$i.tsv" --runs-dir "$RUNS" --tmp-dir "$TMP" \
    > "$PAR/build_$i.log" 2>&1 &
  pids+=("$!")
done
echo "[info] workers: ${pids[*]}"
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done

# merge shards
"$PY" - <<PY
import pandas as pd, glob
fs=sorted(glob.glob("$PAR/msa_out_*.tsv"))
d=pd.concat([pd.read_csv(f,sep="\t") for f in fs if __import__("os").path.getsize(f)>0],ignore_index=True)
d=d.sort_values(["query_id","tag"]).reset_index(drop=True)
d.to_csv("$BASE/msa_metrics.tsv",sep="\t",index=False)
print(f"[merge] {len(d)} rows, {d['query_id'].nunique()} queries -> $BASE/msa_metrics.tsv")
print("[status]", d['msa_status'].value_counts().to_dict())
PY
echo "[DONE] phase-1 MSA build rc=$rc"
