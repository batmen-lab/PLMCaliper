#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"; cd "$ROOT"

source src/PLMs_cmds/.env
DHR_PY="${DHR_PYTHON_PATH}"
AGG="${DHR_SRC_DIR}/data/db_ur50/agg"
N="${DHR_RETRIEVAL_N:-1000000}"
OUT="data/uniref50/dhr_search_astral4f"; mkdir -p "$OUT"
[ -e "${AGG}/index-ebd.index" ] || { echo "[ERROR] DHR UR50 index missing: ${AGG}"; exit 1; }

echo "[1/3] build astral4f query + decoy tsv (83 queries)"
"${CALIB_FDR_PYTHON}" - <<'PY'
import random
random.seed(43)
def fa(p):
    d={};n=None;s=[]
    for line in open(p):
        if line[0]=='>':
            if n:d[n]=''.join(s)
            n=line[1:].split()[0];s=[]
        else:s.append(line.strip())
    if n:d[n]=''.join(s)
    return d
seqs=fa('data/astral.fa')
qids=[l.strip() for l in open('data/querylist_4families.txt') if l.strip() and not l.startswith('#')]
nq=0
with open('data/astral4f_query.tsv','w') as q, open('data/astral4f_extended_shuf.tsv','w') as dec:
    for qid in qids:
        if qid not in seqs: continue
        s=seqs[qid]; q.write(f"{qid}\t{s}\n")
        sh=list(s); random.shuffle(sh); dec.write(f"{qid}_shuf\t{s}{''.join(sh)}\n"); nq+=1
print(f"  built {nq} query + decoy")
PY

echo "[2/3] DHR retrieval (top-${N}) vs UR50 index"
run_one(){ local qtsv="$1" tag="$2"
  if [ -s "$OUT/$tag.txt" ]; then echo "  [skip] $tag exists ($(wc -l < "$OUT/$tag.txt") rows)"; return; fi
  echo "  $tag: $qtsv x UR50 (n=$N)"
  ( cd "$DHR_SRC_DIR" && "$DHR_PY" ./do_retrieval.py -i "$ROOT/$qtsv" -d "$AGG" -o "$ROOT/$OUT/$tag.txt" -n "$N" )
  echo "    -> $OUT/$tag.txt ($(wc -l < "$OUT/$tag.txt") rows)"
}
run_one data/astral4f_query.tsv          astral4f
run_one data/astral4f_extended_shuf.tsv  astral4f_extended_shuf

echo "[3/3] convert to dhr_postprocess tsv (score = 150 - distance, descending similarity)"
conv(){ awk -F',' 'BEGIN{OFS="\t"; print "qid","tid","score","homo_type","rank"}
  $1!=lq{lq=$1; r=0} {r++; print $1,$2,150-$3,-1,r}' "$1" > "$2"; echo "  wrote $2"; }
conv "$OUT/astral4f.txt"               data/result_dhr_postprocess_astral4f_ur50.txt
conv "$OUT/astral4f_extended_shuf.txt" data/result_dhr_postprocess_astral4f_extended_shuf_ur50.txt
echo "[DONE] astral4f x UR50 retrieval ready"
