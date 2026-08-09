set -e
QUERY_FA_PATH=
TARGET_FA_PATH=

while getopts "q:t:" opt; do
    case $opt in
        q) QUERY_FA_PATH="$OPTARG" ;;
        t) TARGET_FA_PATH="$OPTARG" ;;
        *) echo "Usage: bash $0 -q <query_fasta> -t <target_fasta>"; exit 1 ;;
    esac
done

if [[ -z "$QUERY_FA_PATH" || -z "$TARGET_FA_PATH" ]]; then
    echo "Usage: bash $0 -q <query_fasta> -t <target_fasta>"
    exit 1
fi

echo "============init env============"
if [ -z "$BASE_DIR" ]; then
    echo "Error: env not set. Run 'source src/PLMs_cmds/.env' first."
    exit 1
fi
export PATH="$(dirname "$BLASTP_PYTHON_PATH"):$PATH"
echo "Using blastp: $(which blastp)"

QUERY_FA_NAME=$(basename "$QUERY_FA_PATH")
query_basename="${QUERY_FA_NAME%.fa}"
target_basename="$(basename "$TARGET_FA_PATH" .fa)"

OUTPUT_DIR="$TEMP_DIR/result_blastp_${query_basename}_${target_basename}_blosum62_Q11R1"
mkdir -p "$OUTPUT_DIR"

NDESC=$(grep -c "^>" "$TARGET_FA_PATH")
echo "Reporting up to $NDESC target descriptions per query."

BLASTP_TARGET_DB_DIR="$BASE_DIR/data/db/db_${target_basename}_blastp"
BLASTP_TARGET_DB_PATH="${BLASTP_TARGET_DB_DIR}/${target_basename}_db"
if [ ! -d "$BLASTP_TARGET_DB_DIR" ]; then
    echo "============makeblastdb from $TARGET_FA_PATH============"
    mkdir -p "$BLASTP_TARGET_DB_DIR"
    makeblastdb -in "$TARGET_FA_PATH" -dbtype prot -out "$BLASTP_TARGET_DB_PATH" \
        || { echo "makeblastdb failed"; exit 1; }
else
    echo "Database $BLASTP_TARGET_DB_DIR already exists. Skipping makeblastdb."
fi

echo "============blastp query============"
blastp \
    -num_alignments 0 \
    -num_descriptions "$NDESC" \
    -gapopen 11 \
    -gapextend 1 \
    -evalue 100000000000 \
    -matrix BLOSUM62 \
    -comp_based_stats -2 \
    -max_hsps 1 \
    -db "$BLASTP_TARGET_DB_PATH" \
    -query "$QUERY_FA_PATH" \
    > "$OUTPUT_DIR/${query_basename}.out" \
    || { echo "blastp failed"; exit 1; }

echo "============parse (BAGEL class homo_type)============"
OUT_TXT="$DATA_DIR/result_blastp_${query_basename}_${target_basename}.txt"
"$BLASTP_PYTHON_PATH" - "$BASE_DIR" "$OUTPUT_DIR/${query_basename}.out" "$OUT_TXT" \
    "$DATA_DIR/bagel_class.txt" "$NDESC" <<'PYEOF'
import sys

import pandas as pd

base_dir, out_file, dst, class_file, max_hits = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
)
sys.path.insert(0, f"{base_dir}/src/utils")
from parser import parse_blast_data

cls = {}
with open(class_file) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            cls[p[0]] = p[1]


def origin(qid):
    return qid[:-5] if qid.endswith("_shuf") else qid


rows = []
for qid, tid, score, ev in parse_blast_data(out_file):
    cq, ct = cls.get(origin(qid)), cls.get(tid)
    ht = 1 if (cq is not None and cq == ct) else -1
    rows.append((qid, tid, score, ht, ev))

df = pd.DataFrame(rows, columns=["qid", "tid", "score", "homo_type", "evalue"])
df["rank"] = df.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
df = df[df["rank"] <= max_hits]
df = df.sort_values(["qid", "rank"])[["qid", "tid", "score", "homo_type", "rank", "evalue"]]
df.to_csv(dst, sep="\t", index=False)
print(f"[OK] parsed {len(df)} rows, {df['qid'].nunique()} queries -> {dst}")
PYEOF

echo "============done: $OUT_TXT============"
