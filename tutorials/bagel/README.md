### 2. BAGEL4 Bacteriocin Analysis

Directory:

```text
tutorials/bagel/
```

This experiment evaluates PLM-Caliper for FDR-guided annotation of putative bacteriocins using the BAGEL4 benchmark.

The analysis applies FDR-controlled protein retrieval to identify candidate bacteriocin homologs while controlling the expected proportion of false discoveries.

First, generate real and decoy similarity scores for the BAGEL dataset using each search method.

```bash
conda activate plmcaliper
source tutorials/PLM_searching_cmds/.env

python src/run_PLMCaliper.py make-decoy \
  --fasta data/putative.fa \
  --out data/putative_extended_mkv2.fa \
  --design extended_mkv \
  --markov-order 2

bash tutorials/PLM_searching_cmds/run_search.sh \
  -m tmvec \
  -q data/putative.fa \
  -t data/class_all.fa

bash tutorials/PLM_searching_cmds/run_search.sh \
  -m tmvec \
  -q data/putative_extended_mkv2.fa \
  -t data/class_all.fa
```

The figures place every sequence on a shared t-SNE map, whose coordinates come
from TM-Vec embeddings. Download the published t-SNE layout into `data/bagel/`:

```bash
python -m pip install -U "huggingface_hub[cli]"

mkdir -p data/bagel
hf download yifanyang993/PLMCaliper --repo-type dataset \
  --include "tsne/*" --local-dir data/bagel
mv data/bagel/tsne/*.npy data/bagel/ && rmdir data/bagel/tsne
rm -rf data/bagel/.cache
```

Run the experiment with:

```bash
bash tutorials/bagel/run_bagel.sh -target_fdr 0.20 -m tmvec
```
