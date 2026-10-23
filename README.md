[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22436908.svg)](https://doi.org/10.5281/zenodo.22436908)
# PLM-Caliper

PLM-Caliper is a model-agnostic framework for false discovery rate (FDR) control in protein search with protein language models (PLMs). It uses query-derived decoys together with score calibration to estimate false discoveries and identify score thresholds corresponding to user-specified FDR levels.

PLM-Caliper operates on the output scores of existing protein search methods and does not require retraining or fine-tuning the underlying search models.

This repository contains:

- the main PLM-Caliper implementation;
- scripts for generating protein-retrieval scores with different search methods;
- utilities for decoy generation and score processing;
- analysis code for reproducing the experiments reported in the paper.

---

## Installation

### Requirements

PLM-Caliper is implemented in Python. Additional software is required only for reproducing experiments involving specific protein search or structure-prediction pipelines.

### Environment setup

Clone the repository:

```bash
git clone git@github.com:batmen-lab/PLMCaliper.git
```

Create and activate the environment:

```bash
conda create -n plmcaliper python=3.11 -y
conda activate plmcaliper
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Data

Download the required dataset (links below) and arrange the project as follows:

```text
PLMCaliper/
├── data/
│   ├── astral.fa             # ASTRAL sequence database
│   ├── astral.tsv
│   ├── astral4f_query.fa     # 4-family query subset
│   ├── astral4f_query.tsv
│   ├── class_all.fa          # BAGEL4 classified bacteriocins
│   ├── class_all.tsv
│   ├── putative.fa           # BAGEL4 putative bacteriocins
│   └── putative.tsv
│
├── libs/
│
├── results/
│
├── src/
│   ├── PLMCaliper.py         # PLMCaliper class
│   └── run_PLMCaliper.py
│
└── tutorials/
    ├── ablation/
    ├── bagel/
    ├── ColabFold/
    ├── conformal_benchmark/
    ├── PLM_searching_cmds/
    ├── utils/
    └── setup.sh
```

- **Reference databases**, **library dependencies (PLM methods)**: — download from [zenodo](https://zenodo.org/records/22436908)
- **PDB structures**: — download from [pdbstyle-2.08](https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz)

---

## Quick Start

PLM-Caliper generates decoy queries and calibrates search scores; the target and decoy score tables are produced by an upstream retrieval method.

### 1. Make Decoy Queries

The input consists of query protein sequences, and the output is the corresponding `extended-Markov` decoy sequences.

```bash
python src/run_PLMCaliper.py make-decoy \
  --fasta data/astral.fa \
  --out data/astral_extended_mkv2.fa \
  --design extended_mkv \
  --markov-order 2
```

### 2. Run Protein Retrieval

Search the original queries and the decoy queries against the same target database using the same method. The modified `libs` used to retain complete retrieval hits are available at [zenodo](https://zenodo.org/records/22436908).

Then create the search environments (`plmsearch`, `dhr`, and `tmvec`) and load the paths:

```bash
bash tutorials/setup.sh --create-envs
source tutorials/PLM_searching_cmds/.env
```

Run retrieval for the original queries:

```bash
bash tutorials/PLM_searching_cmds/run_search.sh \
  -m tmvec \
  -q data/astral.fa \
  -t data/astral.fa
```

Run retrieval for the decoy queries:

```bash
bash tutorials/PLM_searching_cmds/run_search.sh \
  -m tmvec \
  -q data/astral_extended_mkv2.fa \
  -t data/astral.fa
```

Use `-m plm`, `-m dhr`, or `-m blastp` to run the other retrieval methods. Note that `.fa` files must be converted to `.tsv` using `tutorials/utils/fa2tsv.py` before running DHR.

The parsed score tables are written as `data/result_<method>_<query>_<target>.txt`.

For custom score tables, provide a delimited file with a header and at least these columns:

```text
qid	tid	score
```

### 3. Run Calibration

> [!IMPORTANT]
> For `dhr` and `blastp`, score preprocessing is required: DHR scores must be converted to a larger-is-better scale, and missing BLASTp hits must be filled in.
>
> ```bash
> # DHR: flip distances into similarities
> python tutorials/utils/process_searching_score.py dhr-postprocess \
>   --query astral \
>   --db astral \
>   --decoy extended_mkv2
>
> # BLASTp: fill the query-target pairs BLASTp never reported
> python tutorials/utils/process_searching_score.py blastp-densify \
>   --real data/result_blastp_astral_astral.txt \
>   --decoy data/result_blastp_astral_extended_mkv2_astral.txt \
>   --fasta data/astral.fa \
>   --decoy-suffix _mkv \
>   --out-real data/result_blastp_postprocessed_astral_astral.txt \
>   --out-decoy data/result_blastp_postprocessed_astral_extended_mkv2_astral.txt
> ```

The calibration module requires the following inputs:
- **Target scores:** scores from the original query sequence.
- **Decoy scores:** scores from the decoy query sequence generated by `make-decoy`.
- **Target FDR:** the desired false discovery rate level, passed as `--target-fdr`.


```bash
mkdir -p results
python src/run_PLMCaliper.py calibrate \
  --real data/result_tmvec_astral_astral.txt \
  --decoy data/result_tmvec_astral_extended_mkv2_astral.txt \
  --out results/plmcaliper_cutoffs.tsv \
  --target-fdr 0.50 \
  --hits-out results/plmcaliper_hits.tsv
```

Main output `plmcaliper_cutoffs.tsv`:

```text
qid	rep_id	target_fdr	cutoff_score	est_fdp	n_selected
```

Optional hits output `plmcaliper_hits.tsv`:

```text
qid	tid	rep_id	score	est_fdp
```

`score` is the retrieval score. `est_fdp` is PLM-Caliper's estimated false discovery proportion (FDP).

---

## Reproducing the Paper Results

The experimental analysis code is organized under:

```text
tutorials/
```

Each subdirectory corresponds to an experiment reported in the paper. For step-by-step reproduction instructions and the expected results, see the README and Jupyter notebook in each subdirectory.

To save computation time, precomputed PLM-based search scores and calibration files are available for download from [Hugging Face](https://huggingface.co/datasets/yifanyang993/PLMCaliper).
