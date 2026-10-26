### 3. FDR-Guided MSA Construction and Structure Prediction

Directory:

```text
tutorials/ColabFold/
```

This experiment evaluates whether FDR-guided sequence selection can reduce the number of retrieved sequences used for multiple-sequence alignment (MSA) construction while preserving downstream protein structure-prediction quality.

The pipeline includes data preparation, MSA construction, PLM-Caliper calibration, structure prediction, evaluation, and visualization.

The search database is UniRef50, available from the [UniProt FTP site](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz). The release used in the paper holds 38,794,121 sequences. Place it at `data/uniref50/uniref50.fasta`.

Install the dependencies for this experiment:

```bash
conda activate plmcaliper
source tutorials/PLM_searching_cmds/.env
conda run -n tmvec python -m pip install --no-deps -e libs/tm-vec-master deepblast==1.0.2
```

Run the experiment with:

```bash
bash tutorials/ColabFold/run_colabfold.sh -m dhr
```
