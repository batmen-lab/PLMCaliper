from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ur50_exp"

# ---- Search database: switched from a UR50 subset to FULL UniRef50 (38,794,121 seqs).
UR50_DIR = DATA_DIR / "uniref50"
UR50_FASTA_GZ = UR50_DIR / "uniref50.fasta.gz"    
UR50_FASTA = UR50_DIR / "uniref50.fasta"     
UR50_TSV = UR50_DIR / "uniref50.tsv"      
UR50_NAME = "ur50"                
# (no subsampling for UR50 — the full set is used)
SUBSAMPLE_SEED = 42
SUBSAMPLE_PROB = 0.368

# ---- DHR index (lib convention: indexes live under the DHR lib data dir) ----
DHR_SRC_DIR = PROJECT_ROOT / "libs" / "Dense-Homolog-Retrieval-main"
DHR_CKPT = DHR_SRC_DIR / "dhr2_ckpt"
DHR_DB_DIR = DHR_SRC_DIR / "data" / "db_ur50"       
DHR_AGG_DIR = DHR_DB_DIR / "agg"
DHR_GPUS = "0,1,2,3,4,5"                        


# ---- Queries: CASP13 domain sequences ----
CASP13_DIR = DATA_DIR / "casp13"
QUERY_NAME = "astral4f"
QUERY_FASTA = DATA_DIR / "astral4f_query.fa"
QUERY_TSV = DATA_DIR / "astral4f_query.tsv"
NATIVE_PDB_DIR = DATA_DIR / "pdbstyle-2.08"    
DECOY_METHOD = "extended_shuf"
DECOY_SUFFIX = "_shuf"
DECOY_FASTA = DATA_DIR / f"{QUERY_NAME}_{DECOY_METHOD}.fa"
DECOY_TSV = DATA_DIR / f"{QUERY_NAME}_{DECOY_METHOD}.tsv"

# ---- methods / FDR ----
SEARCH_METHODS = ["dhr_postprocess"]                   # also plm / tmvec (same downstream)
WEIGHT_METHOD = "AdaptiveBell"
EFDR_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 .. 0.9
VANILLA_TAG = "vanilla"
# Retrieval depth is not set here: it is a search-time knob read by
# data_build/dhr/search_dhr_astral4f_ur50.sh from $DHR_RETRIEVAL_N (default 1000000).

# ---- shared downstream (reuse src/ColabFold/astral pipeline for MSA/ColabFold/pLDDT/stats) ----
DHR_EXP_DIR = PROJECT_ROOT / "src" / "ColabFold" / "astral"
CORE_DIR = PROJECT_ROOT / "src" / "core"
QJACKHMMER_BIN = DHR_SRC_DIR / "bin" / "qjackhmmer"


def method_results_dir(method: str) -> Path:
    return RESULTS_DIR / method


def method_runs_dir(method: str) -> Path:
    return method_results_dir(method) / "runs"


def method_tmp_dir(method: str) -> Path:
    return method_results_dir(method) / "_tmp"


def method_summary_path(method: str) -> Path:
    return method_results_dir(method) / "plddt_time.tsv"


def method_ttest_path(method: str) -> Path:
    return method_results_dir(method) / "paired_ttest.tsv"


def method_boxplot_path(method: str) -> Path:
    return method_results_dir(method) / "plddt_time_vs_efdr_boxplot.pdf"


def efdr_tag(efdr: float) -> str:
    return f"efdr_{efdr:.2f}".replace(".", "p")
