from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ColabFold" / "astral_db"
PLOT_DATA_DIR = PROJECT_ROOT / "data" / "plot_data" / "ColabFold" / "astral_db"

# Single-query demo (hard case from the DHR paper, full ASTRAL40)
QUERY_ID = "d1w0ha_"
SEQ_NAME = "astral"
SEARCH_METHOD = "dhr_postprocess"
DECOY_METHOD = "extended_shuf"
DECOY_SUFFIX = "_shuf"
WEIGHT_METHOD = "AdaptiveBell"

# Conditions: 9 eFDR thresholds + 1 vanilla = 10 structures per query
EFDR_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 .. 0.9
VANILLA_TAG = "vanilla"  # all search hits, no FDR filter

REP_ID = 0

FASTA_PATH = DATA_DIR / f"{SEQ_NAME}.fa"
QJACKHMMER_BIN = PROJECT_ROOT / "libs/Dense-Homolog-Retrieval-main/bin/qjackhmmer"

# DHR paper JackHMMER settings (Methods)
JACKHMMER_ITERS = 3
JACKHMMER_CPU = 8
JACKHMMER_INC_E = "1e-3"
JACKHMMER_E = "1e-3"
JACKHMMER_F1 = "0.0005"
JACKHMMER_F2 = "5e-05"
JACKHMMER_F3 = "5e-07"

# ColabFold: one structure per condition; fixed across conditions for fair timing
COLABFOLD_NUM_MODELS = 1
COLABFOLD_NUM_RECYCLE = 3

EXP_DIR = RESULTS_DIR / QUERY_ID.rstrip("_")
DOWNSTREAM_DIR = EXP_DIR / "downstream"
ASTRAL_BATCH_DIR = RESULTS_DIR / SEARCH_METHOD / "iter1"

CORE_DIR = PROJECT_ROOT / "src" / "core"
