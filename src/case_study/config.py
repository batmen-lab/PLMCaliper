import os
import sys
from pathlib import Path

# repo root: .../src/case_study/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORE_DIR = PROJECT_ROOT / "src" / "core"
# All case-study outputs (cache, step1, step2)
CASE_STUDY_DIR = PROJECT_ROOT / "results_final" / "case_study"
RESULTS_DIR = CASE_STUDY_DIR
RESULTS_FINAL_DIR = CASE_STUDY_DIR

# Default dataset: full ASTRAL self-search (sequences from data/astral.fa)
SEARCH_METHOD = "plm"
QUERY_NAME = "astral"
TARGET_NAME = "astral"
FASTA_NAME = "astral"
DECOY_METHOD = "extended_shuf"
WEIGHT_METHOD = "AdaptiveBell"
TAU = 0.2
REP_ID = 0

# Pair tables larger than this (GB) are processed query-by-query in streaming mode
STREAMING_THRESHOLD_GB = 1.0

METHOD2TYPE = {
    "shuf": "shuf",
    "rev": "rev",
    "mkv1": "mkv1",
    "mkv2": "mkv2",
    "dplm": "dplm",
    "PGmsk25pct": "PGmsk25pct",
    "extended_shuf": "shuf",
    "extended_rev": "rev",
    "extended_dup": "dup",
    "extended_then_shuf": "shuf",
    "extended_shufpartly40": "shufpartly",
    "extended_mkv1": "mkv",
    "extended_mkv2": "mkv",
}

# Selection criteria
Q_REFERENCE = 0.10
MAX_SEQ_IDENTITY = 0.30
TOP_N_CANDIDATES = 5
MAX_NONHITS_PER_QUERY = 100  # per (qid, rep_id); FDR hits have no cap

# SCOPe40 pdbstyle (official domain structures for RCSB export)
PDBSTYLE_DIR = DATA_DIR / "pdbstyle-2.08"
PDBSTYLE_DIR_ALT = DATA_DIR / "db" / "astral40" / "pdbstyle-2.08"
# Interpreter used to spawn helper processes; defaults to the running one.
PYTHON_BIN = Path(os.environ.get("PLMCALIPER_PYTHON", sys.executable))


def decoy_suffix(decoy_method: str) -> str:
    if decoy_method not in METHOD2TYPE:
        raise ValueError(f"Unknown decoy method '{decoy_method}'")
    return f"_{METHOD2TYPE[decoy_method]}"


def path_target_noisy(search_method: str, query_name: str, target_name: str) -> Path:
    return DATA_DIR / f"result_{search_method}_{query_name}_target_noisy_{target_name}.txt"


def path_calibrated_decoy(
    search_method: str,
    query_name: str,
    decoy_method: str,
    target_name: str,
) -> Path:
    return DATA_DIR / (
        f"result_{search_method}_{query_name}_{decoy_method}_calibrated_gam_"
        f"{WEIGHT_METHOD}_{target_name}.txt"
    )


def fasta_path(name: str | None = None) -> Path:
    return DATA_DIR / f"{name or FASTA_NAME}.fa"


def pdbstyle_path(protein_id: str, pdbstyle_dir: Path | None = None) -> Path:
    base = pdbstyle_dir or find_pdbstyle_dir() or PDBSTYLE_DIR
    origin_id = protein_id
    subdir = origin_id[2:4] if len(origin_id) >= 4 else origin_id[:2]
    return base / subdir / f"{origin_id}.ent"


def find_pdbstyle_dir() -> Path | None:
    for candidate in (PDBSTYLE_DIR, PDBSTYLE_DIR_ALT):
        if candidate.exists():
            return candidate
    return None


def results_final_tag_dir(
    search_method: str = SEARCH_METHOD,
    query_name: str = QUERY_NAME,
    decoy_method: str = DECOY_METHOD,
    q: float | None = None,
) -> Path:
    return RESULTS_FINAL_DIR / run_tag(search_method, query_name, decoy_method, q=q)


def cache_dir(
    search_method: str,
    query_name: str,
    decoy_method: str,
    q: float,
) -> Path:
    return RESULTS_DIR / "cache" / f"{search_method}_{query_name}_{decoy_method}_q{q:.2f}"


def run_tag(
    search_method: str = SEARCH_METHOD,
    query_name: str = QUERY_NAME,
    decoy_method: str = DECOY_METHOD,
    q: float | None = None,
) -> str:
    q_val = q if q is not None else Q_REFERENCE
    return f"{search_method}_{query_name}_{decoy_method}_q{q_val:.2f}"
