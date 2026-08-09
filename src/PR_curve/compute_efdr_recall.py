#!/usr/bin/env python3
import argparse
import csv
import gc
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from collections.abc import Iterator
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results_final" / "pr_curve"
DATA_SUBDIR = "data"

SEQ_NAME = "astral_s3000_seed123"
WEIGHT_METHOD = "AdaptiveBell"

POSITIVE_HOMO_TYPES = (1, 2)
NON_HOMO_TYPE = -1
IGNORE_HOMO_TYPES = (0,)

SEARCH_DECOY_COMBOS = [
    ("plm", "shuf"),
]

Q_LEVELS = np.arange(0.05, 1.00, 0.05)  # 0.05, 0.10, ..., 0.95
Q_LABEL_LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # only these q values get text labels on plots


def q_should_label(q: float, label_levels: np.ndarray | None = None) -> bool:
    levels = label_levels if label_levels is not None else Q_LABEL_LEVELS
    return any(np.isclose(float(q), float(t), atol=1e-6) for t in levels)


def format_q_label(q: float) -> str:
    return f"{float(q):.1f}".rstrip("0").rstrip(".")
SCORE_DESCENDING = True
STREAMING_THRESHOLD_GB = 1.0
STREAM_LOG_EVERY = 500
SHOW_PROGRESS = True

# Cohort binning for grouped plots: [0.1,0.3), [0.3,0.5), ...
COHORT_BIN_START = 0.1
COHORT_BIN_WIDTH = 0.1

METHOD2TYPE = {
    "shuf": "shuf",
    "extended_shuf": "shuf",
    "extended_rev": "rev",
    "extended_dup": "dup",
    "extended_mkv1": "mkv",
    "extended_mkv2": "mkv",
}

LINE_COLORS = [
    "#1A1A1A", "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#E377C2", "#8C564B", "#17BECF",
]

METHOD_LABEL = {"plm": "PLM", "dhr_postprocess": "DHR", "tmvec": "TMVec"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def ensure_core_imports() -> None:
    core = Path(__file__).resolve().parents[1] / "core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    if "tqdm" not in sys.modules:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            class _FakeTqdm:
                def __init__(self, iterable=None, **kwargs):
                    self.iterable = iterable

                def __iter__(self):
                    return iter(self.iterable)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def update(self, n=1):
                    return None

                def close(self):
                    return None

            fake = types.ModuleType("tqdm")
            fake.tqdm = _FakeTqdm
            sys.modules["tqdm"] = fake


def get_tqdm():
    ensure_core_imports()
    from tqdm import tqdm

    return tqdm


def load_query_list(path: Path) -> set[str]:
    qids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qids.add(line.split()[0])
    if not qids:
        raise ValueError(f"No query ids in {path}")
    return qids


def filter_df_by_queries(df: pd.DataFrame, query_ids: set[str] | None) -> pd.DataFrame:
    if df.empty or not query_ids or "qid" not in df.columns:
        return df
    return df.loc[df["qid"].isin(query_ids)].copy()


def report_query_list_coverage(
    df_per: pd.DataFrame,
    query_ids: set[str],
    *,
    label: str,
) -> None:
    if df_per.empty:
        print(f"[WARN] {label}: no per-query rows after filtering")
        return
    found = set(df_per["qid"].unique())
    missing = query_ids - found
    extra = found - query_ids
    print(
        f"[INFO] {label}: pooling {len(found)} / {len(query_ids)} queries from list"
        + (f"; missing={len(missing)}" if missing else "")
        + (f"; extra={len(extra)}" if extra else "")
    )
    if missing and len(missing) <= 10:
        print(f"[WARN]   not in data: {sorted(missing)}")
    elif missing:
        print(f"[WARN]   not in data (first 10): {sorted(missing)[:10]} ...")


def _grep_qids_to_file(
    src: Path,
    dst: Path,
    qids: set[str],
    *,
    decoy: bool,
    decoy_suff: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"Empty file: {src}")
    tmp = dst.with_suffix(".tmp")
    with tmp.open("w", newline="") as out:
        out.write(header)
    for qid in sorted(qids):
        key = f"{qid}{decoy_suff}" if decoy else qid
        pattern = f"^{re.escape(key)}\t"
        with tmp.open("ab") as out_ab:
            subprocess.run(
                ["grep", "-E", pattern, str(src)],
                check=False,
                stdout=out_ab,
            )
    n_lines = sum(1 for _ in tmp.open()) - 1
    print(f"[INFO] grep -> {dst.name}: {n_lines:,} data rows")
    tmp.replace(dst)  # atomic rename; only reaches here if all qids were processed


def _cache_is_valid(path: Path, expected_qids: set[str], *, decoy: bool, decoy_suff: str) -> bool:
    if not path.exists() or path.stat().st_size <= 1:
        return False
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if "qid" not in (reader.fieldnames or []):
                return False
            found: set[str] = set()
            for row in reader:
                raw = row["qid"]
                qid = raw[: -len(decoy_suff)] if (decoy and raw.endswith(decoy_suff)) else raw
                found.add(qid)
        return expected_qids <= found
    except Exception:
        return False


def grep_extract_pair_files(
    real_path: Path,
    decoy_path: Path,
    qids: set[str],
    decoy_suff: str,
    cache_dir: Path,
) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{len(qids)}q"
    real_sub = cache_dir / f"{real_path.stem}_{tag}.tsv"
    decoy_sub = cache_dir / f"{decoy_path.stem}_{tag}.tsv"
    if not _cache_is_valid(real_sub, qids, decoy=False, decoy_suff=decoy_suff):
        print(f"[INFO] grep extract real ({len(qids)} queries) ...")
        _grep_qids_to_file(real_path, real_sub, qids, decoy=False, decoy_suff=decoy_suff)
    if not _cache_is_valid(decoy_sub, qids, decoy=True, decoy_suff=decoy_suff):
        print(f"[INFO] grep extract decoy ({len(qids)} queries) ...")
        _grep_qids_to_file(decoy_path, decoy_sub, qids, decoy=True, decoy_suff=decoy_suff)
    return real_sub, decoy_sub


def count_distinct_qids(path: Path) -> int:
    n = 0
    last: str | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = row["qid"]
            if qid != last:
                n += 1
                last = qid
    return n


def remove_decoy_suffix(qid: str, decoy_suffix: str) -> str:
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def file_size_gb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024**3)


def _rows_to_pair_frame(real_rows: list[dict], decoy_rows: list[dict], decoy_suffix: str) -> pd.DataFrame:
    df_real = pd.DataFrame(real_rows)
    df_decoy = pd.DataFrame(decoy_rows)
    if df_real.empty:
        return pd.DataFrame()

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
    df_real["score_real"] = df_real["score_real"].astype(np.float32)

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
    df_decoy["score_decoy"] = df_decoy["score_decoy"].astype(np.float32)

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig", "tid"], errors="ignore")
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    return df_merge


def _iter_qid_blocks(path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    current_qid: str | None = None
    block: list[dict] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["qid"]
            if current_qid is not None and qid != current_qid:
                yield current_qid, block
                block = []
            current_qid = qid
            block.append(row)
    if block and current_qid is not None:
        yield current_qid, block


def iter_pair_tables_by_qid(
    path_real: str | Path,
    path_decoy: str | Path,
    decoy_suffix: str,
) -> Iterator[tuple[str, pd.DataFrame]]:
    real_iter = _iter_qid_blocks(path_real)
    decoy_iter = _iter_qid_blocks(path_decoy)

    for (real_qid, real_rows), (decoy_qid, decoy_rows) in zip(real_iter, decoy_iter, strict=True):
        qid = remove_decoy_suffix(decoy_qid, decoy_suffix)
        if qid != real_qid:
            raise ValueError(f"QID mismatch while streaming: real={real_qid!r} decoy={decoy_qid!r}")
        df_merge = _rows_to_pair_frame(real_rows, decoy_rows, decoy_suffix)
        if not df_merge.empty:
            yield real_qid, df_merge


def decoy_suffix(decoy_method: str) -> str:
    if decoy_method not in METHOD2TYPE:
        raise ValueError(f"Unknown decoy method '{decoy_method}'")
    return f"_{METHOD2TYPE[decoy_method]}"


def path_target_noisy(search_method: str, seq_name: str) -> Path:
    return DATA_DIR / f"result_{search_method}_{seq_name}_target_noisy_{seq_name}.txt"


def path_calibrated_decoy(search_method: str, decoy_method: str, seq_name: str) -> Path:
    return DATA_DIR / (
        f"result_{search_method}_{seq_name}_{decoy_method}_calibrated_gam_"
        f"{WEIGHT_METHOD}_{seq_name}.txt"
    )


def combo_label(search_method: str, decoy_method: str) -> str:
    return f"{METHOD_LABEL.get(search_method, search_method)} + {decoy_method}"


def combo_id(search_method: str, decoy_method: str) -> str:
    return f"{search_method}__{decoy_method}"


def _as_int_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(-999).astype(int).values


def classification_masks(homo_types) -> tuple[np.ndarray, np.ndarray]:
    homo = _as_int_array(homo_types)
    is_tp = np.isin(homo, POSITIVE_HOMO_TYPES)
    is_fp = homo == NON_HOMO_TYPE
    return is_tp, is_fp


def select_threshold_at_q(df_curve: pd.DataFrame, q: float) -> float | None:
    valid = df_curve[df_curve["est_fdp"] <= q]
    if valid.empty:
        return None
    return float(valid.loc[valid["threshold"].idxmin(), "threshold"])


def apply_recall_zero_precision_one(recall: float, precision: float) -> tuple[float, float]:
    if recall <= 1e-12:
        return 0.0, 1.0
    return recall, precision


def metrics_at_threshold(
    df_q: pd.DataFrame,
    threshold: float,
    score_descending: bool = SCORE_DESCENDING,
) -> dict[str, float]:
    scores = df_q["score_real"].astype(float).values
    is_tp, is_fp = classification_masks(df_q["homo_type_real"].values)
    selected = scores >= threshold if score_descending else scores <= threshold

    tp = int(np.sum(selected & is_tp))
    fp = int(np.sum(selected & is_fp))
    total_pos = int(np.sum(is_tp))

    if total_pos == 0:
        return {"tp": tp, "fp": fp, "total_pos": total_pos, "recall": np.nan, "precision": np.nan}

    recall = tp / total_pos
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall, precision = apply_recall_zero_precision_one(recall, precision)
    return {"tp": tp, "fp": fp, "total_pos": total_pos, "recall": recall, "precision": precision}


def metrics_at_q(
    df_q: pd.DataFrame,
    df_curve: pd.DataFrame,
    q: float,
) -> dict[str, float]:
    threshold = select_threshold_at_q(df_curve, q)
    if threshold is None:
        return {
            "tp": 0,
            "fp": 0,
            "total_pos": int(classification_masks(df_q["homo_type_real"].values)[0].sum()),
            "recall": NO_THRESHOLD_RECALL,
            "precision": NO_THRESHOLD_PRECISION,
        }
    return metrics_at_threshold(df_q, threshold)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def load_pair_merge(search_method: str, decoy_method: str, seq_name: str) -> pd.DataFrame:
    ensure_core_imports()
    from fdr import make_pair_table

    real_path = path_target_noisy(search_method, seq_name)
    decoy_path = path_calibrated_decoy(search_method, decoy_method, seq_name)
    if not real_path.exists():
        raise FileNotFoundError(f"Missing target_noisy: {real_path}")
    if not decoy_path.exists():
        raise FileNotFoundError(f"Missing calibrated decoy: {decoy_path}")

    return make_pair_table(str(real_path), str(decoy_path), decoy_suffix(decoy_method))


def compute_fdr_curves(df_merge: pd.DataFrame) -> pd.DataFrame:
    ensure_core_imports()
    from fdr import compute_tda_fdr_per_query

    curves = []
    for (_, _), df_q in df_merge.groupby(["qid", "rep_id"], sort=False):
        curve = compute_tda_fdr_per_query(df_q)
        if not curve.empty:
            curves.append(curve)
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def first_valid_q(df_curve: pd.DataFrame, q_levels: np.ndarray) -> float | None:
    for q in sorted(q_levels):
        if select_threshold_at_q(df_curve, float(q)) is not None:
            return float(q)
    return None


def bin_first_valid_q(fvq: float, start: float = COHORT_BIN_START, width: float = COHORT_BIN_WIDTH) -> float:
    if fvq is None or (isinstance(fvq, float) and np.isnan(fvq)):
        return 1.0
    if fvq < start:
        return start
    idx = int(np.floor((fvq - start) / width + 1e-9))
    return round(start + idx * width, 10)


def cohort_range_label(q_cohort: float, width: float = COHORT_BIN_WIDTH) -> str:
    hi = min(q_cohort + width, 1.0)
    return f"[{q_cohort:.1f}, {hi:.1f})"


def cohort_group_label(q_cohort: float) -> str:
    return f"{q_cohort:.1f} group"


def combo_legend_text(search_method: str, decoy_method: str) -> str:
    sm = METHOD_LABEL.get(search_method, search_method)
    return f"{sm} + {decoy_method}"


def _combo_paths(search_method: str, decoy_method: str, seq_name: str) -> tuple[Path, Path]:
    real_path = path_target_noisy(search_method, seq_name)
    decoy_path = path_calibrated_decoy(search_method, decoy_method, seq_name)
    if not real_path.exists():
        raise FileNotFoundError(f"Missing target_noisy: {real_path}")
    if not decoy_path.exists():
        raise FileNotFoundError(f"Missing calibrated decoy: {decoy_path}")
    return real_path, decoy_path


def _should_stream_pair_table(real_path: Path, decoy_path: Path, force_load_all: bool) -> bool:
    if force_load_all:
        return False
    return (
        file_size_gb(real_path) > STREAMING_THRESHOLD_GB
        or file_size_gb(decoy_path) > STREAMING_THRESHOLD_GB
    )


def _metrics_rows_for_query(
    df_qall: pd.DataFrame,
    df_curve: pd.DataFrame,
    *,
    seq_name: str,
    search_method: str,
    decoy_method: str,
    qid: str,
    rep_id,
    q_levels: np.ndarray,
) -> list[dict]:
    fvq = first_valid_q(df_curve, q_levels)
    rows = []
    for q in q_levels:
        q = float(q)
        m = metrics_at_q(df_qall, df_curve, q)
        rows.append({
            "seq_name": seq_name,
            "search_method": search_method,
            "decoy_method": decoy_method,
            "combo_id": combo_id(search_method, decoy_method),
            "combo_label": combo_label(search_method, decoy_method),
            "qid": qid,
            "rep_id": rep_id,
            "first_valid_q": fvq,
            "q": q,
            "recall": m["recall"],
            "precision": m["precision"],
        })
    return rows


def _pr_curve_rows_for_query(
    df_qall: pd.DataFrame,
    df_curve: pd.DataFrame,
    *,
    seq_name: str,
    search_method: str,
    decoy_method: str,
    qid: str,
    rep_id,
) -> list[dict]:
    cid = combo_id(search_method, decoy_method)
    th = df_curve["threshold"].astype(float).values
    if th.size == 0:
        return []

    scores = df_qall["score_real"].astype(float).values
    is_tp, is_fp = classification_masks(df_qall["homo_type_real"].values)
    if SCORE_DESCENDING:
        sel = scores[:, None] >= th[None, :]
    else:
        sel = scores[:, None] <= th[None, :]
    tp = np.sum(sel & is_tp[:, None], axis=0)
    fp = np.sum(sel & is_fp[:, None], axis=0)
    total_pos = int(is_tp.sum())
    if total_pos == 0:
        return []

    recall = tp.astype(float) / total_pos
    denom = tp + fp
    precision = np.where(denom > 0, tp / denom, 1.0)
    zero_rec = recall <= 1e-12
    precision = np.where(zero_rec, 1.0, precision)
    recall = np.where(zero_rec, 0.0, recall)

    rows = []
    base = {
        "seq_name": seq_name,
        "search_method": search_method,
        "decoy_method": decoy_method,
        "combo_id": cid,
        "combo_label": combo_label(search_method, decoy_method),
        "qid": qid,
        "rep_id": rep_id,
    }
    for i, t in enumerate(th):
        r, p = float(recall[i]), float(precision[i])
        if np.isnan(r) or np.isnan(p):
            continue
        rows.append({**base, "threshold": float(t), "recall": r, "precision": p})
    return rows


def evaluate_combo_per_rep_streaming(
    search_method: str,
    decoy_method: str,
    seq_name: str,
    q_levels: np.ndarray,
    *,
    query_ids: set[str] | None = None,
    grep_extract_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_core_imports()
    from fdr import compute_tda_fdr_per_query

    real_path, decoy_path = _combo_paths(search_method, decoy_method, seq_name)
    suffix = decoy_suffix(decoy_method)
    if query_ids and grep_extract_dir is not None:
        real_path, decoy_path = grep_extract_pair_files(
            real_path, decoy_path, query_ids, suffix, grep_extract_dir
        )
    label = combo_label(search_method, decoy_method)
    print(
        f"[INFO] Streaming by qid ({label}): "
        f"real={file_size_gb(real_path):.1f} GB, decoy={file_size_gb(decoy_path):.1f} GB"
    )
    if query_ids:
        print(f"[INFO] Query list: {len(query_ids)} queries")
    rows: list[dict] = []
    pr_rows: list[dict] = []
    query_iter = iter_pair_tables_by_qid(real_path, decoy_path, suffix)
    if SHOW_PROGRESS:
        tqdm = get_tqdm()
        if query_ids:
            n_queries = len(query_ids)
        else:
            print(f"[INFO] Counting queries in {real_path.name} ...")
            n_queries = count_distinct_qids(real_path)
            print(f"[INFO] {n_queries:,} queries")
        query_iter = tqdm(
            query_iter,
            total=n_queries,
            desc=f"PR scan ({label})",
            unit="query",
            colour="cyan",
            dynamic_ncols=True,
        )

    for n_qid, (qid, df_qall) in enumerate(query_iter, start=1):
        if query_ids and qid not in query_ids:
            continue
        for rep_id, df_q in df_qall.groupby("rep_id", sort=False):
            df_curve = compute_tda_fdr_per_query(df_q)
            if df_curve.empty:
                continue
            kw = dict(
                seq_name=seq_name,
                search_method=search_method,
                decoy_method=decoy_method,
                qid=qid,
                rep_id=rep_id,
            )
            rows.extend(_metrics_rows_for_query(df_q, df_curve, q_levels=q_levels, **kw))
            pr_rows.extend(_pr_curve_rows_for_query(df_q, df_curve, **kw))
        if not SHOW_PROGRESS and n_qid % STREAM_LOG_EVERY == 0:
            print(f"[INFO]   processed {n_qid} queries ...")
        if n_qid % STREAM_LOG_EVERY == 0:
            gc.collect()

    return pd.DataFrame(rows), pd.DataFrame(pr_rows)


def evaluate_combo_per_rep_inmemory(
    search_method: str,
    decoy_method: str,
    seq_name: str,
    q_levels: np.ndarray,
    *,
    query_ids: set[str] | None = None,
    grep_extract_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = combo_label(search_method, decoy_method)
    print(f"[INFO] Loading pair table ({label}) ...")
    df_merge = load_pair_merge(search_method, decoy_method, seq_name)
    if query_ids:
        df_merge = df_merge.loc[df_merge["qid"].isin(query_ids)].copy()
        print(f"[INFO] Filtered to {df_merge['qid'].nunique()} queries from list")
    print(f"[INFO] Computing TDA FDR curves ({label}) ...")
    df_curve_all = compute_fdr_curves(df_merge)

    merge_groups = {k: g for k, g in df_merge.groupby(["qid", "rep_id"], sort=False)}
    curve_groups = {k: g for k, g in df_curve_all.groupby(["qid", "rep_id"], sort=False)}

    rows: list[dict] = []
    pr_rows: list[dict] = []
    curve_items = curve_groups.items()
    if SHOW_PROGRESS:
        curve_items = get_tqdm()(
            curve_items,
            total=len(curve_groups),
            desc=f"PR metrics ({label})",
            unit="query×rep",
            colour="cyan",
            dynamic_ncols=True,
        )
    for key, df_curve in curve_items:
        qid, rep_id = key
        kw = dict(
            seq_name=seq_name,
            search_method=search_method,
            decoy_method=decoy_method,
            qid=qid,
            rep_id=rep_id,
        )
        rows.extend(_metrics_rows_for_query(merge_groups[key], df_curve, q_levels=q_levels, **kw))
        pr_rows.extend(_pr_curve_rows_for_query(merge_groups[key], df_curve, **kw))
    del df_merge, df_curve_all, merge_groups, curve_groups
    gc.collect()
    return pd.DataFrame(rows), pd.DataFrame(pr_rows)


def evaluate_combo_per_rep(
    search_method: str,
    decoy_method: str,
    seq_name: str,
    q_levels: np.ndarray,
    *,
    force_load_all: bool = False,
    query_ids: set[str] | None = None,
    grep_extract_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_path, decoy_path = _combo_paths(search_method, decoy_method, seq_name)
    use_stream = _should_stream_pair_table(real_path, decoy_path, force_load_all)
    if query_ids and grep_extract_dir is not None:
        use_stream = True
    if use_stream:
        return evaluate_combo_per_rep_streaming(
            search_method,
            decoy_method,
            seq_name,
            q_levels,
            query_ids=query_ids,
            grep_extract_dir=grep_extract_dir,
        )
    return evaluate_combo_per_rep_inmemory(
        search_method,
        decoy_method,
        seq_name,
        q_levels,
        query_ids=query_ids,
        grep_extract_dir=grep_extract_dir,
    )


def collapse_reps_to_query(df_per_rep: pd.DataFrame) -> pd.DataFrame:
    if df_per_rep.empty:
        return df_per_rep.copy()

    fvq_by_qid = (
        df_per_rep.groupby(["combo_id", "qid"], as_index=False)["first_valid_q"]
        .min()
        .rename(columns={"first_valid_q": "first_valid_q"})
    )

    rows = []
    group_cols = ["seq_name", "search_method", "decoy_method", "combo_id", "combo_label", "qid", "q"]
    for keys, grp in df_per_rep.groupby(group_cols, sort=False):
        row = grp.iloc[0]
        r = float(grp["recall"].mean())
        p = float(grp["precision"].mean())
        r, p = apply_recall_zero_precision_one(r, p)
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": row["combo_id"],
            "combo_label": row["combo_label"],
            "qid": row["qid"],
            "q": float(row["q"]),
            "recall": r,
            "precision": p,
            "n_reps": len(grp),
        })
    df = pd.DataFrame(rows)
    df = df.merge(fvq_by_qid, on=["combo_id", "qid"], how="left")
    return df.sort_values(["combo_id", "qid", "q"]).reset_index(drop=True)


def collapse_reps_pr_to_query(df_pr_rep: pd.DataFrame) -> pd.DataFrame:
    if df_pr_rep.empty:
        return df_pr_rep.copy()

    rows = []
    group_cols = [
        "seq_name", "search_method", "decoy_method", "combo_id", "combo_label", "qid", "threshold",
    ]
    for _, grp in df_pr_rep.groupby(group_cols, sort=False):
        row = grp.iloc[0]
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": row["combo_id"],
            "combo_label": row["combo_label"],
            "qid": row["qid"],
            "threshold": float(row["threshold"]),
            "recall": float(grp["recall"].mean()),
            "precision": float(grp["precision"].mean()),
            "n_reps": len(grp),
        })
    return pd.DataFrame(rows).sort_values(["combo_id", "qid", "recall"]).reset_index(drop=True)


def _ensure_pr_origin_point(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "combo_id" not in df.columns:
        return df
    parts = []
    for cid, sub in df.groupby("combo_id", sort=False):
        sub = sub.sort_values("mean_recall").copy()
        has_origin = (sub["mean_recall"] <= 1e-12).any()
        if not has_origin:
            row = sub.iloc[0].to_dict()
            row["mean_recall"] = 0.0
            row["mean_precision"] = 1.0
            row["threshold"] = np.nan
            sub = pd.concat([pd.DataFrame([row]), sub], ignore_index=True)
        else:
            sub.loc[sub["mean_recall"] <= 1e-12, "mean_precision"] = 1.0
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def aggregate_pr_curve(df_per_pr: pd.DataFrame) -> pd.DataFrame:
    if df_per_pr.empty:
        return df_per_pr.copy()

    rows = []
    for (combo_id_val, threshold), grp in df_per_pr.groupby(["combo_id", "threshold"]):
        row = grp.iloc[0]
        mean_r = float(grp["recall"].mean())
        mean_p = float(grp["precision"].mean())
        mean_r, mean_p = apply_recall_zero_precision_one(mean_r, mean_p)
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": combo_id_val,
            "combo_label": row["combo_label"],
            "threshold": float(threshold),
            "mean_recall": mean_r,
            "mean_precision": mean_p,
            "n_queries": len(grp),
        })
    out = pd.DataFrame(rows).sort_values(["combo_id", "mean_recall"]).reset_index(drop=True)
    return _ensure_pr_origin_point(out)


def aggregate_pr_curve_grouped(df_per_pr: pd.DataFrame, df_per_q: pd.DataFrame) -> pd.DataFrame:
    if df_per_pr.empty:
        return df_per_pr.copy()

    fvq = (
        df_per_q.groupby(["combo_id", "qid"], as_index=False)["first_valid_q"]
        .min()
    )
    df = df_per_pr.merge(fvq, on=["combo_id", "qid"], how="left")
    df["q_cohort"] = df["first_valid_q"].map(bin_first_valid_q)
    df["cohort_label"] = df["q_cohort"].map(cohort_range_label)

    rows = []
    for (combo_id_val, q_cohort, threshold), grp in df.groupby(["combo_id", "q_cohort", "threshold"]):
        row = grp.iloc[0]
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": combo_id_val,
            "combo_label": row["combo_label"],
            "q_cohort": float(q_cohort),
            "cohort_label": row["cohort_label"],
            "threshold": float(threshold),
            "mean_recall": float(grp["recall"].mean()),
            "mean_precision": float(grp["precision"].mean()),
            "n_queries": len(grp),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["combo_id", "q_cohort", "mean_recall"]).reset_index(drop=True)
    parts = []
    for keys, sub in out.groupby(["combo_id", "q_cohort"], sort=False):
        env = _pr_upper_envelope(
            sub.rename(columns={"mean_recall": "recall", "mean_precision": "precision"})
        ).rename(columns={"recall": "mean_recall", "precision": "mean_precision"})
        for col in ("seq_name", "search_method", "decoy_method", "combo_label", "q_cohort", "cohort_label"):
            env[col] = sub[col].iloc[0]
        env["combo_id"] = keys[0]
        parts.append(env)
    return pd.concat(parts, ignore_index=True) if parts else out


def aggregate_pooled(df_per: pd.DataFrame) -> pd.DataFrame:
    if df_per.empty:
        return df_per.copy()
    rows = []
    for (combo_id_val, q), grp in df_per.groupby(["combo_id", "q"]):
        row = grp.iloc[0]
        mean_r = float(grp["recall"].mean())
        mean_p = float(grp["precision"].mean())
        mean_r, mean_p = apply_recall_zero_precision_one(mean_r, mean_p)
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": combo_id_val,
            "combo_label": row["combo_label"],
            "q": float(q),
            "mean_recall": mean_r,
            "mean_precision": mean_p,
            "n_valid": len(grp),
        })
    return pd.DataFrame(rows)


def aggregate_grouped(df_per: pd.DataFrame) -> pd.DataFrame:
    df = df_per.copy()
    df["q_cohort"] = df["first_valid_q"].map(bin_first_valid_q)

    rows = []
    for (combo_id_val, q_cohort, q), grp in df.groupby(["combo_id", "q_cohort", "q"]):
        row = grp.iloc[0]
        rows.append({
            "seq_name": row["seq_name"],
            "search_method": row["search_method"],
            "decoy_method": row["decoy_method"],
            "combo_id": combo_id_val,
            "combo_label": row["combo_label"],
            "q_cohort": float(q_cohort),
            "cohort_label": cohort_range_label(float(q_cohort)),
            "q": float(q),
            "mean_recall": float(grp["recall"].mean()),
            "mean_precision": float(grp["precision"].mean()),
            "n_valid": len(grp),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["combo_id", "q_cohort", "q"]).reset_index(drop=True)
    return out


def evaluate_combo(
    search_method: str,
    decoy_method: str,
    seq_name: str,
    q_levels: np.ndarray,
    *,
    force_load_all: bool = False,
    query_ids: set[str] | None = None,
    grep_extract_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_per_rep, df_pr_rep = evaluate_combo_per_rep(
        search_method,
        decoy_method,
        seq_name,
        q_levels,
        force_load_all=force_load_all,
        query_ids=query_ids,
        grep_extract_dir=grep_extract_dir,
    )
    df_per_rep = filter_df_by_queries(df_per_rep, query_ids)
    df_pr_rep = filter_df_by_queries(df_pr_rep, query_ids)
    df_per = collapse_reps_to_query(df_per_rep)
    if query_ids:
        report_query_list_coverage(df_per, query_ids, label=combo_label(search_method, decoy_method))
    df_pr = collapse_reps_pr_to_query(df_pr_rep)
    return df_per_rep, df_per, aggregate_pooled(df_per), aggregate_pr_curve(df_pr), df_pr


def run_tag(
    seq_name: str,
    search_methods: list[str],
    decoy_methods: list[str],
    *,
    query_list_stem: str | None = None,
) -> str:
    sm = "_".join(sorted(search_methods))
    dm = "_".join(sorted(decoy_methods))
    tag = f"{seq_name}_{sm}_{dm}"
    if query_list_stem:
        tag = f"{tag}_{query_list_stem}"
    return tag


@dataclass(frozen=True)
class RunOutputs:

    tag: str
    out_dir: Path
    data_dir: Path
    plot_data_tsv: Path
    recall_vs_q_pdf: Path
    precision_vs_q_pdf: Path
    pr_curve_from_q_pdf: Path
    grep_cache_dir: Path

    @classmethod
    def build(
        cls,
        out_dir: Path,
        seq_name: str,
        search_methods: list[str],
        decoy_methods: list[str],
        *,
        query_list_stem: str | None = None,
    ) -> RunOutputs:
        tag = run_tag(
            seq_name, search_methods, decoy_methods, query_list_stem=query_list_stem
        )
        suffix = f"_{tag}"
        data_dir = out_dir / DATA_SUBDIR
        return cls(
            tag=tag,
            out_dir=out_dir,
            data_dir=data_dir,
            plot_data_tsv=data_dir / f"pooled_pr_plot_data{suffix}.tsv",
            recall_vs_q_pdf=out_dir / f"recall_vs_fdr_q_pooled{suffix}.pdf",
            precision_vs_q_pdf=out_dir / f"precision_vs_fdr_q_pooled{suffix}.pdf",
            pr_curve_from_q_pdf=out_dir / f"pr_curve_from_q_pooled{suffix}.pdf",
            grep_cache_dir=out_dir / "grep_cache",
        )

    def plot_data_tsv_for_combo(self, combo_id: str) -> Path:
        return self.data_dir / f"pooled_pr_plot_data_{combo_id}_{self.tag}.tsv"

    def pr_curve_pooled_tsv(self, combo_id: str) -> Path:
        return self.data_dir / f"pr_curve_pooled_{combo_id}_{self.tag}.tsv"

def discover_combos(
    seq_name: str,
    search_methods: list[str],
    decoy_methods: list[str],
) -> list[tuple[str, str]]:
    combos = []
    for s, d in product(search_methods, decoy_methods):
        if path_target_noisy(s, seq_name).exists() and path_calibrated_decoy(s, d, seq_name).exists():
            combos.append((s, d))
    return combos


def _find_per_combo_tsv(out_dir: Path, stem: str, cid: str) -> Path | None:
    matches = sorted(out_dir.glob(f"{stem}_{cid}_*.tsv"))
    return matches[-1] if matches else None


def load_pooled_metrics_for_plot(
    outputs: RunOutputs,
    search_methods: list[str],
    decoy_methods: list[str],
    *,
    query_ids: set[str] | None = None,
) -> pd.DataFrame:
    if query_ids:
        raise ValueError(
            "plot-only with a query subset requires re-computing; "
            "run without --plot-only when --query-list is set."
        )
    if outputs.plot_data_tsv.exists():
        print(f"[INFO] plot-only: {outputs.plot_data_tsv.name}")
        return pd.read_csv(outputs.plot_data_tsv, sep="\t")

    pooled_parts: list[pd.DataFrame] = []
    for sm, dm in product(search_methods, decoy_methods):
        cid = combo_id(sm, dm)
        per_combo = outputs.plot_data_tsv_for_combo(cid)
        if per_combo.exists():
            print(f"[INFO] plot-only: reuse {per_combo.name}")
            pooled_parts.append(pd.read_csv(per_combo, sep="\t"))
            continue
        pooled_path = _find_per_combo_tsv(outputs.data_dir, "pooled_pr_plot_data", cid)
        if pooled_path is not None:
            print(f"[INFO] plot-only: reuse {pooled_path.name}")
            pooled_parts.append(pd.read_csv(pooled_path, sep="\t"))

    if pooled_parts:
        return pd.concat(pooled_parts, ignore_index=True)

    raise FileNotFoundError(
        f"No plot data for tag={outputs.tag} under {outputs.data_dir}."
    )


def run_compute(
    seq_name: str,
    search_methods: list[str],
    decoy_methods: list[str],
    outputs: RunOutputs,
    *,
    force_load_all: bool = False,
    query_ids: set[str] | None = None,
    use_grep_extract: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    outputs.data_dir.mkdir(parents=True, exist_ok=True)
    combos = discover_combos(seq_name, search_methods, decoy_methods)
    if not combos:
        raise FileNotFoundError(
            f"No valid combos for seq_name={seq_name}, search={search_methods}, decoy={decoy_methods}"
        )

    grep_dir = outputs.grep_cache_dir if (query_ids and use_grep_extract) else None
    print(
        f"[INFO] tag={outputs.tag}, combos={len(combos)}, "
        f"out_dir={outputs.out_dir}, data_dir={outputs.data_dir}"
    )
    if query_ids:
        print(f"[INFO] Pooled subset: {len(query_ids)} queries from query list")
    all_pooled, all_pr_pooled = [], []
    for search_method, decoy_method in combos:
        label = combo_label(search_method, decoy_method)
        print(f"[INFO] Evaluating {label} ...")
        df_per_rep, df_per, df_pooled, df_pr_pooled, df_pr = evaluate_combo(
            search_method,
            decoy_method,
            seq_name,
            Q_LEVELS,
            force_load_all=force_load_all,
            query_ids=query_ids,
            grep_extract_dir=grep_dir,
        )
        gc.collect()

        cid = combo_id(search_method, decoy_method)
        save_pooled_plot_data(df_pooled, outputs.plot_data_tsv_for_combo(cid))
        df_pr_pooled.to_csv(outputs.pr_curve_pooled_tsv(cid), sep="\t", index=False)
        print(f"[OK] {outputs.pr_curve_pooled_tsv(cid).name}")

        print(
            f"[OK] {label}: {df_per['qid'].nunique()} queries, "
            f"pooled={len(df_pooled)} q-pts, pr={len(df_pr_pooled)} curve pts"
        )
        all_pooled.append(df_pooled)
        all_pr_pooled.append(df_pr_pooled)

    combined_pooled = pd.concat(all_pooled, ignore_index=True)
    combined_pr = pd.concat(all_pr_pooled, ignore_index=True)
    save_pooled_plot_data(combined_pooled, outputs.plot_data_tsv)
    print(f"[DONE] Plot data -> {outputs.plot_data_tsv}")
    return combined_pooled, combined_pr


# Plot
# ---------------------------------------------------------------------------
PANEL_SIZE_IN = 6.5
LEGEND_WIDTH_IN = 3.0
LEGEND_OFFSET_FIG = 0.012  # grouped: legend x = panel right + this (figure fraction)
POOLED_COUNT_AXIS_OUTWARD_PT = 72  # third y-axis spine offset (points)
POOLED_LEGEND_LABEL_PAD_IN = 0.65  # extra inches past count-axis ticks before legend
POOLED_LEGEND_EXTRA_FIG = 0.0  # add on top of auto gap; increase to shift pooled legend further right
PANEL_GAP_IN = 2.0
MARGIN_LEFT_IN = 0.55
MARGIN_BOTTOM_IN = 1.35
MARGIN_TOP_IN = 0.35
FIG_TITLE_Y = 0.008
AXIS_FONTSIZE = 17
FIG_TITLE_FONTSIZE = 17
COMBO_LINESTYLES = ["-", "--", "-.", ":"]
COUNT_LINE_COLOR = "0.45"
COUNT_LINE_STYLE = "--"
POOLED_RECALL_COLOR = LINE_COLORS[0]
POOLED_PRECISION_COLOR = LINE_COLORS[1]
POOLED_MARKER_SIZE = 5.5
POOLED_LINE_WIDTH = 1.8
Q_VLINE_ALPHA = 0.55
Q_VLINE_WIDTH = 0.9
PR_CURVE_LINE_WIDTH = 1.8
NO_THRESHOLD_RECALL = 0.0
NO_THRESHOLD_PRECISION = 1.0


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"[OK] Saved {path}")


def _remove_stale_plots(out_dir: Path) -> None:
    for name in (
        "recall_vs_fdr_q_grouped.pdf",
        "precision_vs_fdr_q_grouped.pdf",
        "recall_vs_fdr_q_grouped.png",
        "precision_vs_fdr_q_grouped.png",
        "recall_precision_vs_fdr_q_grouped.png",
        "recall_precision_vs_fdr_q_grouped.pdf",
        "recall_precision_vs_fdr_q_pooled.pdf",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()
            print(f"[OK] Removed stale plot {path}")


def _blue_gradient_colors(n_groups: int) -> list[tuple[float, float, float, float]]:
    if n_groups <= 0:
        return []
    if n_groups == 1:
        return [plt.cm.Blues(0.75)]
    levels = np.linspace(0.95, 0.45, n_groups)
    return [plt.cm.Blues(v) for v in levels]


def _prepare_grouped_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "q_cohort" not in out.columns:
        if "first_valid_q" not in out.columns:
            raise ValueError("Grouped TSV must contain q_cohort or first_valid_q")
        out["q_cohort"] = out["first_valid_q"].map(bin_first_valid_q)
    if "cohort_label" not in out.columns:
        out["cohort_label"] = out["q_cohort"].map(cohort_range_label)
    return out


def _style_pr_panel(ax: plt.Axes, title: str) -> None:
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel("Recall", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_box_aspect(1)


def _add_square_panel(fig: plt.Figure, left_in: float, bottom_in: float) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes(
        [
            left_in / fig_w,
            bottom_in / fig_h,
            PANEL_SIZE_IN / fig_w,
            PANEL_SIZE_IN / fig_h,
        ]
    )
    ax.set_box_aspect(1)
    return ax


def _pooled_legend_offset_fig(fig: plt.Figure) -> float:
    fig_w_in, _ = fig.get_size_inches()
    clear_in = POOLED_COUNT_AXIS_OUTWARD_PT / 72.0 + POOLED_LEGEND_LABEL_PAD_IN
    return LEGEND_OFFSET_FIG + clear_in / fig_w_in + POOLED_LEGEND_EXTRA_FIG


def _place_legend_right(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str | None = None,
    *,
    handles: list | None = None,
    labels: list[str] | None = None,
    offset_fig: float | None = None,
    title_fontweight: str | None = None,
) -> None:
    pos = ax.get_position()
    dx = LEGEND_OFFSET_FIG if offset_fig is None else offset_fig
    anchor = (pos.x1 + dx, pos.y1)
    legend_kw: dict = dict(
        loc="upper left",
        bbox_to_anchor=anchor,
        bbox_transform=fig.transFigure,
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="0.35",
        facecolor="white",
        borderaxespad=0.0,
    )
    if title_fontweight:
        legend_kw["title_fontproperties"] = {"weight": title_fontweight, "size": 9}
    else:
        legend_kw["title_fontsize"] = 9
    if handles is not None:
        ax.legend(handles, labels, title=title, **legend_kw)
    else:
        ax.legend(title=title, **legend_kw)


def _create_single_pr_figure() -> tuple[plt.Figure, plt.Axes]:
    fig_w = MARGIN_LEFT_IN + PANEL_SIZE_IN + LEGEND_WIDTH_IN + 0.35
    fig_h = MARGIN_BOTTOM_IN + PANEL_SIZE_IN + MARGIN_TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _add_square_panel(fig, MARGIN_LEFT_IN, MARGIN_BOTTOM_IN)
    return fig, ax


def _create_metric_vs_q_figure() -> tuple[plt.Figure, plt.Axes]:
    return _create_single_pr_figure()


def _style_metric_vs_q_panel(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set(xlim=(0, 1), ylim=(-0.02, 1.02))
    ax.set_xlabel("FDR threshold (q)", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_box_aspect(1)


def pooled_recall_precision_functions(df_pooled: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seq_name", "search_method", "decoy_method", "combo_id", "combo_label",
        "q", "mean_recall", "mean_precision", "n_valid",
    ]
    present = [c for c in cols if c in df_pooled.columns]
    out = df_pooled[present].copy()
    return out.sort_values(["combo_id", "q"]).reset_index(drop=True)


def save_pooled_plot_data(df_pooled: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled_recall_precision_functions(df_pooled).to_csv(path, sep="\t", index=False)
    print(f"[OK] Plot data -> {path}")
    return path


def _combos_from_pooled(df_pooled: pd.DataFrame) -> pd.DataFrame:
    return (
        df_pooled[["combo_id", "search_method", "decoy_method", "combo_label"]]
        .drop_duplicates()
        .sort_values("combo_label")
    )


def _plot_combo_metric_vs_q(
    ax: plt.Axes,
    sub: pd.DataFrame,
    metric_col: str,
    *,
    color,
    label: str,
) -> None:
    sub = sub.sort_values("q")
    ax.plot(
        sub["q"],
        sub[metric_col],
        marker="o",
        markersize=POOLED_MARKER_SIZE,
        linewidth=POOLED_LINE_WIDTH,
        color=color,
        linestyle="-",
        label=label,
    )


def plot_pooled_recall_vs_q(
    df_pooled: pd.DataFrame,
    seq_name: str,
    outputs: RunOutputs,
    *,
    n_queries: int | None = None,
) -> None:
    if df_pooled.empty:
        print("[WARN] Pooled metrics empty; skip recall vs q")
        return
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    combos = _combos_from_pooled(df_pooled)
    fig, ax = _create_metric_vs_q_figure()
    for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
        sub = df_pooled[df_pooled["combo_id"] == combo_row.combo_id]
        color = LINE_COLORS[combo_idx % len(LINE_COLORS)]
        if len(combos) == 1:
            color = POOLED_RECALL_COLOR
        _plot_combo_metric_vs_q(
            ax,
            sub,
            "mean_recall",
            color=color,
            label=combo_legend_text(combo_row.search_method, combo_row.decoy_method),
        )
    title = combos["combo_label"].iloc[0] if len(combos) == 1 else "Pooled recall vs FDR q"
    _style_metric_vs_q_panel(ax, title, "Recall")
    _place_legend_right(fig, ax)
    n_line = f"; n={n_queries} queries" if n_queries else ""
    fig.text(
        0.5, FIG_TITLE_Y,
        f"Pooled recall vs FDR q ({seq_name}{n_line})",
        ha="center", va="bottom", fontsize=FIG_TITLE_FONTSIZE, transform=fig.transFigure,
    )
    _save_pdf(fig, outputs.recall_vs_q_pdf)
    plt.close(fig)


def plot_pooled_precision_vs_q(
    df_pooled: pd.DataFrame,
    seq_name: str,
    outputs: RunOutputs,
    *,
    n_queries: int | None = None,
) -> None:
    if df_pooled.empty:
        print("[WARN] Pooled metrics empty; skip precision vs q")
        return
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    combos = _combos_from_pooled(df_pooled)
    fig, ax = _create_metric_vs_q_figure()
    for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
        sub = df_pooled[df_pooled["combo_id"] == combo_row.combo_id]
        color = LINE_COLORS[combo_idx % len(LINE_COLORS)]
        if len(combos) == 1:
            color = POOLED_PRECISION_COLOR
        _plot_combo_metric_vs_q(
            ax,
            sub,
            "mean_precision",
            color=color,
            label=combo_legend_text(combo_row.search_method, combo_row.decoy_method),
        )
    title = combos["combo_label"].iloc[0] if len(combos) == 1 else "Pooled precision vs FDR q"
    _style_metric_vs_q_panel(ax, title, "Precision")
    _place_legend_right(fig, ax)
    n_line = f"; n={n_queries} queries" if n_queries else ""
    fig.text(
        0.5, FIG_TITLE_Y,
        f"Pooled precision vs FDR q ({seq_name}{n_line})",
        ha="center", va="bottom", fontsize=FIG_TITLE_FONTSIZE, transform=fig.transFigure,
    )
    _save_pdf(fig, outputs.precision_vs_q_pdf)
    plt.close(fig)


def plot_pooled_pr_from_q(
    df_pooled: pd.DataFrame,
    seq_name: str,
    outputs: RunOutputs,
    *,
    n_queries: int | None = None,
) -> None:
    if df_pooled.empty:
        print("[WARN] Pooled metrics empty; skip PR from q")
        return
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_plots(outputs.out_dir)
    df_fn = pooled_recall_precision_functions(df_pooled)
    combos = _combos_from_pooled(df_fn)
    fig, ax = _create_single_pr_figure()

    for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
        sub = df_fn[df_fn["combo_id"] == combo_row.combo_id].sort_values("q")
        color = LINE_COLORS[combo_idx % len(LINE_COLORS)]
        if len(combos) == 1:
            color = POOLED_RECALL_COLOR
        r = sub["mean_recall"].astype(float).values
        p = sub["mean_precision"].astype(float).values
        ax.plot(
            r,
            p,
            marker="o",
            markersize=POOLED_MARKER_SIZE,
            linewidth=PR_CURVE_LINE_WIDTH,
            color=color,
            linestyle="-",
            label=combo_legend_text(combo_row.search_method, combo_row.decoy_method),
        )
        for _, row in sub.iterrows():
            if not q_should_label(float(row["q"])):
                continue
            ax.annotate(
                format_q_label(row["q"]),
                (row["mean_recall"], row["mean_precision"]),
                fontsize=7,
                color=color,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )

    title = combos["combo_label"].iloc[0] if len(combos) == 1 else "Pooled PR from recall(q), precision(q)"
    _style_pr_panel(ax, title)
    _place_legend_right(fig, ax)
    n_line = f"; n={n_queries} queries" if n_queries else ""
    q_label = ", ".join(format_q_label(x) for x in Q_LABEL_LEVELS)
    fig.text(
        0.5,
        FIG_TITLE_Y,
        f"PR curve (recall(q), precision(q)); q ∈ {{{q_label}}} ({seq_name}{n_line})",
        ha="center",
        va="bottom",
        fontsize=FIG_TITLE_FONTSIZE - 1,
        transform=fig.transFigure,
    )
    _save_pdf(fig, outputs.pr_curve_from_q_pdf)
    plt.close(fig)


def _legend_header(combos: pd.DataFrame) -> str:
    return "\n".join(
        combo_legend_text(row.search_method, row.decoy_method)
        for row in combos.itertuples(index=False)
    )


def _plot_pr_curve_line(
    ax: plt.Axes,
    sub_pr: pd.DataFrame,
    *,
    color,
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    if sub_pr.empty:
        return
    sub = sub_pr.sort_values("mean_recall").copy()
    if not (sub["mean_recall"] <= 1e-12).any():
        origin = sub.iloc[0].to_dict()
        origin["mean_recall"] = 0.0
        origin["mean_precision"] = 1.0
        sub = pd.concat([pd.DataFrame([origin]), sub], ignore_index=True)
    else:
        sub.loc[sub["mean_recall"] <= 1e-12, "mean_precision"] = 1.0
    ax.plot(
        sub["mean_recall"],
        sub["mean_precision"],
        linewidth=PR_CURVE_LINE_WIDTH,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def _plot_q_threshold_vlines(
    ax: plt.Axes,
    sub_q: pd.DataFrame,
    *,
    color,
    q_levels: np.ndarray | None = None,
) -> list[Line2D]:
    handles: list[Line2D] = []
    levels = q_levels if q_levels is not None else sorted(sub_q["q"].unique())
    for q in levels:
        row = sub_q.loc[np.isclose(sub_q["q"], float(q))]
        if row.empty:
            continue
        r = float(row["mean_recall"].iloc[0])
        p = float(row["mean_precision"].iloc[0])
        ax.plot(
            [r, r],
            [0.0, p],
            linestyle="--",
            color=color,
            linewidth=Q_VLINE_WIDTH,
            alpha=Q_VLINE_ALPHA,
        )
        handles.append(
            Line2D([0], [0], color=color, linestyle="--", linewidth=Q_VLINE_WIDTH, label=f"q={q:.1f}")
        )
    return handles


def _plot_cohort_pr_curves(
    ax: plt.Axes,
    df_pr: pd.DataFrame,
    df_q: pd.DataFrame,
    combos: pd.DataFrame,
    cohort_values: list[float],
    cohort_color_map: dict[float, tuple],
) -> None:
    for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
        sub_combo_pr = df_pr[df_pr["combo_id"] == combo_row.combo_id]
        sub_combo_q = df_q[df_q["combo_id"] == combo_row.combo_id]
        ls = COMBO_LINESTYLES[combo_idx % len(COMBO_LINESTYLES)]
        for q_cohort in cohort_values:
            sub_pr = sub_combo_pr[sub_combo_pr["q_cohort"] == q_cohort].sort_values("mean_recall")
            sub_q = sub_combo_q[sub_combo_q["q_cohort"] == q_cohort]
            sub_q = sub_q[sub_q["q"] >= q_cohort - 1e-9]
            if sub_pr.empty and sub_q.empty:
                continue
            color = cohort_color_map[q_cohort]
            n = int(sub_q["n_valid"].max()) if not sub_q.empty else int(sub_pr["n_queries"].max())
            _plot_pr_curve_line(
                ax,
                sub_pr,
                color=color,
                linestyle=ls,
                label=f"{cohort_group_label(q_cohort)} (n={n})",
            )
            _plot_q_threshold_vlines(ax, sub_q, color=color)


def plot_grouped_pr_curves(
    df_q: pd.DataFrame,
    df_pr: pd.DataFrame,
    seq_name: str,
    outputs: RunOutputs,
) -> None:
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_plots(outputs.out_dir)

    df_q = _prepare_grouped_df(df_q)
    combos = (
        df_q[["combo_id", "search_method", "decoy_method", "combo_label"]]
        .drop_duplicates()
        .sort_values("combo_label")
    )
    cohort_values = sorted(df_q["q_cohort"].unique())
    cohort_colors = _blue_gradient_colors(len(cohort_values))
    cohort_color_map = {q: cohort_colors[i] for i, q in enumerate(cohort_values)}

    fig, ax = _create_single_pr_figure()
    if not df_pr.empty:
        _plot_cohort_pr_curves(ax, df_pr, df_q, combos, cohort_values, cohort_color_map)
    else:
        print("[WARN] No grouped PR curve TSV; plotting q operating points only")
        for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
            sub_q = df_q[df_q["combo_id"] == combo_row.combo_id]
            color = LINE_COLORS[combo_idx % len(LINE_COLORS)]
            sub = sub_q.sort_values("mean_recall")
            ax.plot(
                sub["mean_recall"],
                sub["mean_precision"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
                linestyle=COMBO_LINESTYLES[combo_idx % len(COMBO_LINESTYLES)],
                label=combo_legend_text(combo_row.search_method, combo_row.decoy_method),
            )
            _plot_q_threshold_vlines(ax, sub_q, color=color)

    _style_pr_panel(ax, "Grouped PR curve")
    _place_legend_right(fig, ax, _legend_header(combos))

    fig.text(
        0.5,
        FIG_TITLE_Y,
        f"Grouped PR curves ({seq_name}; cohort width={COHORT_BIN_WIDTH})",
        ha="center",
        va="bottom",
        fontsize=FIG_TITLE_FONTSIZE,
        transform=fig.transFigure,
    )
    _save_pdf(fig, outputs.grouped_pdf)
    plt.close(fig)


def plot_pooled_pr_curves(
    df_q: pd.DataFrame,
    df_pr: pd.DataFrame,
    seq_name: str,
    outputs: RunOutputs,
    *,
    n_queries: int | None = None,
) -> None:
    outputs.out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_plots(outputs.out_dir)
    if df_q.empty:
        print("[WARN] Pooled metrics empty; skip pooled plot")
        return

    combos = (
        df_q[["combo_id", "search_method", "decoy_method", "combo_label"]]
        .drop_duplicates()
        .sort_values("combo_label")
    )
    fig, ax = _create_single_pr_figure()
    q_legend_handles: list[Line2D] = []

    for combo_idx, combo_row in enumerate(combos.itertuples(index=False)):
        color = LINE_COLORS[combo_idx % len(LINE_COLORS)]
        if len(combos) == 1:
            color = POOLED_RECALL_COLOR
        sub_q = df_q[df_q["combo_id"] == combo_row.combo_id].sort_values("q")
        sub_pr = df_pr[df_pr["combo_id"] == combo_row.combo_id] if not df_pr.empty else pd.DataFrame()
        label = combo_legend_text(combo_row.search_method, combo_row.decoy_method)
        if not sub_pr.empty:
            _plot_pr_curve_line(ax, sub_pr, color=color, linestyle="-", label=label)
        elif not sub_q.empty:
            sub_pts = sub_q.sort_values("mean_recall")
            ax.plot(
                sub_pts["mean_recall"],
                sub_pts["mean_precision"],
                marker="o",
                markersize=POOLED_MARKER_SIZE,
                linewidth=POOLED_LINE_WIDTH,
                color=color,
                linestyle="-",
                label=label,
            )
        q_legend_handles.extend(_plot_q_threshold_vlines(ax, sub_q, color=color, q_levels=Q_LEVELS))

    _style_pr_panel(ax, combos["combo_label"].iloc[0] if len(combos) == 1 else "Pooled PR curve")

    method_handles = [
        Line2D([0], [0], color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=PR_CURVE_LINE_WIDTH, label=row.combo_label)
        for i, row in enumerate(combos.itertuples(index=False))
    ]
    q_seen: set[float] = set()
    q_handles: list[Line2D] = []
    for h in q_legend_handles:
        q_val = float(h.get_label().split("=")[1])
        if q_val in q_seen:
            continue
        q_seen.add(q_val)
        h.set_color("0.35")
        q_handles.append(h)
    handles = method_handles + q_handles if len(combos) > 1 else (
        [Line2D([0], [0], color=POOLED_RECALL_COLOR, linewidth=PR_CURVE_LINE_WIDTH, label="PR curve")] + q_handles
    )
    _place_legend_right(fig, ax, handles=[h for h in handles], labels=[h.get_label() for h in handles])

    n_line = f"; n={n_queries} queries" if n_queries else ""
    fig.text(
        0.5,
        FIG_TITLE_Y,
        f"Pooled PR curve ({seq_name}{n_line})",
        ha="center",
        va="bottom",
        fontsize=FIG_TITLE_FONTSIZE,
        transform=fig.transFigure,
    )
    _save_pdf(fig, outputs.pr_curve_from_q_pdf)
    plt.close(fig)


def load_pr_curve_for_plot(
    outputs: RunOutputs,
    search_methods: list[str],
    decoy_methods: list[str],
) -> pd.DataFrame:
    parts = []
    for sm, dm in product(search_methods, decoy_methods):
        cid = combo_id(sm, dm)
        path = outputs.pr_curve_pooled_tsv(cid)
        if path.exists():
            parts.append(pd.read_csv(path, sep="\t"))
            continue
        pooled = _find_per_combo_tsv(outputs.data_dir, "pr_curve_pooled", cid)
        if pooled is not None:
            parts.append(pd.read_csv(pooled, sep="\t"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_curves(
    seq_name: str,
    outputs: RunOutputs,
    *,
    df_pooled: pd.DataFrame,
    n_queries: int | None = None,
) -> None:
    save_pooled_plot_data(df_pooled, outputs.plot_data_tsv)
    plot_pooled_recall_vs_q(df_pooled, seq_name, outputs, n_queries=n_queries)
    plot_pooled_precision_vs_q(df_pooled, seq_name, outputs, n_queries=n_queries)
    plot_pooled_pr_from_q(df_pooled, seq_name, outputs, n_queries=n_queries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDR precision–recall curves with q markers (compute + plot)")
    p.add_argument("--seq-name", default=SEQ_NAME)
    p.add_argument("--search-methods", nargs="+", default=None)
    p.add_argument("--decoy-methods", nargs="+", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--plot-only", action="store_true", help="Skip compute, plot existing TSV")
    p.add_argument(
        "--force-load-all",
        action="store_true",
        help="Load full pair table into memory (may OOM on large datasets like astral)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during query scan",
    )
    p.add_argument(
        "--query-list",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Text file with one qid per line (# comments ok). "
            "Compute and pooled metrics average only over this subset."
        ),
    )
    p.add_argument(
        "--grep-extract",
        action="store_true",
        help="grep subset of real/decoy TSVs first (fast on huge files; default with --query-list)",
    )
    p.add_argument(
        "--no-grep-extract",
        action="store_true",
        help="Do not grep-extract even when --query-list is set",
    )
    return p.parse_args()


def main() -> None:
    global SHOW_PROGRESS
    args = parse_args()
    SHOW_PROGRESS = not args.no_progress
    out_dir = args.out_dir or (RESULTS_DIR / args.seq_name)
    search_methods = args.search_methods or sorted({s for s, _ in SEARCH_DECOY_COMBOS})
    decoy_methods = args.decoy_methods or sorted({d for _, d in SEARCH_DECOY_COMBOS})
    query_ids = load_query_list(args.query_list) if args.query_list else None
    use_grep = (bool(query_ids) and not args.no_grep_extract) or args.grep_extract
    query_stem = args.query_list.stem if args.query_list else None
    outputs = RunOutputs.build(
        out_dir,
        args.seq_name,
        search_methods,
        decoy_methods,
        query_list_stem=query_stem,
    )

    if args.plot_only:
        df_pooled = load_pooled_metrics_for_plot(
            outputs, search_methods, decoy_methods, query_ids=query_ids
        )
    else:
        df_pooled, _df_pr_pooled = run_compute(
            args.seq_name,
            search_methods,
            decoy_methods,
            outputs,
            force_load_all=args.force_load_all,
            query_ids=query_ids,
            use_grep_extract=use_grep,
        )

    plot_curves(
        args.seq_name,
        outputs,
        df_pooled=df_pooled,
        n_queries=len(query_ids) if query_ids else None,
    )


if __name__ == "__main__":
    main()
