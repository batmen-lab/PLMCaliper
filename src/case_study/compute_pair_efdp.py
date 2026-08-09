import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR))

from config import (  # noqa: E402
    QUERY_NAME,
    REP_ID,
    RESULTS_DIR,
    TARGET_NAME,
    decoy_suffix,
    path_calibrated_decoy,
    path_target_noisy,
)
from utils import (  # noqa: E402
    _rows_to_pair_frame,
    est_fdp_at_score,
    file_size_gb,
    load_pair_table_for_qids,
    make_pair_table_keep_tid,
)


def minmax_normalize_pair_scores(df_merge: pd.DataFrame) -> pd.DataFrame:
    def _norm_group(df_q: pd.DataFrame) -> pd.DataFrame:
        raw_real = df_q["score_real"].to_numpy(dtype=np.float64)
        raw_decoy = df_q["score_decoy"].to_numpy(dtype=np.float64)
        real_min = float(np.min(raw_real))
        real_max = float(np.max(raw_real))
        real_range = (real_max - real_min) + 1e-8
        out = df_q.copy()
        out["score_real"] = ((raw_real - real_min) / real_range).astype(np.float32)
        out["score_decoy"] = ((raw_decoy - real_min) / real_range).astype(np.float32)
        return out

    if df_merge.empty:
        return df_merge
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrameGroupBy.apply operated on the grouping columns.*",
            category=DeprecationWarning,
        )
        return df_merge.groupby("qid", group_keys=False).apply(_norm_group)


def plot_grep_query_score_real_dists(
    df_merge: pd.DataFrame,
    qids: set[str],
    *,
    search_method: str,
    rep_id: int,
    pair_set: set[tuple[str, str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for qid in sorted(qids):
        df_q = df_merge[(df_merge["qid"] == qid) & (df_merge["rep_id"] == rep_id)]
        if df_q.empty:
            print(f"[WARN] No rows for qid={qid!r}, skip score dist plot")
            continue

        scores = df_q["score_real"].to_numpy(dtype=np.float64)
        highlight_tids = {tid for q, tid in pair_set if q == qid}

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=50, alpha=0.75, color="#4C72B0", edgecolor="white", linewidth=0.2)
        for tid in sorted(highlight_tids):
            hit = df_q[df_q["tid"].astype(str) == tid]
            if hit.empty:
                continue
            score = float(hit["score_real"].iloc[0])
            ax.axvline(
                score,
                color="#C44E52",
                linestyle="--",
                linewidth=1.2,
                label=f"{tid} ({score:.4f})",
            )

        ax.set_title(f"{search_method} — {qid}")
        ax.set_xlabel("score_real")
        ax.set_ylabel("pair count")
        if highlight_tids:
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()

        out_path = out_dir / f"{qid}_score_real_dist.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] score_real dist -> {out_path}")


def parse_pair_token(token: str) -> tuple[str, str]:
    qid, tid = token.strip().split("-", 1)
    return qid, tid


def _grep_qid_rows(path: Path, qids: set[str], *, decoy_suffix: str = "") -> dict[str, list[dict]]:
    if not qids:
        return {}
    pattern = "|".join(f"^{qid}{decoy_suffix}\t" for qid in sorted(qids))
    proc = subprocess.run(
        ["grep", "-E", pattern, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return {}
    with path.open() as handle:
        header = handle.readline().strip().split("\t")
    grouped: dict[str, list[dict]] = {qid: [] for qid in qids}
    for line in proc.stdout.splitlines():
        values = line.split("\t")
        row = dict(zip(header, values, strict=False))
        qid = row["qid"]
        if decoy_suffix and qid.endswith(decoy_suffix):
            qid = qid[: -len(decoy_suffix)]
        if qid in grouped:
            grouped[qid].append(row)
    return grouped


def _load_pair_table_for_qids_fast(
    path_real: Path,
    path_decoy: Path,
    decoy_suffix: str,
    qids: set[str],
) -> pd.DataFrame:
    real_by_qid = _grep_qid_rows(path_real, qids)
    decoy_by_qid = _grep_qid_rows(path_decoy, qids, decoy_suffix=decoy_suffix)
    chunks: list[pd.DataFrame] = []
    for qid in sorted(qids):
        df_q = _rows_to_pair_frame(real_by_qid.get(qid, []), decoy_by_qid.get(qid, []), decoy_suffix)
        if not df_q.empty:
            chunks.append(df_q)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def lookup_pairs_efdp(
    pairs: list[tuple[str, str]],
    *,
    search_method: str,
    query_name: str = QUERY_NAME,
    target_name: str = TARGET_NAME,
    decoy_method: str,
    rep_id: int = REP_ID,
) -> pd.DataFrame:
    suffix = decoy_suffix(decoy_method)
    path_real = path_target_noisy(search_method, query_name, target_name)
    path_decoy = path_calibrated_decoy(search_method, query_name, decoy_method, target_name)
    for p in (path_real, path_decoy):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    qids = {qid for qid, _ in pairs}
    pair_set = {(qid, tid) for qid, tid in pairs}

    print(
        f"[lookup] method={search_method}  decoy={decoy_method}  "
        f"rep_id={rep_id}  pairs={len(pairs)}  queries={len(qids)}"
    )
    print(f"[lookup] real:  {path_real}")
    print(f"[lookup] decoy: {path_decoy}")

    if file_size_gb(path_real) > 1.0:
        print("[lookup] Grep-extracting selected queries...")
        df_merge = _load_pair_table_for_qids_fast(path_real, path_decoy, suffix, qids)
        if df_merge.empty:
            print("[lookup] Grep found nothing; falling back to full streaming...")
            df_merge = load_pair_table_for_qids(path_real, path_decoy, suffix, qids)
    else:
        print("[lookup] Loading pair table...")
        df_merge = make_pair_table_keep_tid(path_real, path_decoy, suffix)
        df_merge = df_merge[df_merge["qid"].isin(qids)]

    if df_merge.empty:
        raise ValueError(f"No pair rows found for queries: {sorted(qids)}")

    normalize_scores = search_method == "dhr_postprocess"
    if normalize_scores:
        print("[lookup] dhr_postprocess: per-query min-max score normalization")
        df_merge = minmax_normalize_pair_scores(df_merge)

    tmp_dir = RESULTS_DIR / "tmp" / "pair_efdp" / search_method
    plot_grep_query_score_real_dists(
        df_merge,
        qids,
        search_method=search_method,
        rep_id=rep_id,
        pair_set=pair_set,
        out_dir=tmp_dir,
    )

    from utils import _compute_tda_fdr_per_query  # noqa: E402

    rows: list[dict] = []
    for (qid, rid), df_q in df_merge.groupby(["qid", "rep_id"], sort=False):
        if int(rid) != rep_id:
            continue
        df_curve = _compute_tda_fdr_per_query(df_q)
        for tid in {t for q, t in pair_set if q == qid}:
            hit = df_q[df_q["tid"].astype(str) == tid]
            if hit.empty:
                rows.append(
                    {
                        "qid": qid,
                        "tid": tid,
                        "rep_id": rep_id,
                        "found": False,
                        "score_real": None,
                        "score_decoy": None,
                        "homo_type_real": None,
                        "est_fdp_at_score": None,
                    }
                )
                continue
            row = hit.iloc[0]
            score = float(row["score_real"])
            rows.append(
                {
                    "qid": qid,
                    "tid": tid,
                    "rep_id": rep_id,
                    "found": True,
                    "score_real": score,
                    "score_decoy": float(row["score_decoy"]),
                    "homo_type_real": int(row["homo_type_real"]),
                    "est_fdp_at_score": est_fdp_at_score(df_curve, score),
                }
            )

    out = pd.DataFrame(rows)
    order = {(q, t): i for i, (q, t) in enumerate(pairs)}
    out["_order"] = out.apply(lambda r: order.get((r["qid"], r["tid"]), 9999), axis=1)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


if __name__ == "__main__":
    # --- parameters ---
    SEARCH_METHOD = "dhr_postprocess"
    DECOY_METHOD = "extended_mkv2"
    REP_ID_MAIN = 0
    QUERY_NAME_MAIN = QUERY_NAME
    TARGET_NAME_MAIN = TARGET_NAME
    OUT_PATH = None  # e.g. Path("results_final/case_study/pair_efdp_lookup.tsv")

    PAIR_TOKENS = [
        "d3nhea_-d1ho8a_",
        "d6lona_-d1hhsa_",
    ]
    # ------------------

    pairs = [parse_pair_token(tok) for tok in PAIR_TOKENS]
    df = lookup_pairs_efdp(
        pairs,
        search_method=SEARCH_METHOD,
        query_name=QUERY_NAME_MAIN,
        target_name=TARGET_NAME_MAIN,
        decoy_method=DECOY_METHOD,
        rep_id=REP_ID_MAIN,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print()
    print(df.to_string(index=False))

    if OUT_PATH is not None:
        out_path = Path(OUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\n[OK] saved -> {out_path}")
