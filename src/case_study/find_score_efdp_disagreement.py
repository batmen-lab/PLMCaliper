import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR))

from config import (
    DATA_DIR,
    DECOY_METHOD,
    FASTA_NAME,
    QUERY_NAME,
    REP_ID,
    RESULTS_DIR,
    SEARCH_METHOD,
    STREAMING_THRESHOLD_GB,
    TARGET_NAME,
    decoy_suffix,
    fasta_path,
    path_calibrated_decoy,
    path_target_noisy,
)
from utils import ( 
    collect_disagreement_pairs_from_merge,
    file_size_gb,
    iter_pair_tables_by_qid,
    load_fasta_dict,
    make_pair_table_keep_tid,
)

OUTPUT_COLUMNS = [
    "qid",
    "tid",
    "rep_id",
    "score_real",
    "score_decoy",
    "real_score",
    "homo_type_real",
    "est_fdp_at_score",
    "query_len",
    "target_len",
    "case_type",
    "score_cutoff",
    "efdp_cutoff",
    "max_score_cutoff",
    "rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find PLM vs eFDP disagreement pairs (score/efdp cutoffs)"
    )
    parser.add_argument("--search-method", default=SEARCH_METHOD)
    parser.add_argument("--query-name", default=QUERY_NAME)
    parser.add_argument("--target-name", default=TARGET_NAME)
    parser.add_argument("--decoy-method", default=DECOY_METHOD)
    parser.add_argument("--score-cutoff", type=float, default=0.5)
    parser.add_argument(
        "--max-score-cutoff",
        type=float,
        default=None,
        help="Upper PLM score for high_score_high_efdp (e.g. 0.8 excludes very high scores)",
    )
    parser.add_argument("--efdp-cutoff", type=float, default=0.5)
    parser.add_argument(
        "--cases",
        choices=("all", "high", "low"),
        default="all",
        help="Which disagreement types to scan (high=high_score_high_efdp only)",
    )
    parser.add_argument("--rep-id", type=int, default=REP_ID, help="Noise rep (default 0)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Top pairs saved per case type (0 = save all)",
    )
    parser.add_argument(
        "--max-pairs-per-query",
        type=int,
        default=50,
        help="Cap disagreement pairs kept per (qid, case); 0 = no cap",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help=f"FASTA for sequence lengths (default: data/{FASTA_NAME}.fa)",
    )
    parser.add_argument(
        "--low-output-name",
        default="pairs_low_score_low_efdp_top500.tsv",
        help="Output TSV filename for low_score_low_efdp case",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--exclude-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


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


def attach_sequence_lengths(df: pd.DataFrame, seq_lookup: dict[str, str]) -> pd.DataFrame:
    len_lookup = {pid: len(seq) for pid, seq in seq_lookup.items()}
    out = df.copy()
    out["query_len"] = out["qid"].astype(str).map(len_lookup)
    out["target_len"] = out["tid"].astype(str).map(len_lookup)
    missing_q = out["query_len"].isna().sum()
    missing_t = out["target_len"].isna().sum()
    if missing_q or missing_t:
        print(
            f"[WARN] Missing lengths: query={missing_q:,}  target={missing_t:,}",
            flush=True,
        )
    return out


def attach_raw_scores(
    df: pd.DataFrame, search_method: str, query_name: str, target_name: str
) -> pd.DataFrame:
    raw_path = DATA_DIR / f"result_{search_method}_{query_name}_{target_name}.txt"
    if not raw_path.exists():
        print(f"[WARN] Raw result file not found: {raw_path} — real_score will be NaN")
        out = df.copy()
        out["real_score"] = float("nan")
        return out

    qids = sorted(df["qid"].astype(str).unique())
    pattern = "|".join(f"^{qid}\t" for qid in qids)
    proc = subprocess.run(
        ["grep", "-E", pattern, str(raw_path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        out = df.copy()
        out["real_score"] = float("nan")
        return out

    with raw_path.open() as fh:
        header = fh.readline().strip().split("\t")

    rows = [dict(zip(header, line.split("\t"), strict=False)) for line in proc.stdout.splitlines()]
    df_raw = pd.DataFrame(rows)
    if "score" not in df_raw.columns:
        out = df.copy()
        out["real_score"] = float("nan")
        return out

    df_raw["score"] = df_raw["score"].astype(float)
    df_raw = df_raw.groupby(["qid", "tid"], as_index=False)["score"].max()
    df_raw = df_raw.rename(columns={"score": "real_score"})
    return df.merge(df_raw[["qid", "tid", "real_score"]], on=["qid", "tid"], how="left")


def rank_and_save(df: pd.DataFrame, case_type: str, top_n: int, out_path: Path) -> None:
    sub = df[df["case_type"] == case_type].copy()
    if sub.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out_path, sep="\t", index=False)
        print(f"[WARN] No pairs for {case_type}")
        return

    if case_type == "high_score_high_efdp":
        sub = sub.copy()
        sub["_score_band"] = sub["score_real"].round(2)
        sub["_total_len"] = (
            sub["query_len"].fillna(0).astype(int)
            + sub["target_len"].fillna(0).astype(int)
        )
        sub = sub.sort_values(
            ["_score_band", "_total_len", "score_real", "est_fdp_at_score"],
            ascending=[False, False, False, False],
        )
        sub = sub.drop(columns=["_score_band", "_total_len"])
    else:
        sub = sub.sort_values(
            ["est_fdp_at_score", "score_real"],
            ascending=[True, True],
        )
    if top_n > 0:
        sub = sub.head(top_n)
    sub = sub.reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)
    cols = [c for c in OUTPUT_COLUMNS if c in sub.columns]
    sub[cols].to_csv(out_path, sep="\t", index=False)
    print(f"[OK] {case_type}: {len(sub):,} -> {out_path}")


def main() -> None:
    args = parse_args()
    suffix = decoy_suffix(args.decoy_method)
    path_real = path_target_noisy(args.search_method, args.query_name, args.target_name)
    path_decoy = path_calibrated_decoy(
        args.search_method, args.query_name, args.decoy_method, args.target_name
    )
    for p in (path_real, path_decoy):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    tag = f"{args.search_method}_{args.query_name}_{args.decoy_method}"
    out_suffix = f"disagreement_s{args.score_cutoff}_e{args.efdp_cutoff}"
    if args.max_score_cutoff is not None:
        out_suffix += f"_max{args.max_score_cutoff}"
    out_dir = args.out_dir or (RESULTS_DIR / tag / out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_types = None
    if args.cases == "high":
        case_types = {"high_score_high_efdp"}
    elif args.cases == "low":
        case_types = {"low_score_low_efdp"}

    score_hi_msg = f"<={args.max_score_cutoff}" if args.max_score_cutoff is not None else "unbounded"
    normalize_scores = args.search_method == "dhr_postprocess"
    print(
        f"[disagreement] cases={args.cases}  "
        f"score in [{args.score_cutoff}, {score_hi_msg}] vs eFDP>={args.efdp_cutoff}  "
        f"(rep_id={args.rep_id})"
    )
    if normalize_scores:
        print("[disagreement] dhr_postprocess: per-query min-max score normalization")

    rows: list[dict] = []
    n_seen = 0
    rep_ids = [args.rep_id]
    collect_kw = dict(
        score_cutoff=args.score_cutoff,
        efdp_cutoff=args.efdp_cutoff,
        max_score_cutoff=args.max_score_cutoff,
        case_types=case_types,
        rep_ids=rep_ids,
        exclude_self=args.exclude_self,
        max_pairs_per_query=args.max_pairs_per_query,
    )

    if file_size_gb(path_real) > STREAMING_THRESHOLD_GB:
        print(f"[disagreement] Streaming ({file_size_gb(path_real):.1f} GB)")
        for _qid, df_merge in iter_pair_tables_by_qid(path_real, path_decoy, suffix):
            if args.max_queries is not None and n_seen >= args.max_queries:
                break
            n_seen += 1
            if n_seen % 500 == 0:
                print(f"        queries: {n_seen:,}  pairs: {len(rows):,}", flush=True)
            if normalize_scores:
                df_merge = minmax_normalize_pair_scores(df_merge)
            rows.extend(collect_disagreement_pairs_from_merge(df_merge, **collect_kw))
    else:
        print(f"[disagreement] Loading {path_real.name}")
        df_merge = make_pair_table_keep_tid(path_real, path_decoy, suffix)
        if args.max_queries is not None:
            qids = df_merge["qid"].unique()[: args.max_queries]
            df_merge = df_merge[df_merge["qid"].isin(qids)]
        if normalize_scores:
            df_merge = minmax_normalize_pair_scores(df_merge)
        rows = collect_disagreement_pairs_from_merge(df_merge, **collect_kw)
        n_seen = df_merge["qid"].nunique()

    print(f"        queries scanned: {n_seen:,}")
    print(f"        disagreement pairs: {len(rows):,}")

    if not rows:
        print("[WARN] No disagreement pairs found.")
        return

    fa_path = args.fasta or fasta_path(FASTA_NAME)
    if not fa_path.exists():
        raise FileNotFoundError(f"Missing FASTA for lengths: {fa_path}")
    print(f"[disagreement] Loading sequence lengths from {fa_path.name}")
    seq_lookup = load_fasta_dict(fa_path)

    df_all = attach_sequence_lengths(pd.DataFrame(rows), seq_lookup)

    n_before = len(df_all)
    df_all = df_all[(df_all["query_len"] > 100) & (df_all["target_len"] > 100)]
    print(f"[disagreement] Length filter (>100aa): {n_before:,} -> {len(df_all):,} pairs")

    df_all = attach_raw_scores(df_all, args.search_method, args.query_name, args.target_name)

    # Reclassify case_type using real (non-noisy) score
    real = df_all["real_score"].to_numpy(dtype=np.float64)
    efdp = df_all["est_fdp_at_score"].to_numpy(dtype=np.float64)
    high = (real >= args.score_cutoff) & (efdp >= args.efdp_cutoff)
    if args.max_score_cutoff is not None:
        high &= real <= args.max_score_cutoff
    low = (real < args.score_cutoff) & (efdp < args.efdp_cutoff)
    keep = high | low
    n_before_reclassify = len(df_all)
    df_all = df_all[keep].copy()
    df_all["case_type"] = np.where(high[keep], "high_score_high_efdp", "low_score_low_efdp")
    print(
        f"[disagreement] Real-score reclassify: {n_before_reclassify:,} -> {len(df_all):,} pairs"
    )

    summary = (
        df_all.groupby("case_type", as_index=False)
        .agg(n_pairs=("tid", "count"), n_queries=("qid", "nunique"))
        .sort_values("case_type")
    )
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    df_all.to_csv(out_dir / "pairs_all.tsv", sep="\t", index=False)
    print(f"[OK] summary -> {out_dir / 'summary.tsv'}")

    if args.cases in ("all", "high"):
        rank_and_save(
            df_all,
            "high_score_high_efdp",
            args.top_n,
            out_dir / f"pairs_high_score_high_efdp_top{args.top_n}.tsv",
        )
    if args.cases in ("all", "low"):
        rank_and_save(
            df_all,
            "low_score_low_efdp",
            args.top_n,
            out_dir / args.low_output_name,
        )
    print(f"[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()
