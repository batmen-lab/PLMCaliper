import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WEIGHT_METHOD = "AdaptiveBell"

METHOD2TYPE: dict[str, str] = {
    "shuf": "shuf",
    "rev": "rev",
    "mkv1": "mkv1",
    "mkv2": "mkv2",
    "dplm": "dplm",
    "PGmsk25pct": "PGmsk25pct",
    "copy": "copy",
    "extended_dup": "dup",
    "extended_shuf": "shuf",
    "extended_rev": "rev",
    "extended_then_shuf": "shuf",
    "extended_shufpartly40": "shufpartly",
    "extended_mkv1": "mkv",
    "extended_mkv2": "mkv",
}


def _decoy_suffix(decoy_method: str) -> str:
    if decoy_method not in METHOD2TYPE:
        raise ValueError(
            f"Unknown decoy method '{decoy_method}'. "
            f"Known: {sorted(METHOD2TYPE)}"
        )
    return f"_{METHOD2TYPE[decoy_method]}"


def _path_raw(search: str, query: str, target: str, data_dir: Path) -> Path:
    return data_dir / f"result_{search}_{query}_{target}.txt"


def _path_target_noisy(search: str, query: str, target: str, data_dir: Path) -> Path:
    return data_dir / f"result_{search}_{query}_target_noisy_{target}.txt"


def _path_calibrated_decoy(
    search: str, query: str, decoy: str, target: str, data_dir: Path
) -> Path:
    return data_dir / (
        f"result_{search}_{query}_{decoy}_calibrated_gam_{WEIGHT_METHOD}_{target}.txt"
    )


# ---------------------------------------------------------------------------
# Grep-based row extraction (avoids loading multi-GB files into memory)
# ---------------------------------------------------------------------------

def _grep_rows(
    path: Path,
    qids: set[str],
    *,
    qid_suffix: str = "",
) -> dict[str, list[dict]]:
    if not qids:
        return {q: [] for q in qids}

    pattern = "|".join(f"^{qid}{qid_suffix}\t" for qid in sorted(qids))
    proc = subprocess.run(
        ["grep", "-E", pattern, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    grouped: dict[str, list[dict]] = {q: [] for q in qids}
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return grouped

    with path.open() as fh:
        header = fh.readline().strip().split("\t")

    suffix_len = len(qid_suffix)
    for line in proc.stdout.splitlines():
        values = line.split("\t")
        row = dict(zip(header, values, strict=False))
        key = row["qid"]
        if qid_suffix and key.endswith(qid_suffix):
            key = key[:-suffix_len]
        if key in grouped:
            grouped[key].append(row)
    return grouped


# ---------------------------------------------------------------------------
# Pair table construction (noisy target + calibrated decoy -> one DataFrame)
# ---------------------------------------------------------------------------

def _build_pair_table(
    noisy_rows: list[dict],
    calibrated_rows: list[dict],
    decoy_suffix: str,
) -> pd.DataFrame:
    if not noisy_rows or not calibrated_rows:
        return pd.DataFrame()

    df_real = pd.DataFrame(noisy_rows)
    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
    df_real["score_real"] = df_real["score_real"].astype(float)
    df_real["rep_id"] = pd.to_numeric(df_real["rep_id"], downcast="integer")

    df_decoy = pd.DataFrame(calibrated_rows)
    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    # qid in the calibrated file is {orig_qid}{suffix}; strip it for joining
    df_decoy["qid_orig"] = df_decoy["qid"].apply(
        lambda x: x[: -len(decoy_suffix)] if x.endswith(decoy_suffix) else x
    )
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
    df_decoy["score_decoy"] = df_decoy["score_decoy"].astype(float)
    df_decoy["rep_id"] = pd.to_numeric(df_decoy["rep_id"], downcast="integer")

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    return df_merge.drop(columns=["qid_orig"], errors="ignore")


# ---------------------------------------------------------------------------
# TDA FDR curve and eFDP lookup
# ---------------------------------------------------------------------------

def _compute_tda_curve(df_q: pd.DataFrame) -> pd.DataFrame:
    core_dir = Path(__file__).resolve().parent
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    from fdr import compute_tda_fdr_per_query  # type: ignore[import]
    return compute_tda_fdr_per_query(df_q)


def _est_fdp_at_score(df_curve: pd.DataFrame, score: float) -> float:
    if df_curve.empty:
        return float("nan")
    thr = df_curve["threshold"].to_numpy(dtype=np.float64)
    efdp = df_curve["est_fdp"].to_numpy(dtype=np.float64)
    idx = int(np.searchsorted(thr, score, side="right") - 1)
    return float(efdp[max(idx, 0)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_pair_scores(
    pairs: list[tuple[str, str]],
    *,
    search_method: str,
    query_name: str,
    target_name: str,
    decoy_method: str,
    rep_id: int = 0,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:

    suffix = _decoy_suffix(decoy_method)

    path_raw = _path_raw(search_method, query_name, target_name, data_dir)
    path_noisy = _path_target_noisy(search_method, query_name, target_name, data_dir)
    path_cal = _path_calibrated_decoy(search_method, query_name, decoy_method, target_name, data_dir)

    for p in (path_noisy, path_cal):
        if not p.exists():
            raise FileNotFoundError(f"Missing calibration file: {p}")

    raw_exists = path_raw.exists()
    if not raw_exists:
        print(f"[WARN] Raw result file not found: {path_raw}  (real_score will be NaN)")

    qids = {qid for qid, _ in pairs}
    pair_set = {(qid, tid) for qid, tid in pairs}

    print(f"[lookup] search={search_method}  decoy={decoy_method}  "
          f"suffix={suffix!r}  rep_id={rep_id}")
    print(f"[lookup] pairs={len(pairs)}  unique queries={len(qids)}")
    print(f"[lookup] noisy  : {path_noisy}")
    print(f"[lookup] cal    : {path_cal}")
    print(f"[lookup] raw    : {path_raw}")

    # --- Grep noisy target and calibrated decoy rows for relevant qids ---
    print("[lookup] Extracting noisy-target rows via grep...")
    noisy_by_qid = _grep_rows(path_noisy, qids)

    print("[lookup] Extracting calibrated-decoy rows via grep...")
    cal_by_qid = _grep_rows(path_cal, qids, qid_suffix=suffix)

    # --- Build per-query pair tables ---
    chunks: list[pd.DataFrame] = []
    for qid in sorted(qids):
        df_q = _build_pair_table(noisy_by_qid.get(qid, []), cal_by_qid.get(qid, []), suffix)
        if not df_q.empty:
            chunks.append(df_q)

    if not chunks:
        raise ValueError(
            f"No pair rows found in noisy/calibrated files for queries: {sorted(qids)}. "
            f"Check that calibration has been run."
        )

    df_merge = pd.concat(chunks, ignore_index=True)

    # --- Grep real (non-noisy) scores from raw result file ---
    real_score_lookup: dict[tuple[str, str], float] = {}
    if raw_exists:
        print("[lookup] Extracting raw (non-noisy) scores via grep...")
        raw_by_qid = _grep_rows(path_raw, qids)
        for rows in raw_by_qid.values():
            for row in rows:
                key = (str(row["qid"]), str(row["tid"]))
                real_score_lookup[key] = float(row["score"])

    # --- Compute eFDP for each requested pair ---
    rows_out: list[dict] = []
    processed_pairs: set[tuple[str, str]] = set()

    for (qid_g, rid_g), df_q in df_merge.groupby(["qid", "rep_id"], sort=False):
        if int(rid_g) != rep_id:
            continue
        df_curve = _compute_tda_curve(df_q)

        for qid_p, tid_p in pair_set:
            if str(qid_p) != str(qid_g):
                continue
            processed_pairs.add((qid_p, tid_p))

            real_score = real_score_lookup.get((str(qid_p), str(tid_p)), float("nan"))
            hit = df_q[df_q["tid"].astype(str) == str(tid_p)]

            if hit.empty:
                rows_out.append({
                    "qid": qid_p,
                    "tid": tid_p,
                    "rep_id": rep_id,
                    "real_score": real_score,
                    "noise_target_score": float("nan"),
                    "score_decoy_cal": float("nan"),
                    "homo_type": None,
                    "efdp": float("nan"),
                    "found_in_noisy": False,
                    "found_in_raw": not np.isnan(real_score),
                })
            else:
                r = hit.iloc[0]
                noise_score = float(r["score_real"])
                rows_out.append({
                    "qid": qid_p,
                    "tid": tid_p,
                    "rep_id": rep_id,
                    "real_score": real_score,
                    "noise_target_score": noise_score,
                    "score_decoy_cal": float(r["score_decoy"]),
                    "homo_type": (
                        int(r["homo_type_real"])
                        if pd.notna(r["homo_type_real"])
                        else None
                    ),
                    "efdp": _est_fdp_at_score(df_curve, noise_score),
                    "found_in_noisy": True,
                    "found_in_raw": not np.isnan(real_score),
                })

    # Pairs whose query block was absent in the noisy/calibrated files
    for qid_p, tid_p in pair_set:
        if (qid_p, tid_p) not in processed_pairs:
            real_score = real_score_lookup.get((str(qid_p), str(tid_p)), float("nan"))
            rows_out.append({
                "qid": qid_p,
                "tid": tid_p,
                "rep_id": rep_id,
                "real_score": real_score,
                "noise_target_score": float("nan"),
                "score_decoy_cal": float("nan"),
                "homo_type": None,
                "efdp": float("nan"),
                "found_in_noisy": False,
                "found_in_raw": not np.isnan(real_score),
            })

    out = pd.DataFrame(rows_out)
    # Restore user-specified ordering
    order = {(q, t): i for i, (q, t) in enumerate(pairs)}
    out["_order"] = out.apply(lambda r: order.get((r["qid"], r["tid"]), len(pairs)), axis=1)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score distribution plots
# ---------------------------------------------------------------------------

SEARCH_METHOD_DISPLAY = {
    "plm": "PLMsearch",
    "tmvec": "TMvec",
    "dhr_postprocess": "DHR",
    "blastp": "BLASTP",
    "dctdomain": "DCTdomain",
}


def plot_score_distributions(
    pairs_df: pd.DataFrame,
    *,
    search_method: str,
    query_name: str,
    target_name: str,
    data_dir: Path,
    plot_dir: Path,
) -> None:
    method_label = SEARCH_METHOD_DISPLAY.get(search_method, search_method)
    path_raw = _path_raw(search_method, query_name, target_name, data_dir)

    if not path_raw.exists():
        print(f"[WARN] Raw result file missing, cannot plot distribution: {path_raw}")
        return

    qids = list(pairs_df["qid"].unique())
    print(f"[plot] Loading raw scores for {len(qids)} queries from {path_raw.name} ...")
    all_rows_by_qid = _grep_rows(path_raw, set(qids))

    plot_dir.mkdir(parents=True, exist_ok=True)

    for _, row in pairs_df.iterrows():
        qid   = str(row["qid"])
        tid   = str(row["tid"])
        score = float(row["real_score"])

        rows = all_rows_by_qid.get(qid, [])
        if not rows:
            print(f"[WARN] No raw scores found for query {qid}, skipping pair ({qid}, {tid})")
            continue

        all_scores = np.array([float(r["score"]) for r in rows], dtype=np.float64)

        # --- Figure: square main axes, legend outside to the right ---
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_box_aspect(1)

        ax.hist(all_scores, bins=80, color="steelblue", alpha=0.7, label="All targets")

        if not np.isnan(score):
            y_top = ax.get_ylim()[1]
            ax.axvline(score, color="red", linestyle="--", linewidth=1.5,
                       label=f"Target: {tid}")
            x_range = ax.get_xlim()
            x_offset = (x_range[1] - x_range[0]) * 0.015
            ax.text(
                score + x_offset, y_top * 0.97,
                f"Target: {tid}",
                color="red", fontsize=12,
                rotation=0, va="top", ha="left",
            )
        else:
            print(f"[WARN] real_score is NaN for ({qid}, {tid}), line not drawn")

        ax.set_xlabel("Score", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", fontsize=13, fontweight="bold")
        ax.set_title(f"Query: {qid} ({method_label})", fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            bbox_to_anchor=(1.02, 1), loc="upper left",
            borderaxespad=0, fontsize=8, frameon=True,
        )

        safe_qid = qid.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_tid = tid.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out = plot_dir / f"score_dist_{search_method}_{safe_qid}__{safe_tid}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_pair(token: str, sep: str) -> tuple[str, str]:
    if sep == "-":
        qid, tid = token.strip().split("-", 1)
    else:
        parts = token.strip().split(sep, 1)
        if len(parts) != 2:
            raise ValueError(
                f"Cannot split '{token}' into qid/tid with separator '{sep}'"
            )
        qid, tid = parts
    return qid, tid


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--search-method", required=True,
                   help="Search method: plm | tmvec | dhr_postprocess | blastp | dctdomain")
    p.add_argument("--decoy-method", required=True,
                   help=f"Decoy method, one of: {sorted(METHOD2TYPE)}")
    p.add_argument("--query-name", required=True,
                   help="Query name used in file paths, e.g. astral")
    p.add_argument("--target-name", required=True,
                   help="Target name used in file paths, e.g. astral")
    p.add_argument("--pairs", nargs="+", required=True, metavar="QID-TID",
                   help='Pairs as "qid-tid" tokens (split on first hyphen by default)')
    p.add_argument("--pair-sep", default="-", metavar="SEP",
                   help="Delimiter separating qid from tid in each --pairs token "
                        "(default: first hyphen). Use e.g. ',' for BAGEL-style IDs.")
    p.add_argument("--rep-id", type=int, default=0,
                   help="Noise replicate ID to look up (default: 0)")
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help=f"Data directory (default: {DATA_DIR})")
    p.add_argument("--output", default=None, metavar="TSV",
                   help="Optional output TSV path")
    p.add_argument("--plot-dir", default=None, metavar="DIR",
                   help="If given, save one score-distribution PDF per query into this directory")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    pairs = [_parse_pair(tok, args.pair_sep) for tok in args.pairs]

    df = lookup_pair_scores(
        pairs,
        search_method=args.search_method,
        query_name=args.query_name,
        target_name=args.target_name,
        decoy_method=args.decoy_method,
        rep_id=args.rep_id,
        data_dir=Path(args.data_dir),
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print()
    print(df.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\n[OK] saved -> {out_path}")

    if args.plot_dir:
        plot_score_distributions(
            df,
            search_method=args.search_method,
            query_name=args.query_name,
            target_name=args.target_name,
            data_dir=Path(args.data_dir),
            plot_dir=Path(args.plot_dir),
        )


if __name__ == "__main__":
    SEARCH_METHOD = "plm"
    DECOY_METHOD  = "extended_shuf"
    QUERY_NAME    = "astral"
    TARGET_NAME   = "astral"
    PAIR_TOKENS   = [
        "d1bwda_-d1xkna1",
        "d1bwda_-d4n2ka3",
        "d2i49a_-d4ntla_",
        "d2i49a_-d4prsa_",
        "d4ghga2-d2rk9a_",
        "d4ghga2-d3rmua1",
        "d6qp1a1-d1wyua_",
        "d6qp1a1-d2c81a_",
        "d3g02a1-d4kmra_",
        "d4nqra_-d2d81a1",
    ]
    OUTPUT   = "results_final/case_study/pair_scores.tsv"
    PLOT_DIR = "results_final/case_study/score_dist_plots"
    PLOT     = True   # set False to skip distribution plots

    args = [
        "--search-method", SEARCH_METHOD,
        "--decoy-method",  DECOY_METHOD,
        "--query-name",    QUERY_NAME,
        "--target-name",   TARGET_NAME,
        "--pairs",         *PAIR_TOKENS,
        "--output",        OUTPUT,
    ]
    if PLOT:
        args += ["--plot-dir", PLOT_DIR]
    main(args)
