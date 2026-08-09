import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEARCH_METHOD = "plm"
QUERY_NAME = "putative"
TARGET_NAME = "class_all"
DECOY_METHOD = "extended_shuf"
WEIGHT_METHOD = "AdaptiveBell"
DECOY_SUFFIX = "_shuf"

# A query is "supported" if at least this many noisy reps produce >=1 accepted hit.
MIN_REPS_WITH_HITS = 1

Q_SINGLE = 0.20
Q_LEVELS = np.round(np.arange(0.10, 0.70, 0.10), 2)


# ============================================================
# Helpers
# ============================================================

def remove_decoy_suffix(qid, decoy_suffix):
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def make_pair_table_keep_tid(path_real, path_decoy, decoy_suffix):
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={
        "score": "score_real",
        "homo_type": "homo_type_real",
    })

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(
        lambda x: remove_decoy_suffix(x, decoy_suffix)
    )
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig"], errors="ignore")
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def _get_class_from_name(name):
    if not isinstance(name, str):
        return "Unknown"
    left = name.split(";", 1)[0].strip()
    parts = left.split(".")
    if len(parts) < 2:
        return "Unknown"
    return {"1": "1", "2": "2", "3": "3"}.get(parts[1], "Unknown")


# ============================================================
# Core: FDR threshold + accepted hits (no class voting)
# ============================================================

def _find_threshold(scores_t, scores_d, q):
    n_t = len(scores_t)
    if n_t == 0:
        return None

    cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
    scores_t_sorted = np.sort(scores_t)
    scores_d_sorted = np.sort(scores_d)

    target_count = n_t - np.searchsorted(scores_t_sorted, cand_t, side="left")
    decoy_count = n_d - np.searchsorted(scores_d_sorted, cand_t, side="left") if (n_d := len(scores_d)) else np.zeros_like(cand_t)
    est_fdp = (decoy_count + 1.0) / np.maximum(target_count, 1.0)

    valid = np.where(est_fdp <= q)[0]
    if len(valid) == 0:
        return None

    best_idx = valid[np.argmin(cand_t[valid])]
    return {
        "T_q": float(cand_t[best_idx]),
        "eFDP_at_T": float(est_fdp[best_idx]),
        "n_pass": int(target_count[best_idx]),
    }


def select_accepted_hits(df_qr, q):
    scores_t = df_qr["score_real"].values
    scores_d = df_qr["score_decoy"].values
    thr = _find_threshold(scores_t, scores_d, q)
    if thr is None:
        return thr, pd.DataFrame()

    pass_mask = scores_t >= thr["T_q"]
    df_pass = df_qr.loc[pass_mask, [
        "tid", "score_real", "score_decoy", "homo_type_real",
    ]].copy()
    df_pass = df_pass.sort_values("score_real", ascending=False).reset_index(drop=True)
    df_pass["hit_rank"] = np.arange(1, len(df_pass) + 1)
    thr["n_pass"] = len(df_pass)
    return thr, df_pass


def run_discovery_for_q(df_merge, q, min_reps_with_hits=MIN_REPS_WITH_HITS):
    summary_rows = []
    hit_rows = []

    grouped = df_merge.groupby("qid", sort=False, observed=True)
    for qid, df_q in tqdm(grouped, desc=f"Discovery @ q={q:.2f}"):
        rep_ids = sorted(df_q["rep_id"].unique().tolist())
        per_rep_n_pass = []
        per_rep_Tq = []
        per_rep_efdp = []

        for rep_id in rep_ids:
            df_qr = df_q[df_q["rep_id"] == rep_id]
            thr, df_pass = select_accepted_hits(df_qr, q)
            if thr is None:
                per_rep_n_pass.append(0)
                per_rep_Tq.append(np.nan)
                per_rep_efdp.append(np.nan)
                continue

            per_rep_n_pass.append(thr["n_pass"])
            per_rep_Tq.append(thr["T_q"])
            per_rep_efdp.append(thr["eFDP_at_T"])

            if len(df_pass) > 0:
                df_pass = df_pass.copy()
                df_pass["qid"] = qid
                df_pass["q"] = q
                df_pass["rep_id"] = rep_id
                hit_rows.append(df_pass)

        n_reps_with_hits = sum(1 for n in per_rep_n_pass if n > 0)
        n_accepted_mean = float(np.mean(per_rep_n_pass))
        supported = n_reps_with_hits >= min_reps_with_hits

        summary_rows.append({
            "qid": qid,
            "q": q,
            "discovery_status": "supported" if supported else "no_supported_hit",
            "n_reps_with_hits": n_reps_with_hits,
            "n_reps_total": len(rep_ids),
            "n_accepted_mean": n_accepted_mean,
            "n_accepted_max": int(max(per_rep_n_pass) if per_rep_n_pass else 0),
            "T_q_mean": float(np.nanmean(per_rep_Tq)),
            "eFDP_at_T_mean": float(np.nanmean(per_rep_efdp)),
            "bagel_predicted_class": _get_class_from_name(qid),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_hits = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    return df_summary, df_hits


def run_discovery_q_scan(df_merge, q_levels, min_reps_with_hits=MIN_REPS_WITH_HITS):
    summaries, hits = [], []
    for q in q_levels:
        df_s, df_h = run_discovery_for_q(
            df_merge, q=q, min_reps_with_hits=min_reps_with_hits
        )
        summaries.append(df_s)
        if len(df_h) > 0:
            hits.append(df_h)
    return pd.concat(summaries, ignore_index=True), (
        pd.concat(hits, ignore_index=True) if hits else pd.DataFrame()
    )


# ============================================================
# Plotting
# ============================================================

def plot_discovery_q_scan(df_scan, out_path):
    q_levels = sorted(df_scan["q"].unique())
    n_supported, n_not = [], []

    for q in q_levels:
        df_q = df_scan[np.isclose(df_scan["q"], q)]
        n_supported.append(int((df_q["discovery_status"] == "supported").sum()))
        n_not.append(int((df_q["discovery_status"] == "no_supported_hit").sum()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(q_levels, n_supported, "-o", color="steelblue", label="supported")
    axes[0].plot(q_levels, n_not, "-o", color="gray", label="no_supported_hit")
    axes[0].set_xlabel("q (target FDR)")
    axes[0].set_ylabel("# putative")
    axes[0].set_title("Discovery coverage vs q")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    mean_hits = []
    for q in q_levels:
        df_q = df_scan[
            np.isclose(df_scan["q"], q)
            & (df_scan["discovery_status"] == "supported")
        ]
        mean_hits.append(float(df_q["n_accepted_mean"].mean()) if len(df_q) else np.nan)

    axes[1].plot(q_levels, mean_hits, "-o", color="darkgreen")
    axes[1].set_xlabel("q (target FDR)")
    axes[1].set_ylabel("Mean # accepted hits (supported only)")
    axes[1].set_title("Discovery yield vs q")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Discovery q-scan plot -> {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FDR-controlled putative BAGEL homolog discovery."
    )
    parser.add_argument(
        "--search-method",
        default=SEARCH_METHOD,
        help="Scoring/search method prefix in result_<method>_* files, e.g. plm, tmvec, dhr_postprocess.",
    )
    parser.add_argument("--query-name", default=QUERY_NAME)
    parser.add_argument("--target-name", default=TARGET_NAME)
    parser.add_argument("--decoy-method", default=DECOY_METHOD)
    parser.add_argument("--weight-method", default=WEIGHT_METHOD)
    parser.add_argument("--decoy-suffix", default=DECOY_SUFFIX)
    parser.add_argument("--q-single", type=float, default=Q_SINGLE)
    parser.add_argument(
        "--q-levels",
        nargs="+",
        type=float,
        default=Q_LEVELS.tolist(),
        help="q levels for the scan.",
    )
    parser.add_argument("--min-reps-with-hits", type=int, default=MIN_REPS_WITH_HITS)
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--write-legacy",
        action="store_true",
        help="Also write the old method-less output filenames.",
    )
    return parser.parse_args()


def write_table(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def method_aware_name(prefix, search_method, decoy_method, suffix):
    joiner = "" if suffix.startswith(".") else "_"
    return f"{prefix}_{search_method}_{decoy_method}{joiner}{suffix}"


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.base_dir, "data")
    out_dir = args.out_dir or os.path.join(args.base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    path_real = os.path.join(
        data_dir,
        f"result_{args.search_method}_{args.query_name}_target_noisy_{args.target_name}.txt",
    )
    path_decoy = os.path.join(
        data_dir,
        f"result_{args.search_method}_{args.query_name}_{args.decoy_method}"
        f"_calibrated_gam_{args.weight_method}_{args.target_name}.txt",
    )
    if not os.path.exists(path_real):
        raise FileNotFoundError(f"Missing real score table: {path_real}")
    if not os.path.exists(path_decoy):
        raise FileNotFoundError(f"Missing decoy score table: {path_decoy}")

    print("[INFO] Loading paired score table...")
    print(f"        search_method: {args.search_method}")
    print(f"        real:  {path_real}")
    print(f"        decoy: {path_decoy}")
    df_merge = make_pair_table_keep_tid(path_real, path_decoy, args.decoy_suffix)
    print(f"        rows: {len(df_merge)}")

    # ----- Single q -----
    print(f"\n=== Discovery @ q = {args.q_single} ===")
    df_single, df_hits_single = run_discovery_for_q(
        df_merge,
        q=args.q_single,
        min_reps_with_hits=args.min_reps_with_hits,
    )

    out_single = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery",
            args.search_method,
            args.decoy_method,
            f"q{args.q_single:.2f}.tsv",
        ),
    )
    out_hits_single = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_hits",
            args.search_method,
            args.decoy_method,
            f"q{args.q_single:.2f}.tsv",
        ),
    )
    write_table(df_single, out_single)
    write_table(df_hits_single, out_hits_single)
    print(f"[OK] Discovery summary -> {out_single}")
    print(f"[OK] Accepted hits (long) -> {out_hits_single}")
    if args.write_legacy:
        legacy_single = os.path.join(
            data_dir, f"putative_discovery_{args.decoy_method}_q{args.q_single:.2f}.tsv"
        )
        legacy_hits_single = os.path.join(
            data_dir, f"putative_discovery_hits_{args.decoy_method}_q{args.q_single:.2f}.tsv"
        )
        write_table(df_single, legacy_single)
        write_table(df_hits_single, legacy_hits_single)
        print(f"[OK] Legacy discovery summary -> {legacy_single}")
        print(f"[OK] Legacy accepted hits (long) -> {legacy_hits_single}")

    n_sup = (df_single["discovery_status"] == "supported").sum()
    print(f"\n[INFO] supported: {n_sup} / {len(df_single)}")
    print(df_single["discovery_status"].value_counts())

    # ----- Q-scan -----
    q_levels = np.round(np.array(args.q_levels, dtype=float), 2)
    print(f"\n=== Discovery q-scan: {q_levels.tolist()} ===")
    df_scan, df_hits_scan = run_discovery_q_scan(
        df_merge,
        q_levels,
        min_reps_with_hits=args.min_reps_with_hits,
    )

    out_scan = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_scan",
            args.search_method,
            args.decoy_method,
            ".tsv",
        ),
    )
    out_hits_scan = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_hits_scan",
            args.search_method,
            args.decoy_method,
            ".tsv",
        ),
    )
    write_table(df_scan, out_scan)
    write_table(df_hits_scan, out_hits_scan)
    print(f"[OK] Q-scan summary -> {out_scan}")
    print(f"[OK] Q-scan accepted hits -> {out_hits_scan}")
    if args.write_legacy:
        legacy_scan = os.path.join(data_dir, f"putative_discovery_scan_{args.decoy_method}.tsv")
        legacy_hits_scan = os.path.join(
            data_dir, f"putative_discovery_hits_scan_{args.decoy_method}.tsv"
        )
        write_table(df_scan, legacy_scan)
        write_table(df_hits_scan, legacy_hits_scan)
        print(f"[OK] Legacy Q-scan summary -> {legacy_scan}")
        print(f"[OK] Legacy Q-scan accepted hits -> {legacy_hits_scan}")

    out_png = os.path.join(
        out_dir,
        f"putative_discovery_scan_{args.search_method}_{args.decoy_method}.png",
    )
    plot_discovery_q_scan(df_scan, out_png)

    print("\n[DONE] Discovery complete.")


if __name__ == "__main__":
    main()
