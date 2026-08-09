#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLOT_PKG_DIR = Path(__file__).resolve().parents[1]   # src/plot
PROJECT_ROOT = PLOT_PKG_DIR.parents[1]
CORE_DIR = PROJECT_ROOT / "src" / "core"
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(PLOT_PKG_DIR))

from fdr import ( 
    aggregate_plot_summary,
    compute_tda_fdr_per_query,
    summarize_fdr_curves_at_q_levels,
)
from paths import experiment_dir  

DECOY_METHOD = "extended_shuf" 
DECOY_SUFFIX = "_shuf"   
FILE_PREFIX = "NoCalib_"    


def _resolve(stem: str) -> Path:
    for suffix in (".txt", ".txt.gz"):
        p = DATA_DIR / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{DATA_DIR}/{stem}.txt[.gz] not found")


def compute_one(search_method: str, query_name: str, target_name: str,
                decoy_method: str = DECOY_METHOD, decoy_suffix: str = DECOY_SUFFIX,
                q_levels=None) -> dict:
    real_path = _resolve(f"result_{search_method}_{query_name}_{target_name}")
    decoy_path = _resolve(
        f"result_{search_method}_{query_name}_{decoy_method}_{target_name}"
    )
    print(f"  real : {real_path.name}")
    print(f"  decoy: {decoy_path.name}")
    print("  loading (pandas C parser, gz-aware)...", flush=True)

    real = pd.read_csv(
        real_path, sep="\t", usecols=["qid", "tid", "score", "homo_type"],
        dtype={"score": "float32", "homo_type": "int16"},
    ).rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
    decoy = pd.read_csv(
        decoy_path, sep="\t", usecols=["qid", "tid", "score"],
        dtype={"score": "float32"},
    ).rename(columns={"score": "score_decoy"})

    decoy["qid"] = decoy["qid"].str.slice(0, -len(decoy_suffix))
    print(f"  real rows={len(real):,}  decoy rows={len(decoy):,}; merging on (qid,tid)...",
          flush=True)

    m = real.merge(decoy, on=["qid", "tid"], how="inner")
    del real, decoy
    if m.empty:
        raise RuntimeError(f"no overlapping (qid,tid) pairs for {search_method}")
    m["homo_type_real"] = m["homo_type_real"].astype("int8")
    m["rep_id"] = np.int8(0)
    print(f"  merged pairs={len(m):,}; per-query TDA curves...", flush=True)

    summaries: list[pd.DataFrame] = []
    n = 0
    for _qid, g in m.groupby("qid", sort=False):
        curve = compute_tda_fdr_per_query(g)
        summaries.append(summarize_fdr_curves_at_q_levels(curve, q_levels))
        n += 1
        if n % 3000 == 0:
            print(f"    ... {n} queries", flush=True)

    df_summary = pd.concat(summaries, ignore_index=True)
    print(f"  processed {n} queries")
    return aggregate_plot_summary(df_summary)


def write_csvs(agg: dict, out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_df = pd.DataFrame({
        "q": agg["q_levels"],
        "mean_real_fdr": agg["mean_real_fdr"],
        "mean_power": agg["mean_power"],
        "mean_accuracy": agg.get("mean_accuracy", [float("nan")] * len(agg["q_levels"])),
        "valid_count": agg["valid_count"],
    })
    agg_df["total_curves"] = agg["total_curves"]
    agg_df["n_unique_qids"] = agg["n_unique_qids"]
    agg_path = out_dir / f"{FILE_PREFIX}fdr_curve_agg_{tag}.csv"
    agg_df.to_csv(agg_path, index=False)

    box_rows = [
        {"q": float(q), "real_fdp": float(v)}
        for q, vals in zip(agg["q_levels"], agg["all_real_fdps_per_q"])
        for v in vals
    ]
    box_path = out_dir / f"{FILE_PREFIX}fdr_boxpoints_{tag}.csv"
    pd.DataFrame(box_rows, columns=["q", "real_fdp"]).to_csv(box_path, index=False)

    power_rows = [
        {"q": float(q), "power": float(v)}
        for q, vals in zip(agg["q_levels"], agg.get("all_powers_per_q", []))
        for v in vals
    ]
    pd.DataFrame(power_rows, columns=["q", "power"]).to_csv(
        out_dir / f"{FILE_PREFIX}power_boxpoints_{tag}.csv", index=False)

    acc_rows = [
        {"q": float(q), "accuracy": float(v)}
        for q, vals in zip(agg["q_levels"], agg.get("all_accuracy_per_q", []))
        for v in vals
    ]
    pd.DataFrame(acc_rows, columns=["q", "accuracy"]).to_csv(
        out_dir / f"{FILE_PREFIX}accuracy_boxpoints_{tag}.csv", index=False)
    return agg_path, box_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute uncalibrated FDR curves (extended_shuf) -> clean CSVs")
    p.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    p.add_argument("--decoy-method", default=DECOY_METHOD,
                   help="decoy result to use, e.g. extended_shuf or shuf")
    p.add_argument("--decoy-suffix", default=DECOY_SUFFIX,
                   help="decoy single-seq id suffix (shuf and extended_shuf both use _shuf)")
    p.add_argument("--query-name", default="astral")
    p.add_argument("--target-name", default="astral")
    p.add_argument("--out-dir", type=Path, default=experiment_dir("core"),
                   help="Output dir for the clean CSVs (default: data/plot_data/core)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Uncalibrated (NoCalib) FDR curves — extended_shuf decoy")
    print(f"  methods : {args.search_methods}")
    print(f"  out dir : {args.out_dir}")
    print("=" * 60)

    ok = 0
    for search_method in args.search_methods:
        print(f"\n>>> {search_method}")
        try:
            agg = compute_one(search_method, args.query_name, args.target_name,
                              args.decoy_method, args.decoy_suffix)
            tag = f"{search_method}_{args.query_name}_{args.decoy_method}_{args.target_name}"
            agg_path, box_path = write_csvs(agg, args.out_dir, tag)
            ok += 1
            print(f"[OK] -> {agg_path.name}, {box_path.name}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[skip] {search_method}: {exc}")

    print(f"\n[DONE] {ok}/{len(args.search_methods)} methods -> {args.out_dir}")
    print("     Next: plot_fdr_method.py --decoy-methods extended_shuf "
          "--weight-method '' --file-prefix NoCalib_")


if __name__ == "__main__":
    main()
