import argparse
import sys
from itertools import product
from pathlib import Path
import pandas as pd

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parents[1]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "results_final" / "performance" / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "performance" / "data"
sys.path.insert(0, str(CORE_DIR))

from fdr import aggregate_plot_summary


def curve_tag(search_method, query_name, decoy_method, weight_method, target_name) -> str:
    return f"{search_method}_{query_name}_{decoy_method}_{weight_method}_{target_name}"


def plot_summary_stem(search_method, query_name, decoy_method, weight_method, target_name) -> str:
    return (
        f"fdr_plot_summary_{search_method}_{query_name}_{decoy_method}_"
        f"{weight_method}_{target_name}"
    )


def resolve_plot_summary_path(summary_dir, search_method, query_name, decoy_method, weight_method, target_name) -> Path:
    stem = plot_summary_stem(search_method, query_name, decoy_method, weight_method, target_name)
    for suffix in (".txt", ".txt.gz"):
        path = summary_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return summary_dir / f"{stem}.txt"


def load_plot_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing plot summary: {path}")
    if str(path).endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", compression="gzip")
    return pd.read_csv(path, sep="\t")


def write_curve_csvs(agg: dict, out_dir: Path, tag: str) -> tuple[Path, Path]:
    """Write the per-q aggregate CSV and the long-form boxplot-points CSV."""
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
    agg_path = out_dir / f"fdr_curve_agg_{tag}.csv"
    agg_df.to_csv(agg_path, index=False)

    box_rows = []
    for q, vals in zip(agg["q_levels"], agg["all_real_fdps_per_q"]):
        for v in vals:
            box_rows.append({"q": float(q), "real_fdp": float(v)})
    box_path = out_dir / f"fdr_boxpoints_{tag}.csv"
    pd.DataFrame(box_rows, columns=["q", "real_fdp"]).to_csv(box_path, index=False)

    power_rows = []
    for q, vals in zip(agg["q_levels"], agg.get("all_powers_per_q", [])):
        for v in vals:
            power_rows.append({"q": float(q), "power": float(v)})
    power_path = out_dir / f"power_boxpoints_{tag}.csv"
    pd.DataFrame(power_rows, columns=["q", "power"]).to_csv(power_path, index=False)

    acc_rows = []
    for q, vals in zip(agg["q_levels"], agg.get("all_accuracy_per_q", [])):
        for v in vals:
            acc_rows.append({"q": float(q), "accuracy": float(v)})
    acc_path = out_dir / f"accuracy_boxpoints_{tag}.csv"
    pd.DataFrame(acc_rows, columns=["q", "accuracy"]).to_csv(acc_path, index=False)

    return agg_path, box_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate section-1 FDR summaries into clean per-combo plotting CSVs",
    )
    parser.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    parser.add_argument("--decoy-methods", nargs="+", default=["rev", "dplm", "PGmsk25pct"])
    parser.add_argument("--query-name", default="astral")
    parser.add_argument("--target-name", default="astral")
    parser.add_argument("--weight-method", default="AdaptiveBell")
    parser.add_argument(
        "--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR,
        help="Directory with fdr_plot_summary_*.txt (default: results_final/performance/data)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory for the clean plotting CSVs (default: results/performance/data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = list(product(args.search_methods, args.decoy_methods))

    print("=" * 60)
    print("Section-1 FDR curves — compute (summaries -> clean CSVs)")
    print(f"  summary dir : {args.summary_dir}")
    print(f"  out dir     : {args.out_dir}")
    print(f"  combos      : {len(combos)}")
    print("=" * 60)

    ok = 0
    for search_method, decoy_method in combos:
        tag = curve_tag(search_method, args.query_name, decoy_method, args.weight_method, args.target_name)
        try:
            summary_path = resolve_plot_summary_path(
                args.summary_dir, search_method, args.query_name,
                decoy_method, args.weight_method, args.target_name,
            )
            df_summary = load_plot_summary(summary_path)
            agg = aggregate_plot_summary(df_summary)
            agg_path, box_path = write_curve_csvs(agg, args.out_dir, tag)
            ok += 1
            print(f"[OK] {search_method}/{decoy_method}: {summary_path.name} "
                  f"({len(df_summary):,} rows) -> {agg_path.name}, {box_path.name}")
        except FileNotFoundError as exc:
            print(f"[skip] {search_method}/{decoy_method}: {exc}")

    print(f"\n[DONE] wrote CSVs for {ok}/{len(combos)} combos -> {args.out_dir}")
    print("     Next: python plot_fdr_method.py   (per-combo figures)")
    print("           python plot_fdr_compare.py  (cross-method overlays)")


if __name__ == "__main__":
    main()
