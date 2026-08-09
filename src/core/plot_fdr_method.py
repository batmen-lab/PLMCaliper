import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "results" / "performance" / "data"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "results" / "performance"
sys.path.insert(0, str(CORE_DIR))

from fdr import (
    _apply_bold_labels,
    _create_square_axes,
    _place_external_legend_and_supp,
    _save_current_figure,
    _supp_text,
)


def curve_tag(search_method, query_name, decoy_method, weight_method, target_name) -> str:
    return f"{search_method}_{query_name}_{decoy_method}_{weight_method}_{target_name}"


def figure_tag(search_method, query_name, decoy_method, weight_method) -> str:
    return f"{search_method}_{query_name}_{decoy_method}_{weight_method}"


def load_curve_data(data_dir: Path, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_path = data_dir / f"fdr_curve_agg_{tag}.csv"
    box_path = data_dir / f"fdr_boxpoints_{tag}.csv"
    if not agg_path.exists():
        raise FileNotFoundError(
            f"Missing {agg_path.name} in {data_dir}. Run: python compute_fdr_curves.py"
        )
    agg = pd.read_csv(agg_path)
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame(columns=["q", "real_fdp"])
    return agg, box


def plot_combo(agg: pd.DataFrame, box: pd.DataFrame, plot_dir: Path, fig_tag: str,
               *, title_suffix: str, weight_method: str, tau: float) -> None:
    q_levels = agg["q"].to_numpy()
    supp_text = _supp_text(title_suffix, weight_method, tau)

    # --- Plot 1: Target q vs Real FDR (boxplots + mean) ---
    fig, ax = _create_square_axes()
    ax.plot([0, 1.0], [0, 1.0], linestyle="--", color="gray", linewidth=2, label="y = x")
    for pos in q_levels:
        vals = box.loc[np.isclose(box["q"], pos), "real_fdp"].to_numpy()
        if len(vals) > 0:
            ax.boxplot(
                vals, positions=[pos], widths=0.02, showfliers=True, patch_artist=True,
                boxprops=dict(facecolor="plum", color="purple", alpha=0.3),
                medianprops=dict(color="indigo", linewidth=1.5),
                whiskerprops=dict(color="purple", alpha=0.6),
                capprops=dict(color="purple", alpha=0.6),
                flierprops=dict(marker=".", color="purple", alpha=0.2, markersize=4),
                manage_ticks=False,
            )
    ax.plot(q_levels, agg["mean_real_fdr"], marker="o", markersize=5, color="purple",
            linewidth=2, label="Mean", zorder=5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.0)
    _apply_bold_labels(ax, "Target FDR Level", "Real FDR", "FDR Control Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text)
    out_fdr = plot_dir / f"{fig_tag}_FDR_boxplots.pdf"
    _save_current_figure(out_fdr)
    plt.close(fig)
    print(f"[OK] {out_fdr}")

    # --- Plot 2: Target q vs Power ---
    fig, ax = _create_square_axes()
    ax.plot(q_levels, agg["mean_power"], marker="s", markersize=5, color="darkgreen",
            linewidth=2, label="Power")
    ax.set_ylim(-0.05, 1.05)
    _apply_bold_labels(ax, "Target FDR Level", "Average Power", "Power Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text)
    out_power = plot_dir / f"{fig_tag}_Power.pdf"
    _save_current_figure(out_power)
    plt.close(fig)
    print(f"[OK] {out_power}")

    # --- Plot 3: Target q vs Number of Valid Queries ---
    total_curves = int(agg["total_curves"].iloc[0])
    n_unique_qids = int(agg["n_unique_qids"].iloc[0])
    fig, ax = _create_square_axes()
    ax.plot(q_levels, agg["valid_count"], marker="^", markersize=5, color="teal",
            linewidth=2, label="Valid (Query, Rep) Pairs")
    ax.axhline(
        y=total_curves, color="gray", linestyle=":", linewidth=1.5,
        label=(f"Total (Query x Rep) = {n_unique_qids} x "
               f"{total_curves // max(n_unique_qids, 1)} = {total_curves}"),
    )
    ax.set_ylim(0, total_curves + total_curves * 0.05)
    _apply_bold_labels(ax, "Target FDR Level", "Number", "Number of Valid Queries")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text, ncol=1)
    out_valid = plot_dir / f"{fig_tag}_ValidCount.pdf"
    _save_current_figure(out_valid)
    plt.close(fig)
    print(f"[OK] {out_valid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot section-1 FDR/Power/ValidCount PDFs per (search, decoy) from clean CSVs",
    )
    parser.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    parser.add_argument("--decoy-methods", nargs="+", default=["rev", "dplm", "PGmsk25pct"])
    parser.add_argument("--query-name", default="astral")
    parser.add_argument("--target-name", default="astral")
    parser.add_argument("--weight-method", default="AdaptiveBell")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory with fdr_curve_agg_*.csv (default: results/performance/data)")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help="Output directory for PDFs (default: results/performance)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    combos = list(product(args.search_methods, args.decoy_methods))

    print("=" * 60)
    print("Section-1 per-combo replot (from CSV)")
    print(f"  data dir : {args.data_dir}")
    print(f"  plot dir : {args.plot_dir}")
    print(f"  combos   : {len(combos)}")
    print("=" * 60)

    failed = 0
    for search_method, decoy_method in combos:
        combo = f"{search_method} / {decoy_method}"
        try:
            tag = curve_tag(search_method, args.query_name, decoy_method, args.weight_method, args.target_name)
            agg, box = load_curve_data(args.data_dir, tag)
            fig_tag = figure_tag(search_method, args.query_name, decoy_method, args.weight_method)
            print(f"\n>>> {combo}")
            plot_combo(
                agg, box, args.plot_dir, fig_tag,
                title_suffix=f"({search_method} - {decoy_method})",
                weight_method=args.weight_method, tau=args.tau,
            )
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {combo}: {exc}")

    print("\n" + "=" * 60)
    print(f"Finished {len(combos)} combo(s): {len(combos) - failed} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
