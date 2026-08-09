import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "results" / "performance" / "data"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "results" / "performance"
sys.path.insert(0, str(CORE_DIR))

from fdr import (
    _create_square_axes,
    _place_external_legend_and_supp,
    _save_current_figure,
)

DECOY_HIGHLIGHT_COLOR = "#E69F00"  # last decoy-method in --decoy-methods

SEARCH_METHOD_DISPLAY = {
    "plm": "PLMsearch",
    "tmvec": "TMvec",
    "dhr_postprocess": "DHR",
}


def search_method_display_name(search_method: str) -> str:
    return SEARCH_METHOD_DISPLAY.get(
        search_method, search_method.replace("_", " ").upper()
    )


def search_method_title_tag(search_methods: list[str]) -> str:
    tags = [search_method_display_name(m) for m in search_methods]
    return f"({', '.join(tags)})"


def compare_plot_title(base_title: str, search_methods: list[str]) -> str:
    return f"{base_title} {search_method_title_tag(search_methods)}"


def _apply_compare_axis_labels(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="normal")


def line_color_for_decoy(decoy_method: str, decoy_methods: list[str]) -> str:
    """Last decoy -> orange; earlier decoys -> light-to-dark blue gradient."""
    if decoy_method == decoy_methods[-1]:
        return DECOY_HIGHLIGHT_COLOR
    palette = decoy_methods[:-1]
    if len(palette) == 1:
        return mcolors.to_hex(plt.cm.Blues(0.62))
    idx = palette.index(decoy_method)
    level = 0.35 + 0.55 * idx / (len(palette) - 1)
    return mcolors.to_hex(plt.cm.Blues(level))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare section-1 FDR, Power, and Valid Count curves across combos",
    )
    parser.add_argument(
        "--search-methods",
        nargs="+",
        default=["plm", "tmvec", "dhr_postprocess"],
        help="Search backends (default: plm tmvec dhr_postprocess)",
    )
    parser.add_argument(
        "--decoy-methods",
        nargs="+",
        default=["rev", "dplm", "PGmsk25pct"],
        help="Decoy types (default: rev dplm PGmsk25pct)",
    )
    parser.add_argument("--query-name", default="astral_s3000_seed123")
    parser.add_argument("--target-name", default="astral_s3000_seed123")
    parser.add_argument("--weight-method", default="AdaptiveBell")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with fdr_curve_agg_*.csv (default: results/performance/data)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory for comparison PDFs (default: results/performance)",
    )
    parser.add_argument(
        "--out-tag",
        default="",
        help="Optional search-method tag for output filenames (default: joined --search-methods)",
    )
    return parser.parse_args()


def curve_tag(search_method, query_name, decoy_method, weight_method, target_name) -> str:
    return f"{search_method}_{query_name}_{decoy_method}_{weight_method}_{target_name}"


def load_curve_agg(data_dir: Path, tag: str) -> dict:
    """Read fdr_curve_agg_<tag>.csv (from compute_fdr_curves.py) into an agg dict."""
    path = data_dir / f"fdr_curve_agg_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} in {data_dir}. Run: python compute_fdr_curves.py"
        )
    df = pd.read_csv(path)
    return {
        "q_levels": df["q"].to_numpy(),
        "mean_real_fdr": df["mean_real_fdr"].to_numpy(),
        "mean_power": df["mean_power"].to_numpy(),
        "valid_count": df["valid_count"].to_numpy(),
        "total_curves": int(df["total_curves"].iloc[0]),
    }


def combo_legend_label(search_method: str, decoy_method: str) -> str:
    return f"{search_method} - {decoy_method}"


def default_out_tag(search_methods: list[str]) -> str:
    return "-".join(search_methods)


def load_combo_aggregates(
    data_dir: Path,
    search_methods: list[str],
    decoy_methods: list[str],
    query_name: str,
    target_name: str,
    weight_method: str,
) -> list[tuple[str, dict, str]]:
    combos: list[tuple[str, dict, str]] = []
    for search_method, decoy_method in product(search_methods, decoy_methods):
        label = combo_legend_label(search_method, decoy_method)
        tag = curve_tag(search_method, query_name, decoy_method, weight_method, target_name)
        print(f"[load] {label} <- fdr_curve_agg_{tag}.csv")
        combos.append((label, load_curve_agg(data_dir, tag), decoy_method))
    if not combos:
        raise FileNotFoundError("No curve CSVs loaded")
    return combos


def plot_fdr_comparison(
    combos: list[tuple[str, dict, str]],
    out_path: Path,
    *,
    decoy_methods: list[str],
    search_methods: list[str],
) -> None:
    fig, ax = _create_square_axes()
    ax.plot([0, 1.0], [0, 1.0], linestyle="--", color="gray", linewidth=2, label="y = x")

    for label, agg, decoy_method in combos:
        color = line_color_for_decoy(decoy_method, decoy_methods)
        ax.plot(
            agg["q_levels"],
            agg["mean_real_fdr"],
            marker="o",
            markersize=5,
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.0)
    _apply_compare_axis_labels(
        ax,
        "Target FDR Level",
        "Real FDR",
        compare_plot_title("FDR Control Curve", search_methods),
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax)
    _save_current_figure(out_path)
    plt.close(fig)
    print(f"[OK] Saved FDR comparison: {out_path}")


def plot_power_comparison(
    combos: list[tuple[str, dict, str]],
    out_path: Path,
    *,
    decoy_methods: list[str],
    search_methods: list[str],
) -> None:
    fig, ax = _create_square_axes()

    for label, agg, decoy_method in combos:
        color = line_color_for_decoy(decoy_method, decoy_methods)
        ax.plot(
            agg["q_levels"],
            agg["mean_power"],
            marker="s",
            markersize=5,
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.0)
    _apply_compare_axis_labels(
        ax,
        "Target FDR Level",
        "Average Power",
        compare_plot_title("Power Curve", search_methods),
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax)
    _save_current_figure(out_path)
    plt.close(fig)
    print(f"[OK] Saved Power comparison: {out_path}")


def plot_valid_count_comparison(
    combos: list[tuple[str, dict, str]],
    out_path: Path,
    *,
    decoy_methods: list[str],
    search_methods: list[str],
) -> None:
    fig, ax = _create_square_axes()

    max_total_curves = 0
    for label, agg, decoy_method in combos:
        color = line_color_for_decoy(decoy_method, decoy_methods)
        ax.plot(
            agg["q_levels"],
            agg["valid_count"],
            marker="^",
            markersize=5,
            color=color,
            linewidth=2,
            label=label,
        )
        max_total_curves = max(max_total_curves, agg["total_curves"])

    if max_total_curves > 0:
        ax.axhline(
            y=max_total_curves,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"Total = {max_total_curves}",
        )

    ax.set_ylim(0, max_total_curves + max_total_curves * 0.05)
    ax.set_xlim(0, 1.0)
    _apply_compare_axis_labels(
        ax,
        "Target FDR Level",
        "Number",
        compare_plot_title("Number of Valid Queries", search_methods),
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, ncol=1)
    _save_current_figure(out_path)
    plt.close(fig)
    print(f"[OK] Saved Valid Count comparison: {out_path}")


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    search_tag = args.out_tag or default_out_tag(args.search_methods)
    file_prefix = f"{args.query_name}_{search_tag}"
    out_fdr = args.plot_dir / f"compare_{file_prefix}_FDR.pdf"
    out_power = args.plot_dir / f"compare_{file_prefix}_Power.pdf"
    out_valid = args.plot_dir / f"compare_{file_prefix}_ValidCount.pdf"

    print("=" * 60)
    print("Section-1 FDR/Power/Valid Count comparison")
    print(f"  data dir    : {args.data_dir}")
    print(f"  plot dir    : {args.plot_dir}")
    print(f"  search      : {args.search_methods}")
    print(f"  decoy       : {args.decoy_methods}")
    print("=" * 60)

    combos = load_combo_aggregates(
        args.data_dir,
        args.search_methods,
        args.decoy_methods,
        args.query_name,
        args.target_name,
        args.weight_method,
    )

    plot_fdr_comparison(
        combos,
        out_fdr,
        decoy_methods=args.decoy_methods,
        search_methods=args.search_methods,
    )
    plot_power_comparison(
        combos,
        out_power,
        decoy_methods=args.decoy_methods,
        search_methods=args.search_methods,
    )
    plot_valid_count_comparison(
        combos,
        out_valid,
        decoy_methods=args.decoy_methods,
        search_methods=args.search_methods,
    )


if __name__ == "__main__":
    main()
