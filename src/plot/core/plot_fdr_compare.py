#!/usr/bin/env python3
import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import experiment_dir
from style import (
    FONT_LABEL,
    FONT_TITLE,
    LINEWIDTH,
    MARKERSIZE,
    create_square_axes,
    place_external_legend,
    save_figure,
)

DEFAULT_DATA_DIR = experiment_dir("core")

DECOY_HIGHLIGHT_COLOR = "#E69F00"  
SEARCH_METHOD_DISPLAY = {"plm": "PLMsearch", "tmvec": "TMvec", "dhr_postprocess": "DHR",
                         "blastp_postprocessed": "BLASTp"}


def search_display(method: str) -> str:
    return SEARCH_METHOD_DISPLAY.get(method, method.replace("_", " ").upper())


def compare_title(base_title: str, search_methods: list[str]) -> str:
    tags = ", ".join(search_display(m) for m in search_methods)
    return f"{base_title} ({tags})"


def apply_axis_labels(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")


def line_color_for_decoy(decoy: str, decoy_methods: list[str]) -> str:
    if decoy == decoy_methods[-1]:
        return DECOY_HIGHLIGHT_COLOR
    palette = decoy_methods[:-1]
    if len(palette) == 1:
        return mcolors.to_hex(plt.cm.Blues(0.62))
    idx = palette.index(decoy)
    level = 0.35 + 0.55 * idx / (len(palette) - 1)
    return mcolors.to_hex(plt.cm.Blues(level))


def curve_tag(search, query, decoy, weight, target) -> str:
    return f"{search}_{query}_{decoy}_{weight}_{target}"


def load_curve_agg(data_dir: Path, tag: str) -> dict:
    path = data_dir / f"fdr_curve_agg_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name} in {data_dir}")
    df = pd.read_csv(path)
    return {
        "q_levels": df["q"].to_numpy(),
        "mean_real_fdr": df["mean_real_fdr"].to_numpy(),
        "mean_power": df["mean_power"].to_numpy(),
        "valid_count": df["valid_count"].to_numpy(),
        "total_curves": int(df["total_curves"].iloc[0]),
    }


def load_combos(data_dir, search_methods, decoy_methods, query, target, weight):
    combos: list[tuple[str, dict, str]] = []
    for search, decoy in product(search_methods, decoy_methods):
        tag = curve_tag(search, query, decoy, weight, target)
        print(f"[load] {search} - {decoy} <- fdr_curve_agg_{tag}.csv")
        combos.append((f"{search} - {decoy}", load_curve_agg(data_dir, tag), decoy))
    if not combos:
        raise FileNotFoundError("No curve CSVs loaded")
    return combos


def _plot_metric(combos, out_path, *, key, marker, ylabel, title, decoy_methods,
                 search_methods, diagonal, axhline_total) -> None:
    fig, ax = create_square_axes()
    if diagonal:
        ax.plot([0, 1.0], [0, 1.0], linestyle="--", color="gray", linewidth=LINEWIDTH, label="y = x")

    linestyles = ["-", "--", "-.", ":"]
    max_total = 0
    for i, (label, agg, decoy) in enumerate(combos):
        color = line_color_for_decoy(decoy, decoy_methods)
        ax.plot(agg["q_levels"], agg[key], marker=marker, markersize=MARKERSIZE, color=color,
                linewidth=LINEWIDTH, linestyle=linestyles[i % len(linestyles)], label=label)
        max_total = max(max_total, agg["total_curves"])

    if axhline_total and max_total > 0:
        ax.axhline(y=max_total, color="gray", linestyle=":", linewidth=LINEWIDTH,
                   label=f"Total = {max_total}")
        ax.set_ylim(0, max_total + max_total * 0.05)
    else:
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.0)
    apply_axis_labels(ax, "Target FDR Level", ylabel, compare_title(title, search_methods))
    ax.grid(True, linestyle="--", alpha=0.4)
    place_external_legend(ax, fig, ncol=1 if axhline_total else 1)
    save_figure(out_path)
    plt.close(fig)
    print(f"[OK] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    p.add_argument("--decoy-methods", nargs="+",
                   default=["shuf", "rev", "dplm", "PGmsk25pct", "extended_shuf"])
    p.add_argument("--query-name", default="astral")
    p.add_argument("--target-name", default="astral")
    p.add_argument("--weight-method", default="AdaptiveBell")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_DATA_DIR / "figures")
    p.add_argument("--out-tag", default="", help="Override search-method tag in output names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    search_tag = args.out_tag or "-".join(args.search_methods)
    prefix = f"{args.query_name}_{search_tag}"
    print(f"[info] data={args.data_dir}  out={args.plot_dir}")

    combos = load_combos(args.data_dir, args.search_methods, args.decoy_methods,
                         args.query_name, args.target_name, args.weight_method)

    _plot_metric(combos, args.plot_dir / f"compare_{prefix}_FDR.pdf",
                 key="mean_real_fdr", marker="o", ylabel="Real FDR",
                 title="FDR Control Curve", decoy_methods=args.decoy_methods,
                 search_methods=args.search_methods, diagonal=True, axhline_total=False)
    _plot_metric(combos, args.plot_dir / f"compare_{prefix}_Power.pdf",
                 key="mean_power", marker="s", ylabel="Average Power",
                 title="Power Curve", decoy_methods=args.decoy_methods,
                 search_methods=args.search_methods, diagonal=False, axhline_total=False)
    _plot_metric(combos, args.plot_dir / f"compare_{prefix}_ValidCount.pdf",
                 key="valid_count", marker="^", ylabel="Number",
                 title="Number of Valid Queries", decoy_methods=args.decoy_methods,
                 search_methods=args.search_methods, diagonal=False, axhline_total=True)


if __name__ == "__main__":
    main()
