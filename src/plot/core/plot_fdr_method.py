#!/usr/bin/env python3
import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import experiment_dir
from style import (
    FONT_ANNOT,
    LINEWIDTH,
    MARKERSIZE,
    apply_bold_labels,
    create_square_axes,
    place_external_legend,
    save_figure,
)

DEFAULT_DATA_DIR = experiment_dir("core")

SEARCH_METHOD_DISPLAY = {"plm": "PLMsearch", "tmvec": "TMvec", "dhr_postprocess": "DHR",
                         "blastp_postprocessed": "BLASTp"}


def search_display(method: str) -> str:
    return SEARCH_METHOD_DISPLAY.get(method, method.replace("_", " ").upper())

FDR_BAR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fdr_mako",
    ["#0b0405", "#28192e", "#3b2f5f", "#3f4a8f", "#366ca0",
     "#348da7", "#39acac", "#59ccad", "#a9e1bd"],
)


def _threshold_bar_colors(n: int) -> list:
    if n <= 1:
        return [FDR_BAR_CMAP(1.0)]
    return [FDR_BAR_CMAP(i / (n - 1)) for i in range(n)]


def _mean_and_ci95(vals: np.ndarray) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0
    mean = float(np.mean(vals))
    if n == 1:
        return mean, 0.0
    sem = float(np.std(vals, ddof=1)) / np.sqrt(n)
    return mean, 1.96 * sem


def _draw_metric_bars(ax, q_levels, values, *, value_fmt, ci_half=None,
                      label_rotation=0, gap=None) -> float:
    values = np.asarray(values, dtype=float)
    x = np.arange(len(q_levels))
    colors = _threshold_bar_colors(len(q_levels))
    ax.bar(x, values, width=0.82, color=colors, edgecolor="none", zorder=2)

    if ci_half is not None:
        ci_half = np.asarray(ci_half, dtype=float)
        ax.errorbar(x, values, yerr=ci_half, fmt="none", ecolor="black",
                    elinewidth=LINEWIDTH, capsize=0, zorder=4, label="95% CI")
        tops = values + ci_half
    else:
        tops = values

    top = float(np.nanmax(tops)) if len(tops) else 1.0
    g = gap if gap is not None else 0.015 * max(top, 1.0)
    for xi, v, t in zip(x, values, tops):
        if np.isnan(v):
            continue
        ax.text(xi, t + g, value_fmt(v), ha="center", va="bottom",
                rotation=label_rotation, fontsize=FONT_ANNOT - 1,
                fontweight="bold", color="black", zorder=5)

    ax.set_xlim(-0.6, len(q_levels) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.2f}" for q in q_levels], rotation=45, ha="right")
    ax.set_axisbelow(True)
    return top


def curve_tag(search, query, decoy, weight, target) -> str:
    parts = [search, query, decoy] + ([weight] if weight else []) + [target]
    return "_".join(parts)


def figure_tag(search, query, decoy, weight) -> str:
    parts = [search, query, decoy] + ([weight] if weight else [])
    return "_".join(parts)


def filter_q_step(agg: pd.DataFrame, box: pd.DataFrame, q_step: float | None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not q_step:
        return agg, box
    ratio = agg["q"].to_numpy() / q_step
    mask = np.isclose(ratio, np.round(ratio), atol=1e-6)
    return agg[mask].reset_index(drop=True), box


def load_curve_data(data_dir: Path, tag: str, prefix: str = "",
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg_path = data_dir / f"{prefix}fdr_curve_agg_{tag}.csv"
    box_path = data_dir / f"{prefix}fdr_boxpoints_{tag}.csv"
    pbox_path = data_dir / f"{prefix}power_boxpoints_{tag}.csv"
    abox_path = data_dir / f"{prefix}accuracy_boxpoints_{tag}.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path.name} in {data_dir}")
    agg = pd.read_csv(agg_path)
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame(columns=["q", "real_fdp"])
    pbox = pd.read_csv(pbox_path) if pbox_path.exists() else pd.DataFrame(columns=["q", "power"])
    abox = pd.read_csv(abox_path) if abox_path.exists() else pd.DataFrame(columns=["q", "accuracy"])
    return agg, box, pbox, abox


def plot_combo(agg: pd.DataFrame, box: pd.DataFrame, pbox: pd.DataFrame, abox: pd.DataFrame,
               plot_dir: Path, fig_tag: str, *, search_label: str) -> None:
    q_levels = agg["q"].to_numpy()

    # --- Target FDR level vs real FDR (mean bar + 95% CI per threshold) ---
    means, ci_half = [], []
    for pos in q_levels:
        vals = box.loc[np.isclose(box["q"], pos), "real_fdp"].to_numpy()
        m, half = _mean_and_ci95(vals)
        means.append(m)
        ci_half.append(half)
    means = np.asarray(means)
    ci_half = np.asarray(ci_half)

    fig, ax = create_square_axes()
    top = _draw_metric_bars(ax, q_levels, means, value_fmt=lambda v: f"{v:.2f}",
                            ci_half=ci_half)
    ax.plot(np.arange(len(q_levels)), q_levels, linestyle="--", color="gray",
            linewidth=LINEWIDTH, marker="o", markersize=MARKERSIZE * 0.6, zorder=3,
            label="Target FDR")
    ax.set_ylim(0, max(top * 1.12, 1.05))
    apply_bold_labels(ax, "Target FDR Level", "Real FDR", f"FDR Control ({search_label})")
    place_external_legend(ax, fig)
    out = plot_dir / f"{fig_tag}_FDR_boxplots.pdf"
    save_figure(out)
    plt.close(fig)
    print(f"[OK] {out}")

    # --- Target FDR level vs power (mean bar + 95% CI per threshold) ---
    fig, ax = create_square_axes()
    if not pbox.empty:
        p_means, p_ci = [], []
        for pos in q_levels:
            vals = pbox.loc[np.isclose(pbox["q"], pos), "power"].to_numpy()
            m, half = _mean_and_ci95(vals)
            p_means.append(m)
            p_ci.append(half)
        _draw_metric_bars(ax, q_levels, np.asarray(p_means),
                          value_fmt=lambda v: f"{v:.2f}", ci_half=np.asarray(p_ci))
    else:
        _draw_metric_bars(ax, q_levels, agg["mean_power"].to_numpy(),
                          value_fmt=lambda v: f"{v:.2f}")
    ax.set_ylim(0, 1.12)
    apply_bold_labels(ax, "Target FDR Level", "Average Power", f"Power ({search_label})")
    place_external_legend(ax, fig)
    out = plot_dir / f"{fig_tag}_Power.pdf"
    save_figure(out)
    plt.close(fig)
    print(f"[OK] {out}")

    # --- Target FDR level vs accuracy (mean bar + 95% CI; zoomed y-axis) ---
    if not abox.empty:
        a_means, a_ci = [], []
        for pos in q_levels:
            vals = abox.loc[np.isclose(abox["q"], pos), "accuracy"].to_numpy()
            m, half = _mean_and_ci95(vals)
            a_means.append(m)
            a_ci.append(half)
        a_means = np.asarray(a_means)
        a_ci = np.asarray(a_ci)
        fig, ax = create_square_axes()
        finite = a_means[np.isfinite(a_means)]
        top_max = float(np.nanmax(a_means + a_ci))
        bot = float(np.min(finite)) if len(finite) else 0.0
        rng = max(top_max - bot, 0.02)
        _draw_metric_bars(ax, q_levels, a_means, value_fmt=lambda v: f"{v:.3f}",
                          ci_half=a_ci, gap=0.03 * rng)
        lo = max(0.0, bot - 0.10 * rng)
        ax.set_ylim(lo, max(top_max, 1.0) + 0.02)
        yt = ax.get_yticks()
        step = float(yt[1] - yt[0]) if len(yt) >= 2 else 0.05
        ax.set_ylim(lo, 1.0 + step)
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= 1.0 + 1e-9])
        apply_bold_labels(ax, "Target FDR Level", "Accuracy", f"Accuracy ({search_label})")
        place_external_legend(ax, fig)
        out = plot_dir / f"{fig_tag}_Accuracy.pdf"
        save_figure(out)
        plt.close(fig)
        print(f"[OK] {out}")

    # --- Target FDR level vs number of valid queries (bar per threshold, no CI) ---
    total_curves = int(agg["total_curves"].iloc[0])
    fig, ax = create_square_axes()
    top = _draw_metric_bars(ax, q_levels, agg["valid_count"].to_numpy(),
                            value_fmt=lambda v: f"{int(round(v))}", label_rotation=90)
    ax.set_ylim(0, total_curves * 1.12)
    apply_bold_labels(ax, "Target FDR Level", "Number",
                      f"Number of Valid Queries ({search_label})")
    place_external_legend(ax, fig, ncol=1)
    out = plot_dir / f"{fig_tag}_ValidCount.pdf"
    save_figure(out)
    plt.close(fig)
    print(f"[OK] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    p.add_argument("--decoy-methods", nargs="+",
                   default=["shuf", "rev", "dplm", "PGmsk25pct", "extended_shuf"])
    p.add_argument("--query-name", default="astral")
    p.add_argument("--target-name", default="astral")
    p.add_argument("--weight-method", default="AdaptiveBell",
                   help="weight label in the tag; pass '' for the NoCalib baseline")
    p.add_argument("--file-prefix", default="",
                   help="filename prefix for inputs and figures, e.g. 'NoCalib_'")
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--q-step", type=float, default=None,
                   help="keep only thresholds that are multiples of this step "
                        "(e.g. 0.1 -> 0.1, 0.2, ..., 0.9); default keeps the full grid")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="output dir; defaults to <data-dir>/figures, or "
                        "<data-dir>/figures_step<q_step> when --q-step is set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_dir is None:
        sub = f"figures_step{args.q_step:g}" if args.q_step else "figures"
        args.plot_dir = args.data_dir / sub
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    combos = list(product(args.search_methods, args.decoy_methods))
    print(f"[info] data={args.data_dir}  out={args.plot_dir}  combos={len(combos)}")

    failed = 0
    for search, decoy in combos:
        combo = f"{search} / {decoy}"
        try:
            tag = curve_tag(search, args.query_name, decoy, args.weight_method, args.target_name)
            agg, box, pbox, abox = load_curve_data(args.data_dir, tag, prefix=args.file_prefix)
            agg, box = filter_q_step(agg, box, args.q_step)
            fig_tag = f"{args.file_prefix}{figure_tag(search, args.query_name, decoy, args.weight_method)}"
            print(f"\n>>> {combo}")
            plot_combo(agg, box, pbox, abox, args.plot_dir, fig_tag,
                       search_label=search_display(search))
        except Exception as exc:  # noqa: BLE001 - report and continue over combos
            failed += 1
            print(f"[ERROR] {combo}: {exc}")

    print(f"\nFinished {len(combos)} combo(s): {len(combos) - failed} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
