import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from compute import (CALIPER_DECOY, DEFAULT_OUT_DIR, SCORE_FILES, SEARCH_METHOD_DISPLAY,
                     SPLIT_LEVELS, default_out_dir, default_tag, fdp_per_query,
                     load_perquery, mask_test, perquery_path, power_per_query)

# Nature-style figure settings, kept local to this experiment.
FONT_LABEL = 7
FONT_TITLE = 7
FONT_TICK = 6
FONT_LEGEND = 6
FONT_ANNOT = 6
LINEWIDTH = 0.9
AXES_LINEWIDTH = 0.6
GRID_LINEWIDTH = 0.4
MARKERSIZE = 3.0
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
}
_SANS_PREF = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "TeX Gyre Heros", "DejaVu Sans"]


def _apply_rcparams() -> None:
    rc = matplotlib.rcParams
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42
    rc["svg.fonttype"] = "none"
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = _SANS_PREF + list(rc["font.sans-serif"])
    rc["axes.unicode_minus"] = False
    rc["font.size"] = FONT_TICK
    rc["axes.labelsize"] = FONT_LABEL
    rc["axes.titlesize"] = FONT_TITLE
    rc["xtick.labelsize"] = FONT_TICK
    rc["ytick.labelsize"] = FONT_TICK
    rc["legend.fontsize"] = FONT_LEGEND
    rc["axes.linewidth"] = AXES_LINEWIDTH
    rc["lines.linewidth"] = LINEWIDTH
    rc["lines.markersize"] = MARKERSIZE
    rc["grid.linewidth"] = GRID_LINEWIDTH
    rc["xtick.major.width"] = AXES_LINEWIDTH
    rc["ytick.major.width"] = AXES_LINEWIDTH
    rc["xtick.major.size"] = 2.5
    rc["ytick.major.size"] = 2.5
    rc["savefig.dpi"] = 300
    rc["figure.dpi"] = 150
    rc["legend.frameon"] = True
    rc["legend.framealpha"] = 1.0


def save_figure(out_path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    if out_path.suffix.lower() == ".pdf":
        plt.savefig(out_path, format="pdf")
    else:
        plt.savefig(out_path, dpi=dpi)


def create_square_axes() -> tuple[plt.Figure, plt.Axes]:
    plot_side_in = 4.3
    margin_left_in = 0.6
    margin_bottom_in = 0.6
    margin_top_in = 0.35
    min_side_panel_in = 0.9
    fig_h = plot_side_in + margin_bottom_in + margin_top_in
    fig_w = margin_left_in + plot_side_in + min_side_panel_in + 0.12

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        margin_left_in / fig_w,
        margin_bottom_in / fig_h,
        plot_side_in / fig_w,
        plot_side_in / fig_h,
    ])
    ax.set_box_aspect(1)
    return fig, ax


def apply_bold_labels(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")


def place_external_legend(ax, fig=None, supp: str = "", legend_fontsize: int = FONT_LEGEND,
                          ncol: int = 1) -> None:
    fig = fig or ax.figure
    plot_pos = ax.get_position()
    side_x_frac = plot_pos.x1 + 0.02

    handles, labels = ax.get_legend_handles_labels()
    legend = None
    if handles:
        legend = ax.legend(
            fontsize=legend_fontsize,
            loc="upper left",
            bbox_to_anchor=(side_x_frac, plot_pos.y1),
            bbox_transform=fig.transFigure,
            frameon=True,
            fancybox=False,
            edgecolor="0.35",
            facecolor="white",
            ncol=ncol,
            borderaxespad=0.0,
        )

    text_artist = None
    if supp:
        text_artist = fig.text(
            side_x_frac,
            plot_pos.y0 + 0.02,
            supp,
            ha="left",
            va="bottom",
            fontsize=FONT_ANNOT,
            transform=fig.transFigure,
        )

    _fit_figure_to_right_panel(fig, ax, legend, text_artist)


def _ensure_figure_canvas(fig) -> None:
    if fig.canvas is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)


def _artist_right_edge_inches(fig, artist) -> float:
    _ensure_figure_canvas(fig)
    fig.canvas.draw()
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    return bbox.x1 / fig.dpi


def _fit_figure_to_right_panel(fig, ax, legend=None, text_artist=None,
                               gap_in: float = 0.12, pad_in: float = 0.18) -> None:
    _ensure_figure_canvas(fig)
    fig.canvas.draw()

    plot_pos = ax.get_position()
    old_w, old_h = fig.get_size_inches()
    plot_left_in = plot_pos.x0 * old_w
    plot_bottom_in = plot_pos.y0 * old_h
    plot_width_in = plot_pos.width * old_w
    plot_height_in = plot_pos.height * old_h
    plot_right_in = plot_left_in + plot_width_in

    rightmost_in = plot_right_in
    for artist in (legend, text_artist):
        if artist is not None:
            rightmost_in = max(rightmost_in, _artist_right_edge_inches(fig, artist))

    target_fig_w = rightmost_in + pad_in
    if target_fig_w <= old_w + 1e-6:
        return

    fig.set_size_inches(target_fig_w, old_h, forward=True)
    ax.set_position([
        plot_left_in / target_fig_w,
        plot_bottom_in / old_h,
        plot_width_in / target_fig_w,
        plot_height_in / old_h,
    ])

    new_plot_pos = ax.get_position()
    side_x_frac = (plot_right_in + gap_in) / target_fig_w

    if legend is not None:
        legend.set_bbox_to_anchor((side_x_frac, new_plot_pos.y1), transform=fig.transFigure)

    if text_artist is not None:
        text_artist.set_transform(fig.transFigure)
        text_artist.set_position((side_x_frac, new_plot_pos.y0 + 0.02))


_apply_rcparams()

# at or below this many splits a 95% CI is not worth drawing, so every split is
# shown individually instead, shaded light -> dark
DEFAULT_CI_MIN_TRIALS = 10

# the ramp data/plot_data/core uses, so these panels match the published ones
FDR_BAR_CMAP_COLORS = ["#0b0405", "#28192e", "#3b2f5f", "#3f4a8f", "#366ca0",
                       "#348da7", "#39acac", "#59ccad", "#a9e1bd"]
FDR_CMAP = mcolors.LinearSegmentedColormap.from_list("fdr_mako", FDR_BAR_CMAP_COLORS)


def TRIAL_SHADES(k: int, n: int):
    """k-th of n shades running light orange -> dark, for per-split lines."""
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "trial_orange", ["#FDD0A2", "#F16913", "#A63603", "#5B1D01"])
    return ramp(0.0 if n <= 1 else k / (n - 1))


def _save(out_path: Path, formats) -> None:
    for fmt in formats:
        save_figure(out_path.with_suffix(f".{fmt}"))
        print(f"[OK] {out_path.with_suffix('.' + fmt)}")


def _mean_ci(trials: pd.DataFrame, col: str, q_levels):
    """Mean over splits with a Student-t 95% CI (at n=5 the normal quantile is 42% short)."""
    g = trials.groupby("q")[col]
    m = g.mean().reindex(q_levels).to_numpy()
    sd = g.std(ddof=1).reindex(q_levels).to_numpy()
    cnt = g.count().reindex(q_levels).to_numpy()
    crit = np.where(cnt > 1, t_dist.ppf(0.975, np.maximum(cnt - 1, 1)), 0.0)
    return m, np.nan_to_num(np.where(cnt > 1, crit * sd / np.sqrt(cnt), 0.0))


# --------------------------------------------------------------------------
# bars: mean over splits, one bar per target FDR level
# --------------------------------------------------------------------------
def plot_bars(trials: pd.DataFrame, plot_dir: Path, tag: str, search_method: str, *,
              formats, ci_min_trials: int = DEFAULT_CI_MIN_TRIALS) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    label = SEARCH_METHOD_DISPLAY.get(search_method, search_method)
    q_levels = np.sort(trials["q"].unique())
    n_trials = int(trials["trial"].nunique())
    show_trials = n_trials <= ci_min_trials

    def bars(values, ci_half, ylabel, title, fname, *, value_fmt, ylim_top,
             reference=True, label_rotation=0, col=None):
        fig, ax = create_square_axes()
        x = np.arange(len(q_levels))
        colors = [FDR_CMAP(i / max(len(q_levels) - 1, 1)) for i in range(len(q_levels))]
        ax.bar(x, values, width=0.82, color=colors, edgecolor="none", zorder=2)
        if show_trials and col is not None:
            # too few splits for a CI -- scatter each split over its bar instead
            for k, (_, sub) in enumerate(trials.groupby("trial")):
                sub = sub.set_index("q").reindex(q_levels)
                ax.plot(x, sub[col].to_numpy(), linestyle="none", marker="o",
                        markersize=MARKERSIZE * 0.5, markeredgewidth=0,
                        color=TRIAL_SHADES(k, n_trials), zorder=5,
                        label=f"{n_trials} splits" if k == 0 else None)
            tops = np.asarray([np.nanmax(trials.loc[np.isclose(trials["q"], qq), col])
                               for qq in q_levels])
        else:
            ax.errorbar(x, values, yerr=ci_half, fmt="none", ecolor="black",
                        elinewidth=LINEWIDTH, capsize=0, zorder=4,
                        label=f"95% CI ({n_trials} splits)")
            tops = np.asarray(values) + np.asarray(ci_half)
        gap = 0.015 * max(float(np.nanmax(tops)), 1.0)
        for xi, v, t in zip(x, values, tops):
            ax.text(xi, t + gap, value_fmt(v), ha="center", va="bottom",
                    rotation=label_rotation, fontsize=FONT_ANNOT - 1,
                    fontweight="bold", color="black", zorder=5)
        if reference:
            ax.plot(x, q_levels, linestyle="--", color="gray", linewidth=LINEWIDTH,
                    marker="o", markersize=MARKERSIZE * 0.6, zorder=3, label="Target FDR")
        ax.set_xlim(-0.6, len(q_levels) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q:.2f}" for q in q_levels], rotation=45, ha="right")
        ax.set_axisbelow(True)
        ax.set_ylim(0, ylim_top(float(np.nanmax(tops))))
        apply_bold_labels(ax, "Target FDR Level", ylabel, title)
        place_external_legend(ax, fig)
        _save(plot_dir / fname, formats)
        plt.close(fig)

    m, half = _mean_ci(trials, "real_fdr", q_levels)
    bars(m, half, "Real FDR", f"Conformal FDR Control ({label})",
         f"conformal_{tag}_FDR.pdf", col="real_fdr",
         value_fmt=lambda v: f"{v:.2f}", ylim_top=lambda t: max(t * 1.12, 1.05))

    m, half = _mean_ci(trials, "power", q_levels)
    bars(m, half, "Average Power", f"Conformal Power ({label})",
         f"conformal_{tag}_Power.pdf", col="power",
         value_fmt=lambda v: f"{v:.2f}", ylim_top=lambda t: 1.12, reference=False)

    m, half = _mean_ci(trials, "lambda", q_levels)
    bars(m, half, r"Calibrated cutoff $\hat{\lambda}$", f"Conformal Threshold ({label})",
         f"conformal_{tag}_Lambda.pdf", col="lambda",
         value_fmt=lambda v: f"{v:.3f}", ylim_top=lambda t: max(t * 1.15, 0.05),
         reference=False, label_rotation=90)


# --------------------------------------------------------------------------
# trial: one split on its own, drawn like the data/plot_data/core panels
# --------------------------------------------------------------------------
def plot_trial(trials: pd.DataFrame, pq: dict, plot_dir: Path, name_stem: str,
               search_method: str, *, trial: int, metric: str, formats,
               convention: str = "risk", value_labels: bool = True) -> None:
    """`metric` is real_fdr or power; per-query spread comes from the NPZ."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    sub = trials[trials["trial"] == trial]
    q_levels = np.sort(sub["q"].unique())
    is_fdr = metric == "real_fdr"

    # Bar AND error bar both come from the per-query counts, so `convention` moves
    # them together; the CI is 1.96 SEM across this split's test queries, the
    # convention the published data/plot_data/core panels use.
    per_q = (mask_test(fdp_per_query(pq, convention=convention), pq) if is_fdr
             else mask_test(power_per_query(pq), pq))[trial]
    with np.errstate(invalid="ignore"):
        values = np.nanmean(per_q, axis=0)
        sd = np.nanstd(per_q, axis=0, ddof=1)
        cnt = np.sum(~np.isnan(per_q), axis=0)
    ci_half = np.nan_to_num(np.where(cnt > 1, 1.96 * sd / np.sqrt(np.maximum(cnt, 1)), 0.0))

    # the `risk` convention is exactly what compute.py wrote to the CSV
    if convention == "risk":
        ref = sub.set_index("q").reindex(q_levels)[metric].to_numpy(dtype=float)
        if not np.allclose(values, ref, rtol=1e-9, atol=1e-12, equal_nan=True):
            raise AssertionError(f"per-query {metric} disagrees with the trials CSV")

    n = len(q_levels)
    colors = [FDR_CMAP(i / max(n - 1, 1)) for i in range(n)]
    fig, ax = create_square_axes()
    x = np.arange(n)
    ax.bar(x, values, width=0.82, color=colors, edgecolor="none", zorder=2)
    ax.errorbar(x, values, yerr=ci_half, fmt="none", ecolor="black",
                elinewidth=LINEWIDTH, capsize=0, zorder=4, label="95% CI")

    tops = np.nan_to_num(values) + ci_half
    top = float(np.nanmax(tops)) if n else 1.0
    gap = 0.015 * max(top, 1.0)
    if value_labels:
        for xi, v, t in zip(x, values, tops):
            if np.isnan(v):
                continue
            ax.text(xi, t + gap, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=FONT_ANNOT - 1, fontweight="bold", color="black", zorder=5)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.2f}" for q in q_levels], rotation=45, ha="right")
    ax.set_axisbelow(True)
    if is_fdr:
        ax.plot(x, q_levels, linestyle="--", color="gray", linewidth=LINEWIDTH,
                marker="o", markersize=MARKERSIZE * 0.6, zorder=3, label="Target FDR")
        ax.set_ylim(0, max(top * 1.12, 1.05))
        ylabel, title, name = "Real FDR", "FDR Control", "FDR"
    else:
        ax.set_ylim(0, 1.12)
        ylabel, title, name = "Average Power", "Power", "Power"

    label = SEARCH_METHOD_DISPLAY.get(search_method, search_method)
    apply_bold_labels(ax, "Target FDR Level", ylabel,
                      f"{title} ({label}, Conformal, trial {trial + 1})")
    place_external_legend(ax, fig)
    _save(plot_dir / f"{name_stem}_trial{trial + 1}_conformal_{name}.pdf", formats)
    plt.close(fig)


# --------------------------------------------------------------------------
# caliper: overlay against the published label-free PLM-Caliper curve
# --------------------------------------------------------------------------
def load_caliper_curve(search_method: str, caliper_dir: Path, decoy, suffix: str
                       ) -> pd.DataFrame:
    decoy = decoy or CALIPER_DECOY[search_method]
    path = (caliper_dir / f"fdr_curve_agg_{search_method}_astral_{decoy}"
                          f"_AdaptiveBell_astral{suffix}.csv")
    if not path.exists():
        raise FileNotFoundError(f"missing PLM-Caliper curve: {path}")
    return pd.read_csv(path)


def plot_caliper_overlay(trials: pd.DataFrame, plot_dir: Path, tag: str,
                         search_method: str, caliper_dir: Path, decoy, suffix: str, *,
                         formats, ci_min_trials: int = DEFAULT_CI_MIN_TRIALS) -> None:
    caliper = load_caliper_curve(search_method, caliper_dir, decoy, suffix)
    plot_dir.mkdir(parents=True, exist_ok=True)
    label = SEARCH_METHOD_DISPLAY.get(search_method, search_method)
    n_calib = int(trials["n_calib"].iloc[0])
    q = np.sort(trials["q"].unique())
    n_trials = int(trials["trial"].nunique())
    show_trials = n_trials <= ci_min_trials

    def panel(conf_col, caliper_col, ylabel, title, fname, *, diagonal):
        m, half = _mean_ci(trials, conf_col, q)
        fig, ax = create_square_axes()
        if diagonal:
            ax.plot(q, q, linestyle="--", color="gray", linewidth=LINEWIDTH,
                    zorder=1, label="Target FDR")
        if show_trials:
            # too few splits for a meaningful CI -- show every split instead
            for k, (_, sub) in enumerate(trials.groupby("trial")):
                sub = sub.set_index("q").reindex(q)
                ax.plot(q, sub[conf_col].to_numpy(), color=TRIAL_SHADES(k, n_trials),
                        linewidth=LINEWIDTH * 0.75, marker="o",
                        markersize=MARKERSIZE * 0.45, zorder=3,
                        label=f"Conformal split {k + 1}" if n_trials <= 6 else None)
            if n_trials > 6:
                ax.plot([], [], color=OKABE_ITO["vermillion"], linewidth=LINEWIDTH,
                        marker="o", markersize=MARKERSIZE * 0.45,
                        label=f"Conformal, {n_trials} splits")
            hi = float(np.nanmax(trials[conf_col].to_numpy()))
        else:
            ax.plot(q, m, color=OKABE_ITO["vermillion"], linewidth=LINEWIDTH, marker="o",
                    markersize=MARKERSIZE * 0.7, zorder=3,
                    label=f"Conformal (n_cal={n_calib})")
            ax.fill_between(q, m - half, m + half, color=OKABE_ITO["vermillion"],
                            alpha=0.22, linewidth=0, zorder=2,
                            label=f"95% CI ({n_trials} splits)")
            hi = float(np.nanmax(m + half))

        ax.plot(caliper["q"], caliper[caliper_col], color=OKABE_ITO["blue"],
                linewidth=LINEWIDTH, marker="s", markersize=MARKERSIZE * 0.7,
                zorder=3, label="PLM-Caliper (label-free)")
        ax.set_xlim(0, 1.0)
        top = float(np.nanmax([hi, caliper[caliper_col].max()]))
        ax.set_ylim(0, max(top * 1.12, 0.05 if not diagonal else 1.05))
        apply_bold_labels(ax, "Target FDR Level", ylabel, f"{title} ({label})")
        place_external_legend(ax, fig)
        _save(plot_dir / fname, formats)
        plt.close(fig)

    panel("real_fdr_nonempty", "mean_real_fdr", "Real FDR", "FDR Control",
          f"compare_{tag}_FDR.pdf", diagonal=True)
    panel("power", "mean_power", "Average Power", "Power",
          f"compare_{tag}_Power.pdf", diagonal=False)


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["bars", "trial", "caliper", "all"])
    p.add_argument("--search-method", default="blastp_postprocessed",
                   choices=sorted(SCORE_FILES))
    p.add_argument("--split-level", default="random", choices=sorted(SPLIT_LEVELS))
    p.add_argument("--n-calib", type=int, default=1000)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--n-lambda", type=int, default=5000)
    p.add_argument("--lambda-grid", default="linear", choices=["linear", "quantile"])
    p.add_argument("--keep-uncertain", action="store_true")
    p.add_argument("--trials", nargs="+", type=int, default=None,
                   help="1-based split numbers for `trial` (default: every split)")
    p.add_argument("--convention", default="risk",
                   choices=["risk", "no_empties", "plmcaliper"],
                   help="how a per-query FDP is defined when drawing the error bars")
    p.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png"])
    p.add_argument("--ci-min-trials", type=int, default=DEFAULT_CI_MIN_TRIALS)
    p.add_argument("--caliper-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "plot_data" / "core")
    p.add_argument("--caliper-decoy", default=None)
    p.add_argument("--caliper-suffix", default="_tau0.25")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="default: <out-dir>/figures")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    tag = a.tag or default_tag(a.search_method, a.n_calib, a.lambda_grid,
                               a.n_lambda, a.keep_uncertain)
    out_dir = (default_out_dir(a.n_calib, a.split_level, a.n_trials)
               if a.out_dir == DEFAULT_OUT_DIR else a.out_dir)
    plot_dir = a.plot_dir or (out_dir / "figures")

    trials_path = out_dir / "data" / f"conformal_trials_{tag}.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"{trials_path} -- run compute.py first")
    trials = pd.read_csv(trials_path)

    if a.stage in ("bars", "all"):
        plot_bars(trials, plot_dir, tag, a.search_method,
                  formats=a.formats, ci_min_trials=a.ci_min_trials)
    if a.stage in ("trial", "all"):
        pq_path = perquery_path(out_dir / "data", tag)
        if not pq_path.exists():
            raise FileNotFoundError(f"{pq_path} -- re-run compute.py to emit it")
        pq = load_perquery(pq_path)
        stem = f"{a.search_method}_{a.split_level}_ncal{a.n_calib}"
        wanted = ([t - 1 for t in a.trials] if a.trials
                  else sorted(trials["trial"].unique()))
        for trial in wanted:
            for metric in ("real_fdr", "power"):
                plot_trial(trials, pq, plot_dir, stem, a.search_method,
                           trial=int(trial), metric=metric, formats=a.formats,
                           convention=a.convention)
    if a.stage in ("caliper", "all"):
        plot_caliper_overlay(trials, plot_dir, tag, a.search_method, a.caliper_dir,
                             a.caliper_decoy, a.caliper_suffix,
                             formats=a.formats, ci_min_trials=a.ci_min_trials)


if __name__ == "__main__":
    main()
