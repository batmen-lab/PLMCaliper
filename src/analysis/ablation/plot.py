import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parents[2]

FONT_LABEL = 7
FONT_TITLE = 7
FONT_TICK = 6
FONT_LEGEND = 6
FONT_ANNOT = 6
LINEWIDTH = 0.9
AXES_LINEWIDTH = 0.6
GRID_LINEWIDTH = 0.4
MARKERSIZE = 3.0
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

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ablation" / "plot_data"

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
               tau_sfx: str,
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
    out = plot_dir / f"{fig_tag}_FDR_boxplots{tau_sfx}.pdf"
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
    out = plot_dir / f"{fig_tag}_Power{tau_sfx}.pdf"
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
        out = plot_dir / f"{fig_tag}_Accuracy{tau_sfx}.pdf"
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
    out = plot_dir / f"{fig_tag}_ValidCount{tau_sfx}.pdf"
    save_figure(out)
    plt.close(fig)
    print(f"[OK] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--search-methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    p.add_argument("--decoy-methods", nargs="+",
                   default=["shuf", "rev", "dplm", "extended_shuf"])
    p.add_argument("--query-name", default="astral")
    p.add_argument("--target-name", default="astral")
    p.add_argument("--weight-method", default="AdaptiveBell",
                   help="weight label in the tag; pass '' for the NoCalib baseline")
    p.add_argument("--file-prefix", default="",
                   help="filename prefix for inputs and figures, e.g. 'NoCalib_'")
    p.add_argument("--tau", type=float, default=0.25,
                   help="appended to every figure name; None to omit")
    p.add_argument("--q-step", type=float, default=None,
                   help="keep only thresholds that are multiples of this step "
                        "(e.g. 0.1 -> 0.1, 0.2, ..., 0.9); default keeps the full grid")
    p.add_argument("--tag", default=None,
                   help="one explicit combo, matching fdr.py --tag")
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
    tau_sfx = f"_tau{args.tau:g}" if args.tau is not None else ""
    combos = ([(args.tag, None)] if args.tag
              else list(product(args.search_methods, args.decoy_methods)))
    print(f"[info] data={args.data_dir}  out={args.plot_dir}  combos={len(combos)}")

    failed = 0
    for search, decoy in combos:
        combo = search if decoy is None else f"{search} / {decoy}"
        try:
            if decoy is None:                  
                tag = fig_tag = search
            else:
                tag = curve_tag(search, args.query_name, decoy,
                                args.weight_method, args.target_name)
                fig_tag = figure_tag(search, args.query_name, decoy, args.weight_method)
            fig_tag = f"{args.file_prefix}{fig_tag}"
            agg, box, pbox, abox = load_curve_data(args.data_dir, tag, prefix=args.file_prefix)
            agg, box = filter_q_step(agg, box, args.q_step)
            print(f"\n>>> {combo}")
            plot_combo(agg, box, pbox, abox, tau_sfx, args.plot_dir, fig_tag,
                       search_label=search_display(search))
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {combo}: {exc}")

    print(f"\nFinished {len(combos)} combo(s): {len(combos) - failed} ok, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
