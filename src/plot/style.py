from pathlib import Path

import matplotlib

# Column widths (inch).
WIDTH_SINGLE = 3.5   # 89 mm
WIDTH_DOUBLE = 7.2   # 183 mm
HEIGHT_MAX = 9.7     # 247 mm (one page)

FONT_LABEL = 7       # axis labels
FONT_TITLE = 7       # in-figure title / panel descriptor (Nature prefers none)
FONT_TICK = 6        # tick numbers
FONT_LEGEND = 6      # legend entries
FONT_ANNOT = 6       # small annotations / supp text

# Line / marker weights (pt).
LINEWIDTH = 0.9      # data lines
AXES_LINEWIDTH = 0.6  # spines / ticks
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


_apply_rcparams()

import matplotlib.pyplot as plt 


def save_figure(out_path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    if out_path.suffix.lower() == ".pdf":
        plt.savefig(out_path, format="pdf")
    else:
        plt.savefig(out_path, dpi=dpi)


def supp_text(title_suffix: str = "", weight_method: str = "", tau: float = 0.0) -> str:
    lines = []
    if title_suffix:
        lines.append(title_suffix.strip())
    if weight_method or tau is not None:
        lines.append("")
    return "\n".join(lines)


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


# --- internals (figure auto-sizing) ---------------------------------------

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
