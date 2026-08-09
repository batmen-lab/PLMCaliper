import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_efdr_recall import (  # noqa: E402
    _combo_paths,
    _metrics_rows_for_query,
    _pr_curve_rows_for_query,
    apply_recall_zero_precision_one,
    collapse_reps_pr_to_query,
    collapse_reps_to_query,
    combo_id,
    combo_label,
    decoy_suffix,
    ensure_core_imports,
    grep_extract_pair_files,
    load_query_list,
)

AXIS_FONTSIZE = 14
FIG_SIZE = (8.0, 6.5)  # per-query PDF size (inches)
POOLED_FIG_SIZE = (10.0, 8.0)  # pooled overlay PDF size (inches)
QUERY_PR_LINE_WIDTH = 1.0  # per-query PR curve (default in compute_efdr_recall is 1.8)
QUERY_PR_LINE_COLOR = "0.55"
QUERY_PR_LINE_ALPHA = 0.55
Q_CMAP_VMIN = 0.1
Q_CMAP_VMAX = 0.9
Q_STAR_SIZE = 60
Q_STAR_SIZE_POOLED = 42
Q_STAR_ALPHA_POOLED = 1.0
Q_CMAP_COLORS = ("#2166AC", "#FFFFFF", "#B2182B")  # blue (low q) → white (q=0.5) → red (high q)


def family_method_output_dir(
    family_base: Path,
    search_method: str,
    decoy_method: str,
) -> Path:
    return family_base / combo_id(search_method, decoy_method)


def make_q_colormap():
    cmap = LinearSegmentedColormap.from_list("q_blue_white_red", Q_CMAP_COLORS)
    norm = Normalize(vmin=Q_CMAP_VMIN, vmax=Q_CMAP_VMAX)
    return cmap, norm


# Figure-coords layout: [plot] | gap | [colorbar] | [legend]
_PLOT_LEFT = 0.09
_PLOT_RIGHT_EDGE = 0.70  # main square ends here; larger => bigger plot
_PLOT_RIGHT_EDGE_NO_CBAR = 0.82
_CBAR_GAP = 0.04
_CBAR_WIDTH = 0.020
_CBAR_LABEL_PAD = 0.07  # space for colorbar y-label before legend


def _apply_figure_margins(fig: plt.Figure) -> None:
    fig.subplots_adjust(left=_PLOT_LEFT, right=0.97, top=0.90, bottom=0.11)


def finalize_right_legend_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    cmap,
    norm,
    q_levels: np.ndarray,
    legend_handles: list,
    combo_label_text: str,
    show_colorbar: bool = True,
    square_plot: bool = False,
) -> None:
    box = ax.get_position()
    plot_right = _PLOT_RIGHT_EDGE if show_colorbar else _PLOT_RIGHT_EDGE_NO_CBAR
    plot_left = _PLOT_LEFT
    max_width = max(plot_right - plot_left, 0.40)
    if square_plot:
        side = min(max_width, box.height)
        y0 = box.y0 + (box.height - side) / 2.0
        ax.set_position([plot_left, y0, side, side])
        ax.set_aspect("equal", adjustable="box")
        box = ax.get_position()
    else:
        ax.set_position([plot_left, box.y0, max_width, box.height])

    if show_colorbar:
        cbar_left = plot_right + _CBAR_GAP
        cbar_ax = fig.add_axes(
            [cbar_left, box.y0 + 0.08, _CBAR_WIDTH, box.height * 0.82],
            frameon=False,
        )
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            "FDR threshold (q)",
            fontsize=AXIS_FONTSIZE - 2,
            labelpad=6,
        )
        ticks = sorted({float(q) for q in q_levels if Q_CMAP_VMIN <= float(q) <= Q_CMAP_VMAX})
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        legend_x = cbar_left + _CBAR_WIDTH + _CBAR_LABEL_PAD
    else:
        legend_x = plot_right + 0.06
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(legend_x, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=9,
        frameon=True,
        title=combo_label_text,
        title_fontsize=9,
    )


def _prepare_pr_curve(df_pr: pd.DataFrame) -> pd.DataFrame:
    sub = df_pr.sort_values("recall").copy()
    if sub.empty:
        return sub
    if not (sub["recall"] <= 1e-12).any():
        origin = sub.iloc[0].to_dict()
        origin["recall"] = 0.0
        origin["precision"] = 1.0
        origin["threshold"] = np.nan
        sub = pd.concat([pd.DataFrame([origin]), sub], ignore_index=True)
    else:
        sub.loc[sub["recall"] <= 1e-12, "precision"] = 1.0
    return sub


def _plot_q_stars(
    ax: plt.Axes,
    df_q: pd.DataFrame,
    q_levels: np.ndarray,
    cmap,
    norm,
    *,
    star_size: float = Q_STAR_SIZE,
    alpha: float = 1.0,
    zorder: int = 5,
) -> ScalarMappable | None:
    recalls, precisions, qs = [], [], []
    for q in q_levels:
        row = df_q.loc[np.isclose(df_q["q"], float(q))]
        if row.empty:
            continue
        r, p = apply_recall_zero_precision_one(
            float(row["recall"].iloc[0]),
            float(row["precision"].iloc[0]),
        )
        recalls.append(r)
        precisions.append(p)
        qs.append(float(q))
    if not qs:
        return None
    sc = ax.scatter(
        recalls,
        precisions,
        c=qs,
        cmap=cmap,
        norm=norm,
        marker="*",
        s=star_size,
        alpha=alpha,
        edgecolors="0.25",
        linewidths=0.25,
        zorder=zorder,
    )
    return sc


def plot_query_pr_curve(
    qid: str,
    df_pr: pd.DataFrame,
    df_q: pd.DataFrame,
    *,
    combo_label_text: str,
    family_id: str,
    q_levels: np.ndarray,
    output_path: Path,
    cmap,
    norm,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _apply_figure_margins(fig)

    sub_pr = _prepare_pr_curve(df_pr)
    if not sub_pr.empty:
        ax.plot(
            sub_pr["recall"],
            sub_pr["precision"],
            linewidth=QUERY_PR_LINE_WIDTH,
            color=QUERY_PR_LINE_COLOR,
            alpha=QUERY_PR_LINE_ALPHA,
            label="PR curve",
            zorder=3,
        )

    sc = _plot_q_stars(ax, df_q.sort_values("q"), q_levels, cmap, norm)

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel("Recall", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(f"{qid}  |  family {family_id}", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_box_aspect(1)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=QUERY_PR_LINE_COLOR,
            alpha=QUERY_PR_LINE_ALPHA,
            linewidth=QUERY_PR_LINE_WIDTH,
            label="PR curve",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=10,
            markerfacecolor=cmap(norm(0.5)),
            markeredgecolor="0.25",
            label="FDR q (stars)",
        ),
    ]
    finalize_right_legend_colorbar(
        fig,
        ax,
        cmap=cmap,
        norm=norm,
        q_levels=q_levels,
        legend_handles=legend_handles,
        combo_label_text=combo_label_text,
        show_colorbar=sc is not None,
        square_plot=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_family_all_pr_curves(
    query_ids: list[str],
    df_per: pd.DataFrame,
    df_pr: pd.DataFrame,
    *,
    family_id: str,
    combo_label_text: str,
    q_levels: np.ndarray,
    output_path: Path,
    cmap,
    norm,
) -> None:
    fig, ax = plt.subplots(figsize=POOLED_FIG_SIZE)
    _apply_figure_margins(fig)
    n_plotted = 0

    for qid in query_ids:
        sub_pr = df_pr.loc[df_pr["qid"] == qid]
        sub_q = df_per.loc[df_per["qid"] == qid]
        sub = _prepare_pr_curve(sub_pr)
        if sub.empty or sub_q.empty:
            continue
        ax.plot(
            sub["recall"],
            sub["precision"],
            color="0.55",
            alpha=0.28,
            linewidth=0.9,
            zorder=1,
        )
        _plot_q_stars(
            ax,
            sub_q.sort_values("q"),
            q_levels,
            cmap,
            norm,
            star_size=Q_STAR_SIZE_POOLED,
            alpha=Q_STAR_ALPHA_POOLED,
            zorder=3,
        )
        n_plotted += 1

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel("Recall", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(
        f"Family {family_id}: all queries (n={n_plotted})",
        fontsize=AXIS_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_box_aspect(1)

    legend_handles = [
        Line2D([0], [0], color="0.55", alpha=0.28, linewidth=0.9, label="Per-query PR curve"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=7,
            markeredgewidth=0.25,
            markerfacecolor=cmap(norm(0.5)),
            alpha=Q_STAR_ALPHA_POOLED,
            markeredgecolor="0.25",
            label="FDR thresholds (stars)",
        ),
    ]
    finalize_right_legend_colorbar(
        fig,
        ax,
        cmap=cmap,
        norm=norm,
        q_levels=q_levels,
        legend_handles=legend_handles,
        combo_label_text=combo_label_text,
        show_colorbar=True,
        square_plot=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def compute_per_query_tables(
    search_method: str,
    decoy_method: str,
    seq_name: str,
    query_ids: set[str],
    q_levels: np.ndarray,
    grep_cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_core_imports()
    from fdr import compute_tda_fdr_per_query, make_pair_table

    real_path, decoy_path = _combo_paths(search_method, decoy_method, seq_name)
    real_sub, decoy_sub = grep_extract_pair_files(
        real_path,
        decoy_path,
        query_ids,
        decoy_suffix(decoy_method),
        grep_cache_dir,
    )

    print("[INFO] Loading pair table from grep subset ...")
    df_merge = make_pair_table(
        str(real_sub),
        str(decoy_sub),
        decoy_suffix(decoy_method),
    )
    df_merge = df_merge.loc[df_merge["qid"].isin(query_ids)].copy()
    print(f"[INFO] Loaded {df_merge['qid'].nunique()} queries, {len(df_merge):,} pairs")

    rows: list[dict] = []
    pr_rows: list[dict] = []
    for (qid, rep_id), df_q in df_merge.groupby(["qid", "rep_id"], sort=False):
        df_curve = compute_tda_fdr_per_query(df_q)
        if df_curve.empty:
            continue
        kw = dict(
            seq_name=seq_name,
            search_method=search_method,
            decoy_method=decoy_method,
            qid=qid,
            rep_id=rep_id,
        )
        rows.extend(_metrics_rows_for_query(df_q, df_curve, q_levels=q_levels, **kw))
        pr_rows.extend(_pr_curve_rows_for_query(df_q, df_curve, **kw))

    df_per = collapse_reps_to_query(pd.DataFrame(rows))
    df_pr = collapse_reps_pr_to_query(pd.DataFrame(pr_rows))
    return df_per, df_pr


def plot_all_queries(
    query_ids: list[str],
    df_per: pd.DataFrame,
    df_pr: pd.DataFrame,
    *,
    family_id: str,
    combo_label_text: str,
    q_levels: np.ndarray,
    output_dir: Path,
    data_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cmap, norm = make_q_colormap()
    for old_plot in list(output_dir.glob("pr_curve_*.png")) + list(output_dir.glob("pr_curve_*.pdf")):
        old_plot.unlink()
    pooled_path = output_dir / "pooled_family_pr_curve.pdf"
    for old_pooled in (
        output_dir / "pooled_family_pr_curve.png",
        output_dir / "pooled_family_pr_curve.pdf",
    ):
        if old_pooled.exists():
            old_pooled.unlink()
    missing: list[str] = []

    for qid in query_ids:
        sub_q = df_per.loc[df_per["qid"] == qid].copy()
        sub_pr = df_pr.loc[df_pr["qid"] == qid].copy()
        if sub_q.empty or sub_pr.empty:
            missing.append(qid)
            print(f"[WARN] No PR data for query {qid}; skip")
            continue

        plot_query_pr_curve(
            qid,
            sub_pr,
            sub_q,
            combo_label_text=combo_label_text,
            family_id=family_id,
            q_levels=q_levels,
            output_path=output_dir / f"pr_curve_{qid}.pdf",
            cmap=cmap,
            norm=norm,
        )

    plotted_ids = [q for q in query_ids if q not in missing]
    if plotted_ids:
        plot_family_all_pr_curves(
            plotted_ids,
            df_per,
            df_pr,
            family_id=family_id,
            combo_label_text=combo_label_text,
            q_levels=q_levels,
            output_path=pooled_path,
            cmap=cmap,
            norm=norm,
        )

    summary_rows = []
    for qid in query_ids:
        sub_q = df_per.loc[df_per["qid"] == qid]
        if sub_q.empty:
            continue
        for _, row in sub_q.sort_values("q").iterrows():
            r, p = apply_recall_zero_precision_one(float(row["recall"]), float(row["precision"]))
            summary_rows.append(
                {
                    "qid": qid,
                    "family_id": family_id,
                    "q": float(row["q"]),
                    "recall": r,
                    "precision": p,
                }
            )
    if summary_rows:
        summary_path = data_dir / "recall_precision_vs_q_per_query.tsv"
        pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
        print(f"[OK] {summary_path}")

    if not df_pr.empty:
        pr_path = data_dir / "pr_curve_per_query.tsv"
        df_pr.to_csv(pr_path, sep="\t", index=False)
        print(f"[OK] {pr_path}")

    if missing:
        print(f"[WARN] Missing data for {len(missing)} queries: {', '.join(missing)}")


def load_saved_plot_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_q_path = data_dir / "recall_precision_vs_q_per_query.tsv"
    pr_path = data_dir / "pr_curve_per_query.tsv"
    if not per_q_path.is_file():
        raise FileNotFoundError(f"Missing {per_q_path} — run full compute first.")
    if not pr_path.is_file():
        raise FileNotFoundError(f"Missing {pr_path} — run full compute first.")
    print(f"[INFO] Loading {per_q_path.name} ...")
    df_per = pd.read_csv(per_q_path, sep="\t")
    print(f"[INFO] Loading {pr_path.name} ({pr_path.stat().st_size / 1e6:.1f} MB) ...")
    df_pr = pd.read_csv(pr_path, sep="\t")
    print(f"[INFO] Loaded q-metrics: {len(df_per):,} rows; PR curve: {len(df_pr):,} rows")
    return df_per, df_pr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Family per-query PR curves.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read data/*.tsv and redraw PDFs only (no grep / TDA FDR).",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    FAMILY_IDS = ["a.4.1.1", "a.39.1.5", "b.47.1.2", "c.2.1.1"]
    SEARCH_METHODS = ["plm", "tmvec", "dhr_postprocess"]
    SEQ_NAME = "astral"
    DECOY_METHOD = "extended_shuf"
    Q_LEVELS = np.arange(0.10, 1.00, 0.10)

    for FAMILY_ID in FAMILY_IDS:
        QUERY_LIST = PROJECT_ROOT / "data" / f"querylist_{FAMILY_ID}.txt"
        FAMILY_BASE = PROJECT_ROOT / "results_final" / "pr_curve" / f"family_{FAMILY_ID}"
        query_ids_set = load_query_list(QUERY_LIST)
        query_ids = [line.strip() for line in QUERY_LIST.read_text().splitlines() if line.strip()]
        query_ids = [q for q in query_ids if q in query_ids_set]

        for SEARCH_METHOD in SEARCH_METHODS:
            OUTPUT_DIR = family_method_output_dir(FAMILY_BASE, SEARCH_METHOD, DECOY_METHOD)
            DATA_DIR = OUTPUT_DIR / "data"
            GREP_CACHE_DIR = OUTPUT_DIR / "grep_cache"

            print(f"\n[INFO] family={FAMILY_ID}, combo={combo_id(SEARCH_METHOD, DECOY_METHOD)}, n_queries={len(query_ids)}")

            if args.plot_only:
                print("[INFO] --plot-only: using saved TSV tables")
                df_per, df_pr = load_saved_plot_tables(DATA_DIR)
            else:
                df_per, df_pr = compute_per_query_tables(
                    SEARCH_METHOD,
                    DECOY_METHOD,
                    SEQ_NAME,
                    query_ids_set,
                    Q_LEVELS,
                    GREP_CACHE_DIR,
                )

            plot_all_queries(
                query_ids,
                df_per,
                df_pr,
                family_id=FAMILY_ID,
                combo_label_text=combo_label(SEARCH_METHOD, DECOY_METHOD),
                q_levels=Q_LEVELS,
                output_dir=OUTPUT_DIR,
                data_dir=DATA_DIR,
            )

            n_single = len(list(OUTPUT_DIR.glob("pr_curve_*.pdf")))
            print(f"[DONE] {n_single} per-query PDFs + pooled under {OUTPUT_DIR}")
