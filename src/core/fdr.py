import csv
import gc
import os
import sys
import types
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

try:
    from tqdm import tqdm
except ImportError:
    class _FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            return None

        def close(self):
            return None

    _fake_tqdm_mod = types.ModuleType("tqdm")
    _fake_tqdm_mod.tqdm = _FakeTqdm
    sys.modules["tqdm"] = _fake_tqdm_mod
    tqdm = _FakeTqdm

warnings.filterwarnings("ignore", category=FutureWarning)

STREAMING_THRESHOLD_GB = 1.0


def file_size_gb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024**3)


def should_stream_pair_table(
    path_real: str | Path,
    path_decoy: str | Path,
    *,
    force_load_all: bool = False,
    threshold_gb: float = STREAMING_THRESHOLD_GB,
) -> bool:
    if force_load_all:
        return False
    return (
        file_size_gb(path_real) > threshold_gb
        or file_size_gb(path_decoy) > threshold_gb
    )


def _iter_qid_blocks(path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    current_qid: str | None = None
    block: list[dict] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["qid"]
            if current_qid is not None and qid != current_qid:
                yield current_qid, block
                block = []
            current_qid = qid
            block.append(row)
    if block and current_qid is not None:
        yield current_qid, block


def _rows_to_pair_frame(real_rows: list[dict], decoy_rows: list[dict], decoy_suffix: str) -> pd.DataFrame:
    df_real = pd.DataFrame(real_rows)
    df_decoy = pd.DataFrame(decoy_rows)
    if df_real.empty:
        return pd.DataFrame()

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
    df_real["score_real"] = df_real["score_real"].astype(np.float32)

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
    df_decoy["score_decoy"] = df_decoy["score_decoy"].astype(np.float32)

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig", "tid"], errors="ignore")
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def iter_pair_tables_by_qid(
    path_real: str | Path,
    path_decoy: str | Path,
    decoy_suffix: str,
) -> Iterator[tuple[str, pd.DataFrame]]:
    real_iter = _iter_qid_blocks(path_real)
    decoy_iter = _iter_qid_blocks(path_decoy)

    for (real_qid, real_rows), (decoy_qid, decoy_rows) in zip(real_iter, decoy_iter, strict=True):
        qid = remove_decoy_suffix(decoy_qid, decoy_suffix)
        if qid != real_qid:
            raise ValueError(f"QID mismatch while streaming: real={real_qid!r} decoy={decoy_qid!r}")
        df_merge = _rows_to_pair_frame(real_rows, decoy_rows, decoy_suffix)
        if not df_merge.empty:
            yield real_qid, df_merge


def remove_decoy_suffix(qid: str, decoy_suffix: str) -> str:
    if qid.endswith(decoy_suffix):
        return qid[:-len(decoy_suffix)]
    return qid


def make_pair_table(path_real, path_decoy, decoy_suffix):

    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={
        "score": "score_real",
        "homo_type": "homo_type_real",
    })

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={
        "score": "score_decoy",
    })

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner"
    )

    # Drop columns not needed downstream to save memory
    df_merge = df_merge.drop(columns=["tid", "qid_orig"], errors="ignore")

    # Downcast numeric types to reduce memory footprint
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")

    df_merge["qid"] = df_merge["qid"].astype("category")

    return df_merge


def compute_tda_fdr_per_query(df_q):
    '''
    homo: 1, 2
    nonhomo: -1
    '''
    scores_t = df_q["score_real"].values
    scores_d = df_q["score_decoy"].values

    is_homo_t = df_q["homo_type_real"].isin([1, 2]).values
    is_strict_nonhomo = (df_q["homo_type_real"] == -1).values
    is_annotated = is_homo_t | is_strict_nonhomo
    
    total_homo = is_homo_t.sum() 

    scores_t_sorted = np.sort(scores_t)
    scores_d_sorted = np.sort(scores_d)

    scores_t_homo_only = np.sort(scores_t[is_homo_t])
    scores_t_false_sorted = np.sort(scores_t[is_strict_nonhomo])
    scores_t_annotated_sorted = np.sort(scores_t[is_annotated])

    # candidate thresholds
    cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))

    target_count = len(scores_t) - np.searchsorted(scores_t_sorted, cand_t, side='left')
    decoy_count = len(scores_d) - np.searchsorted(scores_d_sorted, cand_t, side='left')
    true_pos_count = len(scores_t_homo_only) - np.searchsorted(scores_t_homo_only, cand_t, side='left')
    false_count = len(scores_t_false_sorted) - np.searchsorted(scores_t_false_sorted, cand_t, side='left')
    annotated_count = len(scores_t_annotated_sorted) - np.searchsorted(scores_t_annotated_sorted, cand_t, side='left')

    ## FDP
    target_count_max1 = np.maximum(target_count, 1.0)
    annotated_count_max1 = np.maximum(annotated_count, 1.0)
    
    est_fdp = (decoy_count + 1.0) / target_count_max1 # eFDP
    real_fdp = false_count / annotated_count_max1 # real FDP

    df_curve = pd.DataFrame({
        "qid": df_q["qid"].iloc[0],
        "rep_id": df_q["rep_id"].iloc[0],
        "threshold": cand_t,
        "est_fdp": est_fdp,
        "real_fdp": real_fdp,
        "n_selected": target_count,
        "n_true_pos": true_pos_count,
        "n_false": false_count,
        "total_homo": total_homo,
        "n_neg_total": int(is_strict_nonhomo.sum()),
    })

    return df_curve


def _compute_curves_for_qid_block(df_qall: pd.DataFrame) -> list[pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    for _, df_q in df_qall.groupby(["qid", "rep_id"], sort=False):
        df_curve = compute_tda_fdr_per_query(df_q)
        if not df_curve.empty:
            curves.append(df_curve)
    return curves


def _maybe_flush_curve_list(curve_list: list[pd.DataFrame], flush_every: int) -> list[pd.DataFrame]:
    if len(curve_list) >= flush_every:
        merged = pd.concat(curve_list, ignore_index=True)
        gc.collect()
        return [merged]
    return curve_list


def estimate_fdr_curves_all_queries(df_merge, flush_every=2000):
    curve_list = []

    grouped = df_merge.groupby(["qid", "rep_id"], sort=False)

    # compute TDA curves
    for i, ((qid, rep_id), df_q) in enumerate(
        tqdm(grouped, desc="  -> Computing TDA curves", leave=False, colour='cyan')
    ):
        df_curve = compute_tda_fdr_per_query(df_q)
        if not df_curve.empty:
            curve_list.append(df_curve)

        # Periodically merge the list to release memory held by many small DFs
        if len(curve_list) >= flush_every:
            curve_list = [pd.concat(curve_list, ignore_index=True)]
            gc.collect()

    if len(curve_list) > 0:
        df_curve_all = pd.concat(curve_list, ignore_index=True)
    else:
        df_curve_all = pd.DataFrame()

    return df_curve_all, pd.DataFrame()  # empty df_combined_all for API compat


def estimate_fdr_curves_streaming(
    path_real: str | Path,
    path_decoy: str | Path,
    decoy_suffix: str,
    flush_every: int = 2000,
    desc: str = "  -> Computing TDA curves",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_list: list[pd.DataFrame] = []
    pbar = tqdm(desc=desc, unit="query", colour="cyan")
    try:
        if n_jobs <= 1:
            for _, df_qall in iter_pair_tables_by_qid(path_real, path_decoy, decoy_suffix):
                curve_list.extend(_compute_curves_for_qid_block(df_qall))
                pbar.update(1)
                curve_list = _maybe_flush_curve_list(curve_list, flush_every)
        else:
            max_inflight = max(n_jobs * 2, n_jobs + 1)
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                pending: set = set()
                for _, df_qall in iter_pair_tables_by_qid(path_real, path_decoy, decoy_suffix):
                    pending.add(pool.submit(_compute_curves_for_qid_block, df_qall))
                    if len(pending) >= max_inflight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            curve_list.extend(fut.result())
                            pbar.update(1)
                        curve_list = _maybe_flush_curve_list(curve_list, flush_every)
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        curve_list.extend(fut.result())
                        pbar.update(1)
                    curve_list = _maybe_flush_curve_list(curve_list, flush_every)
    finally:
        pbar.close()

    if curve_list:
        df_curve_all = pd.concat(curve_list, ignore_index=True)
    else:
        df_curve_all = pd.DataFrame()
    return df_curve_all, pd.DataFrame()



def _save_current_figure(out_path, dpi=300):
    out_path = Path(out_path)
    if out_path.suffix.lower() == ".pdf":
        plt.savefig(out_path, format="pdf")
    else:
        plt.savefig(out_path, dpi=dpi)


def _supp_text(title_suffix="", weight_method="", tau=0):
    lines = []
    if title_suffix:
        lines.append(title_suffix.strip())
    if weight_method or tau is not None:
        lines.append(f"")
    return "\n".join(lines)


def _ensure_figure_canvas(fig):
    if fig.canvas is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)


def _artist_right_edge_inches(fig, artist):
    _ensure_figure_canvas(fig)
    fig.canvas.draw()
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    return bbox.x1 / fig.dpi


def _fit_figure_to_right_panel(fig, ax, legend=None, text_artist=None, gap_in=0.12, pad_in=0.18):
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
    ax.set_position(
        [
            plot_left_in / target_fig_w,
            plot_bottom_in / old_h,
            plot_width_in / target_fig_w,
            plot_height_in / old_h,
        ]
    )

    new_plot_pos = ax.get_position()
    side_x_frac = (plot_right_in + gap_in) / target_fig_w

    if legend is not None:
        legend.set_bbox_to_anchor(
            (side_x_frac, new_plot_pos.y1),
            transform=fig.transFigure,
        )

    if text_artist is not None:
        text_artist.set_transform(fig.transFigure)
        text_artist.set_position((side_x_frac, new_plot_pos.y0 + 0.02))


def _create_square_axes():
    plot_side_in = 4.8
    margin_left_in = 0.65
    margin_bottom_in = 0.84
    margin_top_in = 0.48
    min_side_panel_in = 0.8
    fig_h = plot_side_in + margin_bottom_in + margin_top_in
    fig_w = margin_left_in + plot_side_in + min_side_panel_in + 0.12

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [
            margin_left_in / fig_w,
            margin_bottom_in / fig_h,
            plot_side_in / fig_w,
            plot_side_in / fig_h,
        ]
    )
    ax.set_box_aspect(1)
    return fig, ax


def _apply_bold_labels(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold")


def _place_external_legend_and_supp(fig, ax, supp_text="", legend_fontsize=11, ncol=1):
    plot_pos = ax.get_position()
    side_x_frac = plot_pos.x1 + 0.02

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
    if supp_text:
        text_artist = fig.text(
            side_x_frac,
            plot_pos.y0 + 0.02,
            supp_text,
            ha="left",
            va="bottom",
            fontsize=11,
            transform=fig.transFigure,
        )

    _fit_figure_to_right_panel(fig, ax, legend, text_artist)


DEFAULT_Q_LEVELS = np.arange(0.05, 1.00, 0.05)


def summarize_fdr_curves_at_q_levels(df_curve_all, q_levels=None):
    if q_levels is None:
        q_levels = DEFAULT_Q_LEVELS

    grouped_curves = list(df_curve_all.groupby(["qid", "rep_id"], sort=False))
    rows = []

    for q in q_levels:
        q = float(q)
        for (_qid, _rep_id), grp in grouped_curves:
            valid_rows = grp[grp["est_fdp"] <= q]
            n_neg_total = float(grp["n_neg_total"].iloc[0]) if "n_neg_total" in grp else 0.0
            if len(valid_rows) > 0:
                row = valid_rows.loc[valid_rows["threshold"].idxmin()]
                total_homo = row["total_homo"]
                power = row["n_true_pos"] / total_homo if total_homo > 0 else 0.0
                tp = float(row["n_true_pos"])
                fp = float(row.get("n_false", np.nan))
                denom = total_homo + n_neg_total
                # Accuracy = (TP + TN) / (TP + FP + TN + FN); TN = n_neg_total - FP.
                accuracy = (tp + (n_neg_total - fp)) / denom if denom > 0 else np.nan
                rows.append(
                    {
                        "qid": _qid,
                        "rep_id": _rep_id,
                        "q": q,
                        "est_fdp": row["est_fdp"],
                        "real_fdp": row["real_fdp"],
                        "power": power,
                        "accuracy": accuracy,
                        "valid": 1,
                    }
                )
            else:
                total_homo = float(grp["total_homo"].iloc[0]) if "total_homo" in grp else 0.0
                denom = total_homo + n_neg_total
                # No hit accepted -> TP=FP=0, TN=n_neg_total, FN=total_homo.
                accuracy = n_neg_total / denom if denom > 0 else np.nan
                rows.append(
                    {
                        "qid": _qid,
                        "rep_id": _rep_id,
                        "q": q,
                        "est_fdp": np.nan,
                        "real_fdp": np.nan,
                        "power": 0.0,
                        "accuracy": accuracy,
                        "valid": 0,
                    }
                )

    return pd.DataFrame(rows)


def aggregate_plot_summary(df_summary):
    q_levels = np.sort(df_summary["q"].unique())
    total_curves = df_summary.groupby(["qid", "rep_id"]).ngroups
    n_unique_qids = df_summary["qid"].nunique()

    has_accuracy = "accuracy" in df_summary.columns

    mean_real_fdr_list = []
    mean_power_list = []
    mean_accuracy_list = []
    valid_count_list = []
    all_real_fdps_per_q = []
    all_powers_per_q = []
    all_accuracy_per_q = []

    for q in q_levels:
        sub = df_summary[df_summary["q"] == q]
        real_fdps = sub["real_fdp"].to_numpy()
        powers = sub["power"].to_numpy()
        valid_count = int(sub["valid"].sum())

        mean_real_fdr_list.append(float(np.nanmean(real_fdps)))
        mean_power_list.append(float(np.nanmean(powers)))
        valid_count_list.append(valid_count)
        all_real_fdps_per_q.append(sub.loc[sub["valid"] == 1, "real_fdp"].dropna().tolist())
        all_powers_per_q.append([float(p) for p in powers if not np.isnan(p)])

        if has_accuracy:
            accs = sub["accuracy"].to_numpy()
            mean_accuracy_list.append(float(np.nanmean(accs)))
            all_accuracy_per_q.append([float(a) for a in accs if not np.isnan(a)])
        else:
            mean_accuracy_list.append(float("nan"))
            all_accuracy_per_q.append([])

    return {
        "q_levels": q_levels,
        "mean_real_fdr": mean_real_fdr_list,
        "mean_power": mean_power_list,
        "mean_accuracy": mean_accuracy_list,
        "valid_count": valid_count_list,
        "all_real_fdps_per_q": all_real_fdps_per_q,
        "all_powers_per_q": all_powers_per_q,
        "all_accuracy_per_q": all_accuracy_per_q,
        "total_curves": total_curves,
        "n_unique_qids": n_unique_qids,
    }


def plot_q_vs_real_fdr_and_discoveries_from_summary(
    df_summary,
    out_png_fdr,
    out_png_power,
    out_png_valid,
    title_suffix="",
    weight_method="",
    tau=0,
):
    out_fdr_path = Path(out_png_fdr)
    out_power_path = Path(out_png_power)
    out_valid_path = Path(out_png_valid)
    out_fdr_boxplots = out_fdr_path.with_name(
        f"{out_fdr_path.stem}_boxplots{out_fdr_path.suffix}"
    )

    agg = aggregate_plot_summary(df_summary)
    q_levels = agg["q_levels"]
    mean_real_fdr_list = agg["mean_real_fdr"]
    mean_power_list = agg["mean_power"]
    valid_count_list = agg["valid_count"]
    all_real_fdps_per_q = agg["all_real_fdps_per_q"]
    total_curves = agg["total_curves"]
    n_unique_qids = agg["n_unique_qids"]

    supp_text = _supp_text(title_suffix, weight_method, tau)

    # ==========================================
    # Plot 1: Target q vs Real FDR (boxplots)
    # ==========================================
    fig, ax = _create_square_axes()
    ax.plot([0, 1.0], [0, 1.0], linestyle="--", color="gray", linewidth=2, label="y = x")

    for pos, vals in zip(q_levels, all_real_fdps_per_q):
        if len(vals) > 0:
            ax.boxplot(
                vals,
                positions=[pos],
                widths=0.02,
                showfliers=True,
                patch_artist=True,
                boxprops=dict(facecolor="plum", color="purple", alpha=0.3),
                medianprops=dict(color="indigo", linewidth=1.5),
                whiskerprops=dict(color="purple", alpha=0.6),
                capprops=dict(color="purple", alpha=0.6),
                flierprops=dict(marker=".", color="purple", alpha=0.2, markersize=4),
                manage_ticks=False,
            )

    ax.plot(
        q_levels,
        mean_real_fdr_list,
        marker="o",
        markersize=5,
        color="purple",
        linewidth=2,
        label="Mean",
        zorder=5,
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.0)
    _apply_bold_labels(ax, "Target FDR Level", "Real FDR", "FDR Control Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text)
    _save_current_figure(out_fdr_boxplots)
    plt.close(fig)
    print(f"  -> [OK] Saved FDR Control Curve with Boxplots: {out_fdr_boxplots}")

    # ==========================================
    # Plot 2: Target q vs Power (Proportion)
    # ==========================================
    fig, ax = _create_square_axes()
    ax.plot(
        q_levels,
        mean_power_list,
        marker="s",
        markersize=5,
        color="darkgreen",
        linewidth=2,
        label="Power",
    )

    ax.set_ylim(-0.05, 1.05)
    _apply_bold_labels(ax, "Target FDR Level", "Average Power", "Power Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text)
    _save_current_figure(out_power_path)
    plt.close(fig)
    print(f"  -> [OK] Saved Power Curve: {out_power_path}")

    # ==========================================
    # Plot 3: Target q vs Number of Valid Queries
    # ==========================================
    fig, ax = _create_square_axes()
    ax.plot(
        q_levels,
        valid_count_list,
        marker="^",
        markersize=5,
        color="teal",
        linewidth=2,
        label="Valid (Query, Rep) Pairs",
    )

    ax.axhline(
        y=total_curves,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Total (Query x Rep) = {n_unique_qids} x "
            f"{total_curves // max(n_unique_qids, 1)} = {total_curves}"
        ),
    )

    ax.set_ylim(0, total_curves + total_curves * 0.05)
    _apply_bold_labels(ax, "Target FDR Level", "Number", "Number of Valid Queries")
    ax.grid(True, linestyle="--", alpha=0.4)
    _place_external_legend_and_supp(fig, ax, supp_text=supp_text, ncol=1)
    _save_current_figure(out_valid_path)
    plt.close(fig)
    print(f"  -> [OK] Saved Valid Queries Curve: {out_valid_path}")

    return out_fdr_boxplots, out_power_path, out_valid_path


def plot_q_vs_real_fdr_and_discoveries(
    df_curve_all,
    out_png_fdr,
    out_png_power,
    out_png_valid,
    title_suffix="",
    weight_method="",
    tau=0,
):
    df_summary = summarize_fdr_curves_at_q_levels(df_curve_all)
    return plot_q_vs_real_fdr_and_discoveries_from_summary(
        df_summary,
        out_png_fdr,
        out_png_power,
        out_png_valid,
        title_suffix=title_suffix,
        weight_method=weight_method,
        tau=tau,
    )


def plot_tda_scatter_for_query(df_merge, target_qid, out_path, q_level=0.10, manual_t=None):
    df_q = df_merge[df_merge["qid"] == target_qid].copy()
    if len(df_q) == 0:
        print(f"[Warning] No data found for query: {target_qid}")
        return

    scores_t = df_q["score_real"].values
    scores_d = df_q["score_decoy"].values
    is_homo = df_q["homo_type_real"].isin([1, 2]).values

    # find TDA threshold T_q
    if manual_t is not None:
        t = manual_t
    else:
        cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
        found_valid_t = False

        for cand in cand_t:
            t_cnt = (scores_t >= cand).sum()
            d_cnt = (scores_d >= cand).sum()
            est_fdp = (d_cnt + 1.0) / max(t_cnt, 1.0)
            if est_fdp <= q_level:
                t = cand
                found_valid_t = True
                break

        if not found_valid_t:
            print(f"  -> [Stats] Query: {target_qid} | FAILED to control FDR at q={q_level}.")
            return

    target_count = (scores_t >= t).sum()
    decoy_count = (scores_d >= t).sum()
    false_pos_count = ((scores_t >= t) & (~is_homo)).sum()

    pos_count_max1 = max(target_count, 1.0)
    final_est_fdp = (decoy_count + 1.0) / pos_count_max1
    real_fdp = false_pos_count / pos_count_max1

    # plot scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    X_null, Y_null = scores_t[~is_homo], scores_d[~is_homo]
    X_nonnull, Y_nonnull = scores_t[is_homo], scores_d[is_homo]

    lim_min, lim_max = 0.0, 1.0
    ax.axvline(x=t, color='blue', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.axhline(y=t, color='orange', linestyle='-', linewidth=1.5, alpha=0.5)

    ax.fill_betweenx([lim_min, lim_max], t, lim_max, color='lightblue', alpha=0.2)
    ax.fill_between([lim_min, lim_max], t, lim_max, color='moccasin', alpha=0.2)

    ax.text(t + 0.02, lim_min + 0.05, f'Discoveries\n($S_T \geq {t:.2f}$)',
            color='blue', fontsize=11, ha='left', va='bottom')
    ax.text(lim_min + 0.05, t + 0.02, f'Decoy Hits\n($S_D \geq {t:.2f}$)',
            color='orange', fontsize=11, ha='left', va='bottom')

    ax.plot([lim_min, lim_max], [lim_min, lim_max], color='gray', linestyle='--', linewidth=1.2, alpha=0.5)

    ax.scatter(X_null, Y_null, color='gray', marker='.', alpha=0.5, s=50, label='Non-homo')
    ax.scatter(X_nonnull, Y_nonnull, color='red', marker='s', alpha=1, s=30, label='Homo')

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Score$_{real}$ (Target)', fontsize=13)
    ax.set_ylabel('Score$_{decoy}$ (Calibrated Decoy)', fontsize=13)
    ax.set_aspect('equal', adjustable='box')

    plt.title(f"Estimated FDP (TDA) at T_q={t:.2f} (q={q_level})\nQuery: {target_qid}", fontsize=14, pad=15)
    ax.legend(loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()




def plot_individual_q_vs_real_fdp(df_curve_all, query_list, out_dir, prefix="", TQ=""):

    q_levels = np.arange(0.05, 1.05, 0.05)

    for qid in query_list:
        df_q = df_curve_all[df_curve_all["qid"] == qid].copy()

        if df_q.empty:
            print(f"  -> [Warning] No curve data found for query: {qid}")
            continue

        # One real-FDP-at-q vector per rep_id, then average across reps.
        rep_ids = sorted(df_q["rep_id"].unique())
        per_rep_real_fdps = []   # list of arrays, one per rep

        for rep_id in rep_ids:
            df_qr = df_q[df_q["rep_id"] == rep_id]
            real_fdps_r = []
            for q in q_levels:
                valid = df_qr[df_qr["est_fdp"] <= q]
                if len(valid) > 0:
                    best_idx = valid["threshold"].idxmin()
                    real_fdps_r.append(valid.loc[best_idx, "real_fdp"])
                else:
                    real_fdps_r.append(np.nan)
            per_rep_real_fdps.append(np.asarray(real_fdps_r, dtype=float))

        per_rep_arr = np.vstack(per_rep_real_fdps)        # (n_reps, n_q)
        mean_real_fdps = np.nanmean(per_rep_arr, axis=0)  # per-query averaged FDR

        plt.figure(figsize=(6, 6), dpi=300)
        plt.plot([0, 1.0], [0, 1.0], linestyle='--', color='gray', linewidth=2, label="Ideal (y = x)")

        # Thin rep curves to show cross-rep variation.
        for rep_id, vec in zip(rep_ids, per_rep_real_fdps):
            plt.plot(q_levels, vec, linestyle='-', linewidth=1.0, alpha=0.4,
                     color='mediumpurple', label=f"rep_{rep_id}")

        # Bold mean curve (the per-query FDR after averaging reps).
        plt.plot(q_levels, mean_real_fdps, linestyle='-', marker='.', markersize=5,
                 color='purple', linewidth=2.2, label="Mean Real FDP (over reps)")

        plt.xlabel("Target FDR Level (q)", fontsize=14)
        plt.ylabel("Real FDP", fontsize=14)
        plt.title(f"Target q vs Real FDP (per rep + mean)\nQuery: {qid}", fontsize=14)

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        plt.legend(fontsize=9, loc='lower right', ncol=2)

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_png = f"{out_dir}/{prefix}Q_vs_RealFDP_{qid}{TQ}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"  -> [OK] Saved q vs Real FDP curve for {qid}: {out_png}")



def find_bad_queries_low_q(df_curve_all, q_max=0.4, out_txt=None):

    q_levels = np.arange(0.05, q_max, 0.05)   # 0.05, 0.10, ..., 0.35

    bad_records = []

    # Group by (qid, rep_id) so a single bad rep within a query is not masked
    # (or amplified) by being conflated with the other reps' curves.
    grouped_curves = list(df_curve_all.groupby(["qid", "rep_id"]))

    for (qid, rep_id), grp in grouped_curves:
        for q in q_levels:
            valid = grp[grp["est_fdp"] <= q]

            if len(valid) > 0:
                best_idx = valid["threshold"].idxmin()
                row = valid.loc[best_idx]

                if row["real_fdp"] > q:
                    bad_records.append({
                        "qid": qid,
                        "rep_id": rep_id,
                        "q_level": q,
                        "threshold": row["threshold"],
                        "est_fdp": row["est_fdp"],
                        "real_fdp": row["real_fdp"],
                        "n_selected": row["n_selected"],
                        "total_homo": row["total_homo"],
                    })

    df_debug = pd.DataFrame(bad_records)

    if len(df_debug) > 0:
        bad_queries = sorted(df_debug["qid"].unique().tolist())
    else:
        bad_queries = []

    if out_txt is not None:
        with open(out_txt, "w") as f:
            for qid in bad_queries:
                f.write(f"{qid}\n")
        print(f"  -> [OK] Saved problematic query list to: {out_txt}")

        if len(df_debug) > 0:
            out_detail = out_txt.replace(".txt", "_details.txt")
            df_debug.to_csv(out_detail, sep="\t", index=False)
            print(f"  -> [OK] Saved problematic query details to: {out_detail}")

    print(f"  -> [DEBUG] Found {len(bad_queries)} problematic queries with q < {q_max}.")
    return bad_queries, df_debug



if __name__ == "__main__":
    
    method2type = {
        "shuf": "shuf",
        "shufpartly": "shuf",
        "shufpartly_60": "shufpartly",
        "shufpartly_40": "shufpartly",
        "shufpartly_80": "shufpartly",
        "rev": "rev",
        "mkv1": "mkv1",
        "mkv2": "mkv2",
        "dplm": "dplm",
        "PGmsk25pct": "PGmsk25pct",
        "adp201515k1_cont_incrsalpha": "adp",
        "adp201515k1_cont_incrsalpha_esm1b": "adp",
        "adp201515k1_cont_incrsalpha_protbert": "adp",
        "adp105015klist2_rand_incrsalpha_impute2L": "adpk2",
        "adp105015klist2_rand_incrsalpha_impute2L_esm1b": "adpk2",
        "adp105015klist2_rand_incrsalpha_impute2L_protbert": "adpk2",
        "adp201515k1_rand_test_opt": "adp",
        "adp201515k1_rand_test_ctl": "adp",
        "adp401515k1_rand_test_ctl": "adp",
        "adp401515k1_rand_test_opt": "adp",
        "adp601515k1_rand_test_ctl": "adp",
        "adp601515k1_rand_test_opt": "adp",
        "adp801515k1_rand_test_ctl": "adp",
        "adp801515k1_rand_test_opt": "adp",
        "copy": "copy",
        "adp_Region_W5S1P1w1": "adp",
        "adp_Region_W5S1P1w2": "adp", 
        "adp_Region_W5S1P1w5": "adp",
        "adp_Region_W5S1P1w1T0": "adp",
        "adp_Region_W5S1P1w1_var": "adp",
        "adp_Region_W5S1P1w0.5_var": "adp",
        "adp_Region_W5S1P1w3_var": "adp",
        "adp_Region_W5S1P1w0_var": "adp",
        "adp_Region_W5S1P1w0T0_var": "adp",
        "adp_Region_W5S1P1w1T0_var": "adp",
        "adp_Region_W5S1P1w1T1_var": "adp",
        "adp_Region_W5S1P1w1T2_var": "adp",
        "adp_Region_W5S1P1w0T1_var": "adp",
        "adp_Region_W5S1P1w1_v1": "adp",
        "adp_Region_W5S1P1w1_v2": "adp",
        "adp_Region_W10S1P1w1_v1": "adp",
        "adp_Region_W10S1P1w1_v2": "adp",
        "adp_Region_W5S1P1T1_sensloc": "adp",
        "adp_Region_W5S1P1T0_sensloc": "adp",
        "extended_dup": "dup",
        "extended_shuf": "shuf",
        "extended_rev": "rev",
        "extended_then_shuf": "shuf",
        "adp_Region_W5S1P1T0_extended": "adp",
        "adp_Region_W5S1P1T1_extended": "adp",
        "extended_shufpartly40": "shufpartly",
        "adp_Region_W5S1P1w1T0": "adp",
        "extended_mkv1": "mkv",
        "extended_mkv2": "mkv",
    }


    # ==========================================
    # parameters
    # ==========================================
    methods = ['plm']
    decoy_methods = ['extended_shuf']
    SEQ_NAME_QUERY = 'astral_case_study'
    SEQ_NAME_TARGET = 'astral'
    tau = 0.2
    weight_method = 'AdaptiveBell'
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    TARGET_QUERY = False

    if TARGET_QUERY:
        TQ = "_target_query"
    else:
        TQ = ""

    calibration = True
    target_queries_list = [
        "d1gy7a_",
        # add more putative ids here if you want individual per-query FDP plots
    ]


    # # DEBUG
    # qid_df = pd.read_csv(f'data/bad_queries_plm_astral_s3000_seed123_extended_shuf_q_lt_0.4_details.txt', sep='\t')
    # target_queries_list = qid_df['qid'].unique().tolist()[:10]
    
    PLOT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'FDP_plots')
    OTHER_PLOT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'plots')
    os.makedirs(PLOT_OUT_DIR, exist_ok=True)
    os.makedirs(OTHER_PLOT_OUT_DIR, exist_ok=True)
    # ==========================================
    
    for sm in methods:
        for mth_bm in decoy_methods: 

            print(f"==================================================")
            print(f"Processing {sm} with {mth_bm}")
            print(f"==================================================")
            SEARCH_METHOD = sm
            METHOD_BENCHMARK = mth_bm
 
            TYPE = method2type[mth_bm]

            path_real = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_target_noisy_{SEQ_NAME_TARGET}.txt'
            if calibration:
                path_decoy = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_calibrated_gam_{weight_method}_{SEQ_NAME_TARGET}{TQ}.txt'
            else:
                path_decoy = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_{SEQ_NAME_TARGET}.txt'

            decoy_suffix = f"_{TYPE}"

            df_merge = make_pair_table(path_real, path_decoy, decoy_suffix)
            print(f"  -> [INFO] Merged {len(df_merge)} Target-Decoy pairs.")

            df_curve_all, df_combined_all = estimate_fdr_curves_all_queries(df_merge)
            del df_merge, df_combined_all
            gc.collect()
            df_curve_all.to_csv(f"{DATA_DIR}/df_curve_all_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_{weight_method}_{SEQ_NAME_TARGET}{TQ}.txt", sep="\t", index=False)

            # check ranking
            out_fdr_curve_png = f"{PLOT_OUT_DIR}/{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_{weight_method}_Standard_Q_vs_RealFDR{TQ}.png"
            out_power_curve_png = f"{PLOT_OUT_DIR}/{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_{weight_method}_Standard_Q_vs_Power{TQ}.png"
            out_valid_curve_png = f"{PLOT_OUT_DIR}/{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{METHOD_BENCHMARK}_{weight_method}_Standard_Q_vs_ValidCount{TQ}.png" 
            
            title_suffix = f"({SEARCH_METHOD} - {METHOD_BENCHMARK})"
            plot_q_vs_real_fdr_and_discoveries(
                df_curve_all,
                out_png_fdr=out_fdr_curve_png,
                out_png_power=out_power_curve_png,
                out_png_valid=out_valid_curve_png,
                title_suffix=title_suffix,
                weight_method=weight_method,
                tau=tau,
            )
            print(f"  -> [OK] Finished plotting for {mth_bm}.")
            print("==================================================\n")


            print(f"\n==================================================")
            print(f"Plotting individual FDP curves for selected queries")
            print(f"==================================================")
            
            plot_individual_q_vs_real_fdp(
                df_curve_all=df_curve_all,
                query_list=target_queries_list,
                out_dir=PLOT_OUT_DIR,
                prefix=f"{SEARCH_METHOD}_{METHOD_BENCHMARK}_",
                TQ=TQ,
            )


            # print(f"\n==================================================")
            # print(f"Debugging problematic queries in low-q region")
            # print(f"==================================================")

            # out_bad_query_txt = f"{DATA_DIR}/bad_queries_{SEARCH_METHOD}_{SEQ_NAME}_{METHOD_BENCHMARK}_q_lt_0.4{TQ}.txt"
            # bad_queries, df_bad_debug = find_bad_queries_low_q(
            #     df_curve_all=df_curve_all,
            #     q_max=0.4,
            #     out_txt=out_bad_query_txt,
            # )


    # for target_qid in target_queries_list:
    #     out_path_scatter = f"{OTHER_PLOT_OUT_DIR}/tda_scatter_{SEARCH_METHOD}_{METHOD_BENCHMARK}_{target_qid}{TQ}.png"
    #     plot_tda_scatter_for_query(
    #         df_merge=df_merge,
    #         target_qid=target_qid,
    #         out_path=out_path_scatter,
    #         q_level=0.10,
    #     )



    
