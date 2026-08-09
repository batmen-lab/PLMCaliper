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
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_efdr_recall import (
    AXIS_FONTSIZE,
    DATA_SUBDIR,
    FIG_TITLE_FONTSIZE,
    PR_CURVE_LINE_WIDTH,
    Q_LEVELS,
    Q_VLINE_ALPHA,
    Q_VLINE_WIDTH,
    RESULTS_DIR,
    _find_per_combo_tsv,
    _plot_pr_curve_line,
    _plot_q_threshold_vlines,
    combo_id,
    combo_legend_text,
    run_tag,
)

FIG_SIZE = (7.0, 7.0)
COMPARE_LINE_WIDTH = 1.8
COMPARE_LINESTYLES = ["-", "--", "-.", ":"]
COMPARE_COLORS = [
    "#0072B2",
    "#E69F00",
    "#CC79A7",
    "#009E73",
    "#56B4E9",
]
FIG_SUBTITLE_Y = -0.03


def _data_dir(out_dir: Path) -> Path:
    data = out_dir / DATA_SUBDIR
    return data if data.exists() else out_dir


def find_plot_data_tsv(data_dir: Path, cid: str) -> Path | None:
    path = _find_per_combo_tsv(data_dir, "pooled_pr_plot_data", cid)
    if path is not None:
        return path
    return _find_per_combo_tsv(data_dir, "fdr_metrics", cid)


def find_pr_pooled_tsv(data_dir: Path, cid: str) -> Path | None:
    return _find_per_combo_tsv(data_dir, "pr_curve_pooled", cid)


def load_tables(
    out_dir: Path,
    search_methods: list[str],
    decoy_methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = _data_dir(out_dir)
    q_parts: list[pd.DataFrame] = []
    pr_parts: list[pd.DataFrame] = []
    for sm, dm in product(search_methods, decoy_methods):
        cid = combo_id(sm, dm)
        path = find_plot_data_tsv(data_dir, cid)
        if path is None:
            print(f"[WARN] Missing plot data TSV for {cid} in {data_dir}")
            continue
        print(f"[INFO] load {path.name}")
        q_parts.append(pd.read_csv(path, sep="\t"))
        pr_path = find_pr_pooled_tsv(data_dir, cid)
        if pr_path is not None:
            print(f"[INFO] load {pr_path.name}")
            pr_parts.append(pd.read_csv(pr_path, sep="\t"))
    if not q_parts:
        raise FileNotFoundError(
            f"No pooled_pr_plot_data_{{combo}}_*.tsv under {data_dir} for "
            f"search={search_methods}, decoy={decoy_methods}"
        )
    return pd.concat(q_parts, ignore_index=True), (
        pd.concat(pr_parts, ignore_index=True) if pr_parts else pd.DataFrame()
    )


def plot_pr_comparison(
    df_q: pd.DataFrame,
    df_pr: pd.DataFrame,
    *,
    out_path: Path,
    seq_name: str,
    decoy_methods: list[str],
) -> None:
    decoy_label = ", ".join(sorted(decoy_methods))
    combos = (
        df_q[["combo_id", "search_method", "decoy_method"]]
        .drop_duplicates()
        .sort_values(["search_method", "decoy_method"])
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    q_legend_handles: list[Line2D] = []
    for idx, row in enumerate(combos.itertuples(index=False)):
        color = COMPARE_COLORS[idx % len(COMPARE_COLORS)]
        ls = COMPARE_LINESTYLES[idx % len(COMPARE_LINESTYLES)]
        label = combo_legend_text(row.search_method, row.decoy_method)
        sub_q = df_q[df_q["combo_id"] == row.combo_id].sort_values("q")
        sub_pr = df_pr[df_pr["combo_id"] == row.combo_id] if not df_pr.empty else pd.DataFrame()
        if not sub_pr.empty:
            _plot_pr_curve_line(ax, sub_pr, color=color, linestyle=ls, label=label)
        elif not sub_q.empty:
            ax.plot(
                sub_q["mean_recall"],
                sub_q["mean_precision"],
                marker="o",
                markersize=4,
                linewidth=COMPARE_LINE_WIDTH,
                color=color,
                linestyle=ls,
                label=label,
            )
        q_legend_handles.extend(_plot_q_threshold_vlines(ax, sub_q, color=color, q_levels=Q_LEVELS))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title("Pooled PR curve (method comparison)", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    q_seen: set[float] = set()
    q_handles: list[Line2D] = []
    for h in q_legend_handles:
        q_val = float(h.get_label().split("=")[1])
        if q_val in q_seen:
            continue
        q_seen.add(q_val)
        h.set_color("0.35")
        q_handles.append(h)
    method_handles, method_labels = ax.get_legend_handles_labels()
    ax.legend(
        method_handles + q_handles,
        method_labels + [h.get_label() for h in q_handles],
        loc="best",
        fontsize=9,
        frameon=True,
        edgecolor="0.35",
    )

    fig.subplots_adjust(bottom=0.14)
    fig.text(
        0.5,
        FIG_SUBTITLE_Y,
        f"{seq_name}; decoy: {decoy_label}",
        ha="center",
        va="top",
        fontsize=FIG_TITLE_FONTSIZE - 2,
        transform=fig.transFigure,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pooled PR curves across search methods")
    p.add_argument("--seq-name", default="astra", help="Dataset tag")
    p.add_argument(
        "--search-methods",
        nargs="+",
        default=["plm", "tmvec", "dhr_postprocess"],
        help="Search backends to compare (default: plm tmvec dhr_postprocess)",
    )
    p.add_argument("--decoy-methods", nargs="+", default=["extended_shuf"])
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (RESULTS_DIR / args.seq_name)
    tag = run_tag(args.seq_name, args.search_methods, args.decoy_methods)
    df_q, df_pr = load_tables(out_dir, args.search_methods, args.decoy_methods)
    plot_pr_comparison(
        df_q,
        df_pr,
        out_path=out_dir / f"pooled_pr_curve_compare_{tag}.pdf",
        seq_name=args.seq_name,
        decoy_methods=args.decoy_methods,
    )


if __name__ == "__main__":
    main()
