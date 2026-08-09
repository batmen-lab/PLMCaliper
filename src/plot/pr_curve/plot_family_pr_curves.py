#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style 
from paths import experiment_dir

DATA_DIR = experiment_dir("pr_curve")

AXIS_FONTSIZE = 7
POOLED_FIG_SIZE = (7.2, 5.76)  
Q_LEVELS = np.round(np.arange(0.10, 1.00, 0.10), 2)
Q_CMAP_VMIN, Q_CMAP_VMAX = 0.1, 0.9
Q_STAR_SIZE = 34

Q_CMAP_NAME = "plasma"

METHOD_LABEL = {"plm": "PLMSearch", "tmvec": "TMvec", "dhr_postprocess": "DHR"}

# figure-coords layout: [plot] | gap | [colorbar] | [legend]
_PLOT_LEFT, _PLOT_RIGHT_EDGE = 0.09, 0.70
_CBAR_GAP, _CBAR_WIDTH, _CBAR_LABEL_PAD = 0.04, 0.020, 0.07


def make_q_colormap():
    cmap = plt.get_cmap(Q_CMAP_NAME)
    return cmap, Normalize(vmin=Q_CMAP_VMIN, vmax=Q_CMAP_VMAX)


def apply_recall_zero_precision_one(recall: float, precision: float) -> tuple[float, float]:
    return (0.0, 1.0) if recall <= 1e-12 else (recall, precision)


def _prepare_pr_curve(df_pr: pd.DataFrame) -> pd.DataFrame:
    sub = df_pr.sort_values("recall").copy()
    if sub.empty:
        return sub
    if not (sub["recall"] <= 1e-12).any():
        origin = sub.iloc[0].to_dict()
        origin["recall"], origin["precision"] = 0.0, 1.0
        sub = pd.concat([pd.DataFrame([origin]), sub], ignore_index=True)
    else:
        sub.loc[sub["recall"] <= 1e-12, "precision"] = 1.0
    return sub


def _plot_q_stars(ax, df_q, q_levels, cmap, norm):
    recalls, precisions, qs = [], [], []
    for q in q_levels:
        row = df_q.loc[np.isclose(df_q["q"], float(q))]
        if row.empty:
            continue
        r, p = apply_recall_zero_precision_one(
            float(row["recall"].iloc[0]), float(row["precision"].iloc[0]))
        recalls.append(r)
        precisions.append(p)
        qs.append(float(q))
    if not qs:
        return None
    return ax.scatter(recalls, precisions, c=qs, cmap=cmap, norm=norm, marker="*",
                      s=Q_STAR_SIZE, alpha=1.0, edgecolors="0.25", linewidths=0.25, zorder=3)


def _finalize(fig, ax, *, cmap, norm, q_levels, legend_handles, combo_label_text):
    box = ax.get_position()
    side = min(_PLOT_RIGHT_EDGE - _PLOT_LEFT, box.height)
    y0 = box.y0 + (box.height - side) / 2.0
    ax.set_position([_PLOT_LEFT, y0, side, side])
    ax.set_aspect("equal", adjustable="box")
    box = ax.get_position()

    cbar_left = _PLOT_RIGHT_EDGE + _CBAR_GAP
    cbar_ax = fig.add_axes([cbar_left, box.y0 + 0.08, _CBAR_WIDTH, box.height * 0.82], frameon=False)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("FDR threshold (q)", fontsize=AXIS_FONTSIZE - 2, labelpad=6)
    ticks = sorted({float(q) for q in q_levels if Q_CMAP_VMIN <= float(q) <= Q_CMAP_VMAX})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

    fig.legend(handles=legend_handles, loc="center left",
               bbox_to_anchor=(cbar_left + _CBAR_WIDTH + _CBAR_LABEL_PAD, 0.5),
               bbox_transform=fig.transFigure, fontsize=6, frameon=True,
               title=combo_label_text, title_fontsize=6)


def plot_family_method(family: str, method: str, df_pr: pd.DataFrame, df_q: pd.DataFrame,
                       out_path: Path) -> None:
    cmap, norm = make_q_colormap()
    fig, ax = plt.subplots(figsize=POOLED_FIG_SIZE)
    fig.subplots_adjust(left=_PLOT_LEFT, right=0.97, top=0.90, bottom=0.11)

    n = 0
    for qid in df_pr["qid"].unique():
        sub = _prepare_pr_curve(df_pr.loc[df_pr["qid"] == qid])
        sub_q = df_q.loc[df_q["qid"] == qid]
        if sub.empty or sub_q.empty:
            continue
        ax.plot(sub["recall"], sub["precision"], color="0.55", alpha=0.28,
                linewidth=0.9, zorder=1)
        _plot_q_stars(ax, sub_q.sort_values("q"), Q_LEVELS, cmap, norm)
        n += 1

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel("Recall", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(f"Family {family}: all queries (n={n})", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_box_aspect(1)

    handles = [
        Line2D([0], [0], color="0.55", alpha=0.28, linewidth=0.9, label="Per-query PR curve"),
        Line2D([0], [0], marker="*", linestyle="", markersize=7, markeredgewidth=0.25,
               markerfacecolor=cmap(norm(0.5)), markeredgecolor="0.25", label="FDR thresholds (stars)"),
    ]
    _finalize(fig, ax, cmap=cmap, norm=norm, q_levels=Q_LEVELS, legend_handles=handles,
              combo_label_text=f"{METHOD_LABEL.get(method, method)} — family {family}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", default=None,
                   help="Subset of families (default: all in meta.json).")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of search methods (default: all in meta.json).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DATA_DIR / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads((args.data_dir / "meta.json").read_text())
    families = args.families or meta["families"]
    methods = args.methods or meta["methods"]

    for family in families:
        for method in methods:
            pr_tsv = args.data_dir / f"prline_{family}_{method}.tsv"
            q_tsv = args.data_dir / f"pr_family_{family}_{method}.tsv"
            if not (pr_tsv.exists() and q_tsv.exists()):
                print(f"[skip] {family}/{method}: missing data")
                continue
            df_pr = pd.read_csv(pr_tsv, sep="\t")
            df_q = pd.read_csv(q_tsv, sep="\t")
            plot_family_method(family, method, df_pr, df_q,
                               args.plot_dir / f"pr_curves_{family}_{method}.pdf")


if __name__ == "__main__":
    main()
