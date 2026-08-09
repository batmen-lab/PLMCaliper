#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style  
from paths import experiment_dir

DATA_DIR = experiment_dir("pr_curve")

from style import FONT_LABEL, FONT_TICK, LINEWIDTH 

AXIS_FONTSIZE = FONT_LABEL
MEAN_LW = LINEWIDTH
MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = 1.0

METHOD_STYLE = {
    "plm": ("PLMSearch", "#0072B2", "o"),
    "tmvec": ("TMvec", "#E69F00", "s"),
    "dhr_postprocess": ("DHR", "#CC79A7", "^"),
}


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0 else 2.0 * precision * recall / denom


def mean_f1_by_q(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d = df.copy()
    d["f1"] = [f1(p, r) for p, r in zip(d["precision"], d["recall"])]
    g = d.groupby("q", sort=True)["f1"]
    mean = g.mean()
    counts = g.count()
    sem = g.std(ddof=1) / np.sqrt(counts.clip(lower=1))
    sem = sem.fillna(0.0)
    n_queries = int(counts.max()) if len(counts) else 0
    return (mean.index.to_numpy(dtype=float), mean.to_numpy(),
            sem.to_numpy(), n_queries)


def plot_family(family: str, series: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    handles = []
    q_ref = None
    for label, qs, mean_f1, sem_f1, color, marker in series:
        q_ref = qs if q_ref is None else q_ref
        ax.fill_between(qs, mean_f1 - sem_f1, mean_f1 + sem_f1, color=color,
                        alpha=0.18, linewidth=0, zorder=1)
        ax.plot(qs, mean_f1, color=color, linewidth=MEAN_LW, zorder=2)
        ax.plot(qs, mean_f1, linestyle="", marker=marker, markersize=MARKER_SIZE,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=MARKER_EDGE_WIDTH, zorder=3)
        handles.append(Line2D([0], [0], color=color, linewidth=MEAN_LW, marker=marker,
                              markersize=MARKER_SIZE, markerfacecolor="white",
                              markeredgecolor=color, markeredgewidth=MARKER_EDGE_WIDTH,
                              label=label))

    ax.set_box_aspect(1)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.02, 1.02)
    if q_ref is not None:
        ax.set_xticks(q_ref)
        ax.set_xticklabels([f"{q:.1f}" for q in q_ref], fontsize=FONT_TICK)
    ax.set_xlabel("Target FDR level", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Mean F1 score", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(f"Family {family}", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=handles, frameon=True, loc="center left",
              bbox_to_anchor=(1.04, 0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", default=None,
                   help="Subset of families (default: all in meta.json).")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of search methods to overlay (default: all in meta.json).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DATA_DIR / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads((args.data_dir / "meta.json").read_text())
    families = args.families or meta["families"]
    methods = args.methods or meta["methods"]

    for family in families:
        series = []
        for method in methods:
            tsv = args.data_dir / f"pr_family_{family}_{method}.tsv"
            if not tsv.exists():
                print(f"[skip] {family}/{method}: no {tsv.name}")
                continue
            qs, mean_f1, sem_f1, n_q = mean_f1_by_q(pd.read_csv(tsv, sep="\t"))
            label, color, marker = METHOD_STYLE.get(method, (method, None, "o"))
            print(f"[info] {family}/{method}: n={n_q} queries; band = mean +/- SEM")
            series.append((label, qs, mean_f1, sem_f1, color, marker))
        if series:
            plot_family(family, series, args.plot_dir / f"f1_vs_q_{family}.pdf")


if __name__ == "__main__":
    main()
