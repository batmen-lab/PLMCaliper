#!/usr/bin/env python3
from __future__ import annotations

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
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_efdr_recall import combo_id, load_query_list  # noqa: E402
from plot_family_query_pr import (  # noqa: E402
    compute_per_query_tables,
    family_method_output_dir,
    load_saved_plot_tables,
)

AXIS_FONTSIZE = 14
FIG_SIZE = (8.0, 5.5)
Q_LEVELS = np.arange(0.10, 1.00, 0.10)

MEAN_LW = 2.0
MARKER_SIZE = 8
MARKER_EDGE_WIDTH = 1.8

# Search method config: internal name → (display label, Okabe-Ito color)
# Palette: blue, orange, reddish-purple — safe for deuteranopia and protanopia
SEARCH_METHODS: list[tuple[str, str, str]] = [
    ("plm",             "PLMSearch", "#0072B2"),  # blue
    ("tmvec",           "TMvec",     "#E69F00"),  # orange
    ("dhr_postprocess", "DHR",       "#CC79A7"),  # reddish-purple
]


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def compute_f1_table(df_per: pd.DataFrame) -> pd.DataFrame:
    df = df_per.copy()
    df["f1"] = df.apply(
        lambda row: f1_from_precision_recall(float(row["precision"]), float(row["recall"])),
        axis=1,
    )
    return df


def mean_f1_by_q(df_f1: pd.DataFrame, q_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pivot = df_f1.pivot_table(index="q", columns="qid", values="f1", aggfunc="first")
    pivot_qs = np.array(pivot.index, dtype=float)
    qs, means = [], []
    for q in q_levels:
        matches = np.where(np.isclose(pivot_qs, float(q), atol=1e-9))[0]
        if matches.size == 0:
            continue
        qs.append(float(q))
        means.append(np.nanmean(pivot.iloc[matches[0]].values))
    return np.array(qs), np.array(means)


def plot_f1_vs_q_combined(
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    *,
    family_id: str,
    q_levels: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    legend_handles = []
    for label, qs, mean_f1, color in series:
        # Line first, then open circles on top so white fill breaks the line
        ax.plot(qs, mean_f1, color=color, linewidth=MEAN_LW, zorder=2)
        ax.plot(
            qs,
            mean_f1,
            linestyle="",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=MEAN_LW,
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=label,
            )
        )

    ax.set_box_aspect(1)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(q_levels)
    ax.set_xticklabels([f"{q:.1f}" for q in q_levels], fontsize=10)
    ax.set_xlabel("Target FDR level", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Mean F1 score", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_title(f"Family {family_id}", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(
        handles=legend_handles,
        fontsize=9,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 vs eFDP threshold for family queries.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute from raw calibrated files instead of reading saved TSVs.",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    FAMILY_IDS = ["a.4.1.1", "a.39.1.5", "b.47.1.2", "c.2.1.1"]
    SEQ_NAME = "astral"
    DECOY_METHOD = "extended_shuf"

    for FAMILY_ID in FAMILY_IDS:
        QUERY_LIST = PROJECT_ROOT / "data" / f"querylist_{FAMILY_ID}.txt"
        FAMILY_BASE = PROJECT_ROOT / "results_final" / "pr_curve" / f"family_{FAMILY_ID}"
        print(f"\n{'='*60}")
        print(f"[INFO] Family {FAMILY_ID}")

        series: list[tuple[str, np.ndarray, np.ndarray, str]] = []

        for search_method, display_name, color in SEARCH_METHODS:
            output_dir = family_method_output_dir(FAMILY_BASE, search_method, DECOY_METHOD)
            data_dir = output_dir / "data"
            grep_cache_dir = output_dir / "grep_cache"

            print(f"\n[INFO] {display_name} ({combo_id(search_method, DECOY_METHOD)})")

            if args.recompute:
                query_ids_set = load_query_list(QUERY_LIST)
                df_per, _ = compute_per_query_tables(
                    search_method,
                    DECOY_METHOD,
                    SEQ_NAME,
                    query_ids_set,
                    Q_LEVELS,
                    grep_cache_dir,
                )
            else:
                df_per, _ = load_saved_plot_tables(data_dir)

            df_f1 = compute_f1_table(df_per)

            f1_tsv_path = data_dir / "f1_vs_q_per_query.tsv"
            data_dir.mkdir(parents=True, exist_ok=True)
            df_f1[["qid", "q", "recall", "precision", "f1"]].sort_values(["qid", "q"]).to_csv(
                f1_tsv_path, sep="\t", index=False
            )
            print(f"[OK] {f1_tsv_path}")

            qs, means = mean_f1_by_q(df_f1, Q_LEVELS)
            series.append((display_name, qs, means, color))

        plot_f1_vs_q_combined(
            series,
            family_id=FAMILY_ID,
            q_levels=Q_LEVELS,
            output_path=FAMILY_BASE / "f1_vs_q.pdf",
        )

    print("\n[DONE]")
