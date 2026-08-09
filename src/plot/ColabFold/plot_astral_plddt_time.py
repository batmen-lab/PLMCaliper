#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_SRC = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
ASTRAL_SRC = ROOT / "src" / "ColabFold" / "astral"
sys.path.insert(0, str(PLOT_SRC))
sys.path.insert(0, str(ASTRAL_SRC))
import style 
from style import FONT_TICK, save_figure

from config import ASTRAL_BATCH_DIR, EFDR_THRESHOLDS, VANILLA_TAG
from stats import build_summary, load_table


def star(p: float) -> str:
    if not (p == p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _box(ax, values, positions, color):
    ax.boxplot(
        values, positions=positions, widths=0.55, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color, linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color=color, linewidth=0.6),
        capprops=dict(color=color, linewidth=0.6),
        flierprops=dict(marker="o", markerfacecolor=color, markeredgecolor="none",
                        alpha=0.25, markersize=2),
    )


def plot(df: pd.DataFrame, summary: pd.DataFrame, q_levels: list[float],
         time_col: str, out_path: Path, title: str) -> None:
    q_levels = sorted(q_levels)
    efdr_df = df[df["efdr_limit"].notna()]
    van_df = df[df["tag"] == VANILLA_TAG]
    pmap = {round(r.q, 1): r.p_plddt for r in summary.itertuples(index=False)}
    pmap_t = {round(r.q, 1): r.p_time for r in summary.itertuples(index=False)}

    fig, (ax_p, ax_t) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    pos = np.arange(1, len(q_levels) + 1)
    van_pos = len(q_levels) + 1.5

    # ---- pLDDT ----
    pl_vals = [efdr_df[efdr_df["efdr_limit"] == q]["plddt"].dropna().tolist() for q in q_levels]
    _box(ax_p, pl_vals, pos, "#029E73")
    if van_df["plddt"].notna().any():
        _box(ax_p, [van_df["plddt"].dropna().tolist()], [van_pos], "#DE8F05")
        ax_p.axhline(van_df["plddt"].dropna().mean(), color="#DE8F05", linestyle=":", alpha=0.6)
    y_top = ax_p.get_ylim()[1]
    for x, q, vals in zip(pos, q_levels, pl_vals):
        if vals:
            ax_p.text(x, max(vals) + 1.5, star(pmap.get(round(q, 1), np.nan)),
                      ha="center", fontsize=FONT_TICK)
    ax_p.set_ylabel("Mean pLDDT", fontweight="bold")
    ax_p.set_title(title, fontweight="bold")
    ax_p.grid(True, axis="y", linestyle="--", alpha=0.35)

    # ---- time ----
    t_vals = [efdr_df[efdr_df["efdr_limit"] == q][time_col].dropna().tolist() for q in q_levels]
    _box(ax_t, t_vals, pos, "#0173B2")
    if van_df[time_col].notna().any():
        _box(ax_t, [van_df[time_col].dropna().tolist()], [van_pos], "#DE8F05")
        ax_t.axhline(van_df[time_col].dropna().mean(), color="#DE8F05", linestyle=":", alpha=0.6)
    for x, q, vals in zip(pos, q_levels, t_vals):
        if vals:
            ax_t.text(x, max(vals), star(pmap_t.get(round(q, 1), np.nan)),
                      ha="center", va="bottom", fontsize=FONT_TICK)

    xticks = list(pos) + [van_pos]
    xlabels = [f"{q:.1f}" for q in q_levels] + ["vanilla"]
    ax_t.set_xticks(xticks)
    ax_t.set_xticklabels(xlabels, fontsize=FONT_TICK)
    ax_t.set_xlabel("Target eFDR threshold", fontweight="bold")
    ax_t.set_ylabel(f"Prediction time: {time_col} (s)", fontweight="bold")
    ax_t.grid(True, axis="y", linestyle="--", alpha=0.35)

    for q, vals in zip(q_levels, pl_vals):
        print(f"  q={q:.1f}: n_queries={len(vals):,}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(out_path)
    plt.close(fig)
    print(f"[OK] Saved boxplot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ASTRAL_BATCH_DIR / "metrics.tsv")
    parser.add_argument("--out", type=Path, default=ASTRAL_BATCH_DIR / "plddt_time_vs_efdr_boxplot.pdf")
    parser.add_argument("--ttest-out", type=Path, default=ASTRAL_BATCH_DIR / "paired_ttest_summary.tsv")
    parser.add_argument("--time-col", type=str, default="total_seconds",
                        choices=["total_seconds", "predict_seconds", "msa_seconds"])
    parser.add_argument("--title", type=str,
                        default="ASTRAL: structure quality & time vs eFDR (vs vanilla)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q-levels", nargs="+", type=float, default=EFDR_THRESHOLDS)
    args = parser.parse_args()

    df = load_table(args.summary)
    q_levels = sorted(set(args.q_levels))
    summary = build_summary(df, q_levels, args.time_col, args.alpha)
    args.ttest_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.ttest_out, sep="\t", index=False)
    plot(df, summary, q_levels, args.time_col, args.out, args.title)


if __name__ == "__main__":
    main()
