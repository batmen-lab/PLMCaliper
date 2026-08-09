#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style  

EFDR = [round(0.1 * i, 1) for i in range(1, 10)]


def _box(ax, vals, pos, color):
    ax.boxplot(vals, positions=pos, widths=0.55, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color, linewidth=0.8),
               medianprops=dict(color="black", linewidth=1.0),
               whiskerprops=dict(color=color, linewidth=0.6),
               capprops=dict(color=color, linewidth=0.6),
               flierprops=dict(marker="o", markerfacecolor=color, markeredgecolor="none",
                               alpha=0.25, markersize=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.summary, sep="\t")
    ok = df[df["af2_status"] == "ok"] if "af2_status" in df else df
    efdr = ok[ok["efdr_limit"].notna()]
    van = ok[ok["tag"] == "vanilla"]

    metrics = [("msa_depth", "MSA depth", "#0173B2", True),
               ("plddt", "Mean pLDDT", "#029E73", False),
               ("rmsd", "RMSD vs native (Å)", "#C44E52", False),
               ("tm_score", "TM-score vs native", "#9467BD", False)]
    metrics = [m for m in metrics if m[0] in ok.columns and ok[m[0]].notna().any()]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    pos = np.arange(1, len(EFDR) + 1)
    vpos = len(EFDR) + 1.5

    for ax, (col, ylab, color, logy) in zip(axes, metrics):
        vals = [efdr[efdr["efdr_limit"] == q][col].dropna().tolist() for q in EFDR]
        _box(ax, vals, pos, color)
        if van[col].notna().any():
            _box(ax, [van[col].dropna().tolist()], [vpos], "#DE8F05")
            ax.axhline(van[col].dropna().median(), color="#DE8F05", ls=":", alpha=0.6)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylab, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        nq = max((len(v) for v in vals), default=0)
        if ax is axes[0]:
            ax.set_title(f"ASTRAL: MSA/structure quality vs eFDR (n<={nq} queries)", fontweight="bold")

    axes[-1].set_xticks(list(pos) + [vpos])
    axes[-1].set_xticklabels([f"{q:.1f}" for q in EFDR] + ["vanilla"])
    axes[-1].set_xlabel("Target eFDR threshold", fontweight="bold")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] saved {args.out} (+ .png)")


if __name__ == "__main__":
    main()
