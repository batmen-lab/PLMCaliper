#!/usr/bin/env python3

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/ColabFold/ur50_db/dhr_postprocess/iter1/msa_metrics.tsv"
FIGDIR = ROOT / "results/ColabFold/ur50_db/dhr_postprocess/iter1/figs_tmp"
EFDR = [round(0.1 * i, 1) for i in range(1, 10)]
GREENS = plt.get_cmap("Greens")
QCOL = [GREENS(0.30 + 0.62 * i / (len(EFDR) - 1)) for i in range(len(EFDR))]
BASES = [("base_alldb_iter1", "DB iter1", "#0072B2"),
         ("base_alldb_iter2", "DB iter2", "#D55E00"),
         ("base_top400k_iter2", "top400k iter2", "#CC79A7")]
BASE_GAP = 1.2  # extra spacing between the FDR group and the baseline group
FLOOR = {"msa_depth": 1.0, "msa_seconds": 0.01, "meff": 1.0}
LAB = {"msa_depth": "MSA depth (# sequences)", "msa_seconds": "MSA build time (s)",
       "meff": "Meff (# effective sequences)"}


def _box(ax, data, pos, color, width=0.45):
    ax.boxplot([data], positions=[pos], widths=width, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=color, alpha=0.9, edgecolor="0.3", lw=1.0),
               medianprops=dict(color="black", lw=1.5),
               whiskerprops=dict(color="0.45"), capprops=dict(color="0.45"))


def plot_metric(d, metric, out):
    flo = FLOOR.get(metric)
    groups = [(f"efdr_{q:.2f}".replace(".", "p"), q) for q in EFDR] + [(t, None) for t, _, _ in BASES]
    # FDR groups at 1..len(EFDR); baselines pushed right by BASE_GAP to separate the two groups.
    positions = list(range(1, len(EFDR) + 1))
    positions += [len(EFDR) + BASE_GAP + 1 + j for j in range(len(BASES))]
    allv = []
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.set_box_aspect(1)
    for i, (tag, q) in enumerate(groups):
        s = d.loc[d["tag"] == tag, metric].dropna()
        if flo is not None:
            s = s.clip(lower=flo)
        vv = s.tolist()
        if not vv:
            continue
        color = QCOL[i] if q is not None else dict((t, c) for t, _, c in BASES)[tag]
        _box(ax, vv, positions[i], color)
        allv += vv
    ax.set_yscale("log")
    if allv:
        ax.set_ylim(min(allv) / 1.6, max(allv) * 1.6)
    ax.set_xticks(positions)
    labels = [f"{q:.1f}" for q in EFDR] + [lab for _, lab, _ in BASES]
    ticklabels = ax.set_xticklabels(labels, fontsize=10)
    for tl in ticklabels[len(EFDR):]:  # tilt only the baseline labels so they fit
        tl.set_rotation(30)
        tl.set_ha("right")
        tl.set_rotation_mode("anchor")
    ax.set_xlabel("Target FDR level                    baselines", fontweight="bold", fontsize=14)
    ax.set_ylabel(LAB[metric], fontweight="bold", fontsize=15)
    ax.set_title("DHR", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.axvline(len(EFDR) + BASE_GAP / 2 + 0.5, color="0.6", ls=":", lw=1.2)
    handles = [mpatches.Patch(facecolor=QCOL[0], edgecolor="0.3", label="FDR 0.1"),
               mpatches.Patch(facecolor=QCOL[-1], edgecolor="0.3", label="FDR 0.9")]
    handles += [mpatches.Patch(facecolor=c, edgecolor="0.3", label=lab.replace("\n", " "))
                for _, lab, c in BASES]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left",
              frameon=True, edgecolor="0.7", fontsize=9)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.png')} + .pdf")


def main():
    d = pd.read_csv(SRC, sep="\t")
    n = d["query_id"].nunique()
    print(f"[load] {len(d)} rows, {n} queries")
    metrics = [m for m in ("msa_depth", "meff", "msa_seconds") if m in d.columns]
    for metric in metrics:
        plot_metric(d, metric, FIGDIR / f"ur50_{metric}_TMP")


if __name__ == "__main__":
    main()
