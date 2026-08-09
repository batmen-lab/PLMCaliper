#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
HIST_DIR = PROJECT / "data" / "plot_data" / "core" / "hist_figures"

COLOR_NONHOMO = "#4C72B0"
COLOR_DECOY = "#7F7F7F"
COLOR_CALIB = "#DE8F05"
COLOR_HOMO = "#D62728"
COLOR_TARGET = "#4C72B0"


def load(qid: str, method: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(HIST_DIR / qid / "data" / f"{method}_{kind}.tsv", sep="\t")


def plot_method(qid: str, method: str, bins: int = 60,
                stat: str = "count", log_y: bool = False,
                homolog_style: str = "dist", suffix: str = "") -> None:
    real = load(qid, method, "real")
    decoy = load(qid, method, "decoy")
    calib = load(qid, method, "calib")

    homo = real.loc[real["homo_type"].isin([1, 2]), "score"].to_numpy(dtype=float)
    nonhomo = real.loc[real["homo_type"] == -1, "score"].to_numpy(dtype=float)
    d_raw = decoy["score"].to_numpy(dtype=float)
    d_cal = calib["score"].to_numpy(dtype=float)

    allq = np.concatenate([a for a in (nonhomo, d_raw, d_cal, homo) if len(a)])
    edges = np.linspace(float(np.min(allq)), float(np.max(allq)), bins)

    density = stat == "density"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.hist(nonhomo, bins=edges, density=density, histtype="stepfilled", alpha=0.35,
            color=COLOR_NONHOMO, label=f"Non-Homolog (n={len(nonhomo)})")
    if homolog_style == "dist":
        ax.hist(homo, bins=edges, density=density, histtype="stepfilled", alpha=0.45,
                color=COLOR_HOMO, label=f"Homolog (n={len(homo)})")
    ax.hist(d_raw, bins=edges, density=density, histtype="step", linewidth=1.8,
            color=COLOR_DECOY, label=f"Decoy (n={len(d_raw)})")
    ax.hist(d_cal, bins=edges, density=density, histtype="step", linewidth=1.8,
            color=COLOR_CALIB, label=f"Calibrated decoy (n={len(d_cal)})")
    if homolog_style == "vline":
        for i, s in enumerate(homo):
            ax.axvline(s, color=COLOR_HOMO, linestyle="--", linewidth=0.8, alpha=0.75,
                       label=f"Homolog (n={len(homo)})" if i == 0 else None)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)
    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density" if density else "Count", fontsize=12, fontweight="bold")
    ax.set_title(qid, fontsize=13, fontweight="bold")
    ax.set_box_aspect(1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
    out = HIST_DIR / qid / f"{method}_hist{suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}  (homo={len(homo)} nonhomo={len(nonhomo)} decoy={len(d_raw)} calib={len(d_cal)})")


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_cdf_method(qid: str, method: str, suffix: str = "",
                    split_target: bool = False, mark_homolog: bool = False) -> None:
    real = load(qid, method, "real")
    decoy = load(qid, method, "decoy")
    calib = load(qid, method, "calib")

    homo = real.loc[real["homo_type"].isin([1, 2]), "score"].to_numpy(dtype=float)
    nonhomo = real.loc[real["homo_type"] == -1, "score"].to_numpy(dtype=float)
    target = real.loc[real["homo_type"].isin([1, 2, -1]), "score"].to_numpy(dtype=float)
    d_raw = decoy["score"].to_numpy(dtype=float)
    d_cal = calib["score"].to_numpy(dtype=float)

    if split_target:
        curves = [
            (nonhomo, COLOR_NONHOMO, f"Non-Homolog (n={len(nonhomo)})"),
            (homo, COLOR_HOMO, f"Homolog (n={len(homo)})"),
            (d_raw, COLOR_DECOY, f"Decoy (n={len(d_raw)})"),
            (d_cal, COLOR_CALIB, f"Calibrated decoy (n={len(d_cal)})"),
        ]
    else:
        curves = [
            (target, COLOR_TARGET, f"Target (n={len(target)})"),
            (d_raw, COLOR_DECOY, f"Decoy (n={len(d_raw)})"),
            (d_cal, COLOR_CALIB, f"Calibrated decoy (n={len(d_cal)})"),
        ]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for arr, color, lab in curves:
        x, y = ecdf(arr)
        ax.plot(x, y, color=color, linewidth=1.8, label=lab)

    if mark_homolog and len(homo):
        tx, _ = ecdf(target)
        homo_cdf = np.searchsorted(tx, homo, side="right") / len(tx)
        ax.scatter(homo, homo_cdf, marker="*", s=80, color=COLOR_HOMO,
                   edgecolors="black", linewidths=0.4, zorder=6,
                   label=f"Homolog (n={len(homo)})")

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("CDF", fontsize=12, fontweight="bold")
    ax.set_title(qid, fontsize=13, fontweight="bold")
    ax.set_box_aspect(1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
    out = HIST_DIR / qid / f"{method}_cdf{suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}  (target={len(target)} decoy={len(d_raw)} calib={len(d_cal)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qid", required=True)
    p.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    p.add_argument("--stat", choices=("count", "density"), default="count")
    p.add_argument("--log-y", dest="log_y", action="store_true", default=False)
    p.add_argument("--no-log-y", dest="log_y", action="store_false")
    p.add_argument("--homolog-style", choices=("dist", "vline"), default="dist")
    p.add_argument("--kind", choices=("hist", "cdf"), default="hist")
    p.add_argument("--split-target", action="store_true")
    p.add_argument("--mark-homolog", action="store_true")
    p.add_argument("--suffix", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for m in args.methods:
        if args.kind == "cdf":
            plot_cdf_method(args.qid, m, suffix=args.suffix,
                            split_target=args.split_target, mark_homolog=args.mark_homolog)
        else:
            plot_method(args.qid, m, stat=args.stat, log_y=args.log_y,
                        homolog_style=args.homolog_style, suffix=args.suffix)


if __name__ == "__main__":
    main()
