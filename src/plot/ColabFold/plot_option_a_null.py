#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style 
from style import FONT_LEGEND, FONT_TITLE

CMAP = "plasma_r"  # yellow (low q) -> purple (high q)


def a3m_depth(runs_dir: Path, tag: str) -> float:
    p = runs_dir / tag / "msa.a3m"
    if not p.exists():
        return np.nan
    return sum(1 for line in p.read_text().splitlines() if line.startswith(">"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Option-A null-result figure (quality/time/MSA depth)")
    ap.add_argument("--table", type=Path,
                    default=Path("results/ColabFold/astral_db/dhr_postprocess/iter1/metrics.tsv"))
    ap.add_argument("--runs", type=Path,
                    default=Path("results/ColabFold/astral_db/dhr_postprocess/iter1/parallel/runs/d1a7sa_"))
    ap.add_argument("--query", type=str, default="d1a7sa_")
    ap.add_argument("--out", type=Path,
                    default=Path("data/plot_data/ColabFold/figures/astral_db/option_a_null_result_d1a7sa.pdf"))
    args = ap.parse_args()

    df = pd.read_csv(args.table, sep="\t")
    if "query_id" in df.columns:
        df = df[df["query_id"].astype(str) == args.query].copy()
    if df.empty:
        raise SystemExit(f"no rows for query {args.query} in {args.table}")
    efdr = df[df["efdr_limit"].notna()].sort_values("efdr_limit").copy()
    van = df[df["tag"] == "vanilla"].iloc[0]
    efdr["a3m_depth"] = [a3m_depth(args.runs, t) for t in efdr["tag"]]
    van_depth = a3m_depth(args.runs, "vanilla")

    q = efdr["efdr_limit"].values
    norm = mcolors.Normalize(vmin=0.1, vmax=0.9)
    sc_kw = dict(c=q, cmap=CMAP, norm=norm, edgecolors="k", linewidths=0.3, zorder=3)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # quality
    ax[0].plot(q, efdr["plddt"], "-", color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(q, efdr["plddt"], s=16, **sc_kw)
    ax[0].axhline(van["plddt"], color="#444", ls="--", lw=0.8, label="vanilla (all 15176 hits)")
    ax[0].set_ylim(80, 100)
    ax[0].set_xlabel("Target eFDR threshold q", fontweight="bold")
    ax[0].set_ylabel("Mean pLDDT", fontweight="bold")
    ax[0].set_title("quality", fontweight="bold")
    ax[0].grid(True, axis="y", ls="--", alpha=0.35)
    ax[0].legend(fontsize=FONT_LEGEND)

    # time
    ax[1].plot(q, efdr["total_seconds"], "-", color="0.6", lw=0.8, zorder=1)
    ax[1].scatter(q, efdr["total_seconds"], s=16, **sc_kw)
    ax[1].axhline(van["total_seconds"], color="#444", ls="--", lw=0.8, label="vanilla")
    ax[1].set_xlabel("Target eFDR threshold q", fontweight="bold")
    ax[1].set_ylabel("seconds (MSA build + ColabFold)", fontweight="bold")
    ax[1].set_title("time", fontweight="bold")
    ax[1].grid(True, axis="y", ls="--", alpha=0.35)
    ax[1].legend(fontsize=FONT_LEGEND)

    # MSA depth (gradient points + colorbar)
    sc = ax[2].scatter(efdr["n_hits"], efdr["a3m_depth"], s=18, **sc_kw)
    ax[2].plot([van["n_hits"]], [van_depth], marker="*", color="#444", ms=8, ls="none",
               label=f"vanilla: {int(van['n_hits'])} hits -> {int(van_depth)}")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("# hits fed to JackHMMER (log)", fontweight="bold")
    ax[2].set_ylabel("Resulting MSA depth", fontweight="bold")
    ax[2].set_title("MSA depth", fontweight="bold")
    ax[2].grid(True, ls="--", alpha=0.35)
    ax[2].legend(fontsize=FONT_LEGEND)
    cb = fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.04)
    cb.set_label("eFDR q")

    fig.suptitle(args.query, fontweight="bold", y=1.02, fontsize=FONT_TITLE)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] saved {args.out} and .png")


if __name__ == "__main__":
    main()
