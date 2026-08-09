#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style  
from style import FONT_LEGEND, FONT_TITLE

import config as C


def a3m_depth(runs: Path, qid: str, tag: str) -> int:
    a = runs / qid / tag / "msa.a3m"
    if not a.exists():
        return 0
    return sum(1 for l in a.read_text().splitlines() if l.startswith(">"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dhr_postprocess")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    summary = args.summary or C.method_summary_path(args.method)
    runs = args.runs or C.method_runs_dir(args.method)
    out = args.out or (C.method_results_dir(args.method) / "invalid.pdf")

    df = pd.read_csv(summary, sep="\t")
    df = df[df["af2_status"] == "ok"]
    queries = df[df["tag"] == "vanilla"]["query_id"].unique()

    rows, pts = [], []
    for qid in queries:
        d = df[df["query_id"] == qid]
        van = d[d["tag"] == "vanilla"].iloc[0]
        fdr = d[d["tag"].str.startswith("efdr")]
        van_dep = a3m_depth(runs, qid, "vanilla")
        fdr_deps = [(int(r.n_hits), a3m_depth(runs, qid, r.tag), r.plddt)
                    for r in fdr.itertuples()]
        best_dep = max((x[1] for x in fdr_deps), default=0)
        best_pl = fdr["plddt"].max()
        rows.append(dict(qid=qid, van_dep=van_dep, fdr_dep=best_dep,
                         van_pl=van["plddt"], fdr_pl=best_pl))
        pts.extend(fdr_deps)
    r = pd.DataFrame(rows)
    P = pd.DataFrame(pts, columns=["n_hits", "depth", "plddt"])

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # (A) MSA depth vanilla vs FDR-best
    a = ax[0]
    a.scatter(r["van_dep"].clip(lower=1), r["fdr_dep"].clip(lower=1),
              s=14, color="#C44E52", edgecolors="k", linewidths=0.3, zorder=3)
    lim = [1, max(r["van_dep"].max(), 10) * 2]
    a.plot(lim, lim, "--", color="gray", lw=0.8, label="equal (y=x)")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(1, lim[1])
    a.set_xlabel("vanilla MSA depth", fontweight="bold")
    a.set_ylabel("FDR-best MSA depth", fontweight="bold")
    a.set_title("(A) FDR MSA is orders of magnitude shallower", fontweight="bold")
    a.legend(fontsize=FONT_LEGEND); a.grid(True, which="both", ls="--", alpha=0.3)

    # (B) pLDDT vanilla vs FDR-best
    b = ax[1]
    b.scatter(r["van_pl"], r["fdr_pl"], s=14, color="#4C72B0", edgecolors="k", linewidths=0.3, zorder=3)
    b.plot([0, 100], [0, 100], "--", color="gray", lw=0.8, label="equal (y=x)")
    b.set_xlim(40, 100); b.set_ylim(40, 100)
    b.set_xlabel("vanilla pLDDT", fontweight="bold")
    b.set_ylabel("FDR-best pLDDT", fontweight="bold")
    b.set_title("(B) FDR pLDDT never reaches vanilla", fontweight="bold")
    b.legend(fontsize=FONT_LEGEND); b.grid(True, ls="--", alpha=0.3)

    # (C) hits fed vs resulting depth (qjackhmmer collapse)
    c = ax[2]
    c.scatter(P["n_hits"].clip(lower=1), P["depth"].clip(lower=1),
              s=10, color="#55A868", edgecolors="k", linewidths=0.2, alpha=0.7, zorder=3)
    c.axhline(1, color="#C44E52", ls=":", lw=0.8, label="depth = 1 (query only)")
    c.plot([1, 1000], [1, 1000], "--", color="gray", lw=0.8, label="depth = hits fed")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(1, 1000); c.set_ylim(0.8, 1000)
    c.set_xlabel("# hits fed to qjackhmmer", fontweight="bold")
    c.set_ylabel("resulting MSA depth", fontweight="bold")
    c.set_title("(C) qjackhmmer collapses few remote hits to ~1", fontweight="bold")
    c.legend(fontsize=FONT_LEGEND); c.grid(True, which="both", ls="--", alpha=0.3)

    fig.suptitle(f"Invalid ({args.method}): FDR-subset MSA cannot match "
                 f"vanilla full-UR90 JackHMMER (n={len(r)} CASP13 queries)",
                 fontweight="bold", y=1.02, fontsize=FONT_TITLE)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[OK] saved {out} (+ .png)")
    print(f"  median vanilla depth: {int(r.van_dep.median())} | median FDR-best depth: {int(r.fdr_dep.median())}")
    print(f"  mean vanilla pLDDT: {r.van_pl.mean():.1f} | mean FDR-best pLDDT: {r.fdr_pl.mean():.1f}")
    print(f"  conditions collapsing to depth<=1: {(P.depth<=1).mean()*100:.0f}% of {len(P)}")


if __name__ == "__main__":
    main()
