#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_ROOT = BASE_DIR / "results_final" / "pr_curve"
OUT_DIR = BASE_DIR / "data" / "plot_data" / "pr_curve"

DECOY_METHOD = "extended_shuf"
FAMILIES = ["a.4.1.1", "c.2.1.1", "b.47.1.2"]
METHODS = ["plm", "tmvec", "dhr_postprocess"]


def combo_data_dir(family: str, method: str) -> Path:
    return SRC_ROOT / f"family_{family}" / f"{method}__{DECOY_METHOD}" / "data"


MAX_PR_POINTS: int | None = 8000


def gather_pr_lines(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src, sep="\t", usecols=["qid", "recall", "precision"])
    if MAX_PR_POINTS is None:
        return df.sort_values(["qid", "recall"]).reset_index(drop=True)
    out = []
    for qid, g in df.groupby("qid", sort=False):
        g = g.sort_values("recall")
        if len(g) > MAX_PR_POINTS:
            idx = np.unique(np.linspace(0, len(g) - 1, MAX_PR_POINTS).round().astype(int))
            g = g.iloc[idx]
        out.append(g)
    return pd.concat(out, ignore_index=True)[["qid", "recall", "precision"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for family in FAMILIES:
        for method in METHODS:
            ddir = combo_data_dir(family, method)
            q_src = ddir / "recall_precision_vs_q_per_query.tsv"
            pr_src = ddir / "pr_curve_per_query.tsv"
            if not (q_src.exists() and pr_src.exists()):
                print(f"[WARN] missing data for {family} / {method}")
                continue

            q_df = pd.read_csv(q_src, sep="\t")
            q_df.to_csv(OUT_DIR / f"pr_family_{family}_{method}.tsv", sep="\t", index=False)

            pr_df = gather_pr_lines(pr_src)
            out_pr = OUT_DIR / f"prline_{family}_{method}.tsv"
            pr_df.to_csv(out_pr, sep="\t", index=False)

            print(f"[OK] {family} / {method}: {q_df['qid'].nunique()} queries, "
                  f"{len(pr_df):,} PR points -> {out_pr.stat().st_size/1e6:.0f} MB")
            written += 1

    (OUT_DIR / "meta.json").write_text(json.dumps({
        "families": FAMILIES, "methods": METHODS, "decoy_method": DECOY_METHOD,
    }, indent=2))
    print(f"\n[DONE] wrote {written} combos -> {OUT_DIR}")


if __name__ == "__main__":
    main()
