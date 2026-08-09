#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

import config as C


def a3m_depth(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith(">"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check UR50 MSA depths")
    ap.add_argument("--method", default="dhr_postprocess")
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--thin-depth", type=int, default=1)
    args = ap.parse_args()

    runs = args.runs_dir or C.method_runs_dir(args.method)
    rows = []
    if not runs.exists():
        print(f"[WARN] runs dir missing: {runs}")
        return

    for qdir in sorted(p for p in runs.iterdir() if p.is_dir()):
        for tdir in sorted(p for p in qdir.iterdir() if p.is_dir()):
            a3m = tdir / "msa.a3m"
            rows.append({
                "method": args.method,
                "query_id": qdir.name,
                "tag": tdir.name,
                "msa_depth": a3m_depth(a3m),
                "has_a3m": a3m.exists(),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[INFO] no run directories yet under {runs}")
        return

    n_thin = int((df["msa_depth"] <= args.thin_depth).sum())
    print(f"[INFO] runs={len(df)}, queries={df['query_id'].nunique()}, depth<={args.thin_depth}: {n_thin}")
    print(df.sort_values(["query_id", "tag"]).to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False)
        print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
