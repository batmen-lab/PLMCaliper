import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PLOT = ROOT / "data/plot_data/ColabFold"
SEQID, SAMPLE = 0.8, 3000
RNG = np.random.default_rng(0)


def a3m_matrix(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    seqs, buf, in_seq = [], [], False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if in_seq:
                seqs.append("".join(buf))
            buf, in_seq = [], True
        else:
            buf.append("".join(c for c in line.strip() if c == "-" or c.isupper()))
    if in_seq:
        seqs.append("".join(buf))
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    L = len(seqs[0])
    seqs = [s for s in seqs if len(s) == L]
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)


def meff(path: Path) -> float:
    M = a3m_matrix(path)
    if M is None:
        return 0.0
    N, L = M.shape
    if N == 1:
        return 1.0
    m = N if N <= SAMPLE else SAMPLE
    idx = np.arange(N) if N <= SAMPLE else RNG.choice(N, m, replace=False)
    Ms = M[idx]
    thr = SEQID * L
    ic = np.zeros((m, N), dtype=np.float32)
    for s in np.unique(M):
        ic += (Ms == s).astype(np.float32) @ (M == s).astype(np.float32).T
    nsim = (ic >= thr).sum(axis=1)
    return float(N * np.mean(1.0 / nsim))


def find_runs(dataset: str, method: str) -> Path | None:
    base = ROOT / f"results/ColabFold/{dataset}/{method}/iter1"
    for cand in (base / "runs", base / "parallel" / "runs"):
        if cand.is_dir():
            return cand
    return None


def process(dataset: str, method: str) -> bool:
    tbl = PLOT / dataset / f"metrics_{method}.tsv"
    if not tbl.exists():
        print(f"[skip] {dataset}/{method}: no plot table")
        return False
    runs = find_runs(dataset, method)
    if runs is None:
        print(f"[skip] {dataset}/{method}: no runs/ a3m dir")
        return False
    df = pd.read_csv(tbl, sep="\t")
    print(f">>> {dataset}/{method}: {len(df)} rows <- {runs}", flush=True)
    vals = []
    for k, (q, t) in enumerate(zip(df["query_id"], df["tag"]), 1):
        vals.append(round(meff(runs / str(q) / str(t) / "msa.a3m"), 2))
        if k % 200 == 0:
            print(f"    {k}/{len(df)}", flush=True)
    df["meff"] = vals
    df.to_csv(tbl, sep="\t", index=False)
    fdr = df[df["efdr_limit"].notna()]
    print(f"[OK] {dataset}/{method}: meff added (median by tag): "
          f"{fdr.groupby('efdr_limit')['meff'].median().round(0).to_dict()}", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ur50_db", "astral_db"])
    ap.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr_postprocess"])
    args = ap.parse_args()
    ok = 0
    for ds in args.datasets:
        for m in args.methods:
            ok += process(ds, m)
    print(f"\n[DONE] meff added to {ok} table(s)")


if __name__ == "__main__":
    main()
